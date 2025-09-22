# engine.py
import math
from typing import Tuple, List, Dict, Optional
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F


class Engine:
    """
    Real-time two-stage inference engine with three modes:
      - 'calib' : always run calibration (Stage-1 + 3 triangle crops with Stage-2s)
      - 'fast'  : reuse last (x,y) if visible; calibrate every K frames (fast_calib_interval)
      - 'smart' : reuse last (x,y) but trigger calibration on four robust conditions

    Agreement rule (visibility):
      - Run the SAME crop through two different Stage-2 models.
      - If their predicted xy differ by < agree_thresh (in crop-normalized coords), deem visible (vis=1).
      - Use MAIN model's xy for the final coordinate.

    Returned triplet is GLOBAL normalized coords in the full frame: (x, y, vis) with vis ∈ {0,1}.
    """
    def __init__(
            self,
            stage1_model,  # Stage1ModelBiFPN (expects [1,4,3,300,300], positions [1,30,3])
            stage2_main,  # your primary Stage-2 (expects crops)
            stage2_support,  # the support Stage-2 (expects same crops)
            frame_hw: Tuple[int, int] = (1080, 1920),  # (H, W) of full frames
            triangle_radius: float = 0.06,  # normalized radius for triangle centers
            agree_thresh: float = 0.05,  # Stage-2 agreement distance (crop-norm) to call visible
            jump_thresh: float = 0.03,  # trigger if |p3 - p2| >= jump_thresh (global-norm)
            angle_thresh: float = math.pi / 12,  # trigger if angle(d1,d2) >= angle_thresh
            vis_thresh: float = 0.65, # Visibility threshold
            invisible_trigger_len: int = 8,  # trigger if last N positions are all invisible
            mode: str = "smart",  # 'calib' | 'fast' | 'smart'
            fast_calib_interval: int = 30,  # for mode='fast', force calibration every K frames
            device: str = "cuda",
            use_amp: bool = True,
    ):
        assert mode in ("calib", "fast", "smart")
        self.stage1 = stage1_model
        self.s2_main = stage2_main
        self.s2_supp = stage2_support

        self.H, self.W = frame_hw
        self.r = triangle_radius
        self.agree_thresh = float(agree_thresh)
        self.jump_thresh = float(jump_thresh)
        self.angle_thresh = float(angle_thresh)
        self.vis_thresh = float(vis_thresh)
        self.inv_len = int(invisible_trigger_len)
        self.mode = mode
        self.fast_calib_interval = int(fast_calib_interval)

        self.device = device
        self.use_amp = use_amp

        # past full frames (oldest .. newest), each [3,H,W] 0..1 float on device
        self._zeros_full = torch.zeros(3, self.H, self.W, device=self.device)
        self._past_full: List[Optional[torch.Tensor]] = [None, None, None]

        # frame index
        self._t = 0
        self._last_n_invisible_trigger_frame = 0

        # For detecting whether the current crop is in the background
        self.is_background = False

        # For debugging purposes only
        self.big_angle = 0
        self.big_jump = 0
        self.freshly_invisible = 0

        # perf: let cuDNN pick best algorithms for fixed shapes
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # --------------------------------------------------------------------------
    # -------------------------- Basic tensor helpers --------------------------
    # --------------------------------------------------------------------------

    @torch.inference_mode()
    def _to_torch_chw01(self, rgb_uint8: np.ndarray) -> torch.Tensor:
        """HWC uint8 -> CHW float32 in [0,1]."""
        t = torch.from_numpy(rgb_uint8).to(self.device, non_blocking=True) # [H, W, 3]
        # if t.ndim == 3 and t.shape[2] == 3:
        t = t.permute(2, 0, 1).contiguous()  # [3,H,W]
        return t.to(torch.float32).div_(255.0)

    @torch.inference_mode()
    def _downscale_300(self, chw: torch.Tensor) -> torch.Tensor:
        """[3,H,W] -> [3,300,300] for Stage-1."""
        return F.interpolate(chw.unsqueeze(0), size=(300, 300), mode="bilinear", align_corners=False).squeeze(0)

    @torch.inference_mode()
    def _triangle_centers(self, s1_xy: torch.Tensor) -> List[Tuple[float, float]]:
        """Equilateral triangle around Stage-1 xy (global-norm)."""
        cx, cy = float(s1_xy[0].item()), float(s1_xy[1].item())
        r = self.r
        return [
            (cx + r, cy),
            (cx - 0.5 * r, cy + (math.sqrt(3) / 2) * r),
            (cx - 0.5 * r, cy - (math.sqrt(3) / 2) * r),
        ]

    @torch.inference_mode()
    def _clamped_corner_from_center(self, cx_norm: float, cy_norm: float) -> Tuple[int, int]:
        """Clamp center so a 224×224 crop fully fits; return top-left corner (x0,y0) in pixels."""
        half = 112
        cx_px = int(round(cx_norm * self.W))
        cy_px = int(round(cy_norm * self.H))
        cx_px = max(half, min(cx_px, self.W - half))
        cy_px = max(half, min(cy_px, self.H - half))
        x0 = int(round(cx_px - half))
        y0 = int(round(cy_px - half))
        return x0, y0

    @torch.inference_mode()
    def _crop224(self, chw_full: torch.Tensor, x0: int, y0: int) -> torch.Tensor:
        """Exact 224×224 tensor crop [3,224,224] from a full CHW image."""
        return chw_full[:, y0:y0 + 224, x0:x0 + 224].contiguous()

    @torch.inference_mode()
    def _adjust_positions_to_crop(self, pos30: torch.Tensor, x0: int, y0: int) -> torch.Tensor:
        """
        pos30: [30,3] global normalized; return [30,3] in crop coords (0..1), vis=0 if out.
        """
        W, H = float(self.W), float(self.H)
        xy = pos30[:, :2].clone()
        vis = pos30[:, 2].clone()
        abs_px = torch.stack([xy[:, 0] * W, xy[:, 1] * H], dim=1)  # [30,2]
        in_x = (abs_px[:, 0] >= x0) & (abs_px[:, 0] < x0 + 224)
        in_y = (abs_px[:, 1] >= y0) & (abs_px[:, 1] < y0 + 224)
        inside = in_x & in_y & (vis > 0.5)
        xy[:, 0] = (abs_px[:, 0] - x0) / 224.0
        xy[:, 1] = (abs_px[:, 1] - y0) / 224.0
        xy[~inside] = 0.0
        vis = torch.where(inside, torch.ones_like(vis), torch.zeros_like(vis))
        return torch.cat([xy, vis.unsqueeze(1)], dim=1)  # [30,3]

    @torch.inference_mode()
    def _crop_pred_to_global_norm(self, corner_px: Tuple[int, int], xy_crop: Tuple[float, float]) -> Tuple[
        float, float]:
        """Map crop-norm (u,v) back to global-norm (x,y)."""
        x0, y0 = corner_px
        u, v = xy_crop
        px = x0 + float(u) * 224.0
        py = y0 + float(v) * 224.0
        return px / self.W, py / self.H

    # --------------------------------------------------------------------------
    # ------------------------ Stage-1 / Stage-2 wrappers ----------------------
    # --------------------------------------------------------------------------

    @torch.inference_mode()
    def _stage1_infer_xy(self, cur_full: torch.Tensor, past_full_list: List[Optional[torch.Tensor]],
                         pos30: torch.Tensor) -> torch.Tensor:
        """
        cur_full: [3,H,W]
        past_full_list: [p_old, p_mid, p_new] each [3,H,W] or None
        pos30: [30,3] global-norm
        returns: [2] in [0,1]^2 (global-norm)
        """
        cur300 = self._downscale_300(cur_full).unsqueeze(0)  # [1,3,300,300]
        # cur300 = cur300.contiguous(memory_format=torch.channels_last)
        # left-pad partial history with black to length 3 (oldest..newest)
        avail = [p for p in past_full_list if p is not None]
        padded = [self._zeros_full] * (3 - len(avail)) + avail
        past300 = torch.stack([self._downscale_300(p) for p in padded], dim=0)  # [3,3,300,300]
        past300 = past300.unsqueeze(0).contiguous()  # [1,3,3,300,300]
        pos = pos30.unsqueeze(0)  # [1,30,3]

        out = self.stage1({
            "current": cur300,
            "past": past300,
            "positions": pos
        })

        return out["xy"].squeeze(0)  # [2]

    # @torch.inference_mode()
    # def _run_stage2_at_center(
    #         self,
    #         center_norm: Tuple[float, float],
    #         cur_full: torch.Tensor,
    #         past_full_list: List[Optional[torch.Tensor]],
    #         pos30: torch.Tensor,
    # ) -> Dict:
    #     """
    #     Create crops at 'center_norm', run both Stage-2 models, return their crop-norm xy and meta.
    #     """
    #     x0, y0 = self._clamped_corner_from_center(center_norm[0], center_norm[1])
    #     # current crop
    #     cur_crop = self._crop224(cur_full, x0, y0).unsqueeze(0)  # [1,3,224,224]
    #     # partially-filled past crops (oldest..newest, fill gaps with black)
    #     filled = [p if p is not None else self._zeros_full for p in past_full_list]
    #     past_crops = torch.stack([self._crop224(pf, x0, y0) for pf in filled], dim=0).unsqueeze(0)  # [1,3,3,224,224]
    #     pos_crop = self._adjust_positions_to_crop(pos30, x0, y0).unsqueeze(0)  # [1,30,3]
    #     # forward both Stage-2s
    #     if self.use_amp:
    #         with torch.autocast(self.device, dtype=torch.float16):
    #             out_main = self.s2_main(cur_crop, past_crops, pos_crop)
    #             out_supp = self.s2_supp(cur_crop, past_crops, pos_crop)
    #     else:
    #         out_main = self.s2_main(cur_crop, past_crops, pos_crop)
    #         out_supp = self.s2_supp(cur_crop, past_crops, pos_crop)
    #
    #     xy_m = out_main["xy"].squeeze(0)  # [2]
    #     xy_s = out_supp["xy"].squeeze(0)  # [2]
    #
    #     vis_prob = torch.sigmoid(out_main["vis_logit"].squeeze(0).to(torch.float32)).item()
    #
    #     # agreement distance in crop-norm
    #     dist = float(torch.linalg.norm(xy_m - xy_s).item())
    #     return {
    #         "corner_px": (x0, y0),
    #         "xy_main": (float(xy_m[0].item()), float(xy_m[1].item())),
    #         "xy_supp": (float(xy_s[0].item()), float(xy_s[1].item())),
    #         "agree_dist": dist,
    #         "vis_prob": vis_prob,
    #     }

    @torch.inference_mode()
    def _build_crops_batch(
            self,
            centers_norm: List[Tuple[float, float]],
            cur_full: torch.Tensor,
            past_full_list: List[Optional[torch.Tensor]],
            pos30: torch.Tensor,
    ):
        """
        Create batched inputs for N centers:
          returns:
            cur_crops : [N,3,224,224]  (NHWC-ready)
            past_crops: [N,3,3,224,224]
            pos_crops : [N,30,3]
            corners   : list[(x0,y0)] len N
        """
        corners = []
        cur_crops = []
        past_crops_list = []
        pos_crops = []

        filled = [p if p is not None else self._zeros_full for p in past_full_list]  # len 3

        for cx, cy in centers_norm:
            x0, y0 = self._clamped_corner_from_center(cx, cy)
            corners.append((x0, y0))
            cur_crops.append(self._crop224(cur_full, x0, y0))
            past_crops_list.append(torch.stack([self._crop224(pf, x0, y0) for pf in filled], dim=0))
            pos_crops.append(self._adjust_positions_to_crop(pos30, x0, y0))

        cur_crops = torch.stack(cur_crops, dim=0).contiguous()  # [N,3,224,224]
        past_crops = torch.stack(past_crops_list, dim=0).contiguous()  # [N,3,3,224,224]
        pos_crops = torch.stack(pos_crops, dim=0)  # [N,30,3]
        return cur_crops, past_crops, pos_crops, corners

    @torch.inference_mode()
    def _run_stage2_batch(
            self,
            cur_crops: torch.Tensor,  # [N,3,224,224]
            past_crops: torch.Tensor,  # [N,3,3,224,224]
            pos_crops: torch.Tensor,  # [N,30,3]
    ):
        """
        Always run BOTH Stage-2 models in a single batched call.
        Returns dict:
          xy_main   : [N,2]
          xy_supp   : [N,2]
          agree_dist: [N]  (||xy_main - xy_supp||_2)
          vis_main  : [N]  (if main model returns 'vis_logit'; else None)
        """
        # Forward both models in AMP for convs; keep any sigmoid in fp32 inside the model
        # if self.use_amp:
        #     with torch.autocast(self.device, dtype=torch.float16):
        #         out_main = self.s2_main(cur_crops, past_crops, pos_crops)
        #         out_supp = self.s2_supp(cur_crops, past_crops, pos_crops)
        # else:
        #     out_main = self.s2_main(cur_crops, past_crops, pos_crops)
        #     out_supp = self.s2_supp(cur_crops, past_crops, pos_crops)
        out_main = self.s2_main({
            "current_crop": cur_crops,
            "past_crops": past_crops,
            "positions": pos_crops
        })
        out_supp = self.s2_supp({
            "current_crop": cur_crops,
            "past_crops": past_crops,
            "positions": pos_crops
        })

        xy_main = out_main["xy"]  # [N,2]
        xy_supp = out_supp["xy"]  # [N,2]
        agree_dist = torch.linalg.norm(xy_main - xy_supp, dim=1)  # [N]

        # optional main visibility
        # vis_main = None
        # if isinstance(out_main, dict) and ("vis_logit" in out_main):
        # compute sigmoid in fp32 for stability
        vis_main = torch.sigmoid(out_main["vis_logit"].to(torch.float32)).view(-1)  # [N]

        return {"xy_main": xy_main, "xy_supp": xy_supp, "agree_dist": agree_dist, "vis_main": vis_main}

    # --------------------------------------------------------------------------
    # ------------------------------ Triggers ----------------------------------
    # --------------------------------------------------------------------------

    @staticmethod
    def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        a = float(np.linalg.norm(v1))
        b = float(np.linalg.norm(v2))
        if a < 1e-8 or b < 1e-8:
            return 0.0
        cos_t = float(np.clip(np.dot(v1, v2) / (a * b), -1.0, 1.0))
        return float(math.acos(cos_t))

    @staticmethod
    def _last_two(positions_deque: deque) -> Tuple[Tuple[float, float, int], Tuple[float, float, int]]:
        p1 = positions_deque[-2]  # (x,y,vis)
        p2 = positions_deque[-1]
        return p1, p2

    @staticmethod
    def _last_n_invisible(positions_deque: deque, n: int) -> bool:
        # last n entries must all have vis==0
        return all(int(v[2]) == 0 for v in list(positions_deque)[-n:])

    @staticmethod
    def _is_background(cur_crops: torch.Tensor, past_crops: torch.Tensor) -> bool:
        """Predicts whether the current frame belongs to the background"""
        cur = cur_crops[0]  # [3, 224, 224]
        past = past_crops[0, 2]  # last past frame, [3, 224, 224]
        diff = (cur - past).abs()
        mean_diff = diff.mean()
        max_diff = diff.max()
        if mean_diff < 0.001 or max_diff < 0.2:
            return True
        return False

    # --------------------------------------------------------------------------
    # --------------------------- Modes / Routines ------------------------------
    # --------------------------------------------------------------------------

    @torch.inference_mode()
    def _calibration_step(self, cur_full: torch.Tensor, pos30: torch.Tensor) -> Tuple[float, float, int]:
        # Stage-1 → triangle centers (global-norm)
        s1_xy = self._stage1_infer_xy(cur_full, self._past_full, pos30)
        centers = self._triangle_centers(s1_xy)  # 3 centers

        # Build batch once
        cur_crops, past_crops, pos_crops, corners = self._build_crops_batch(centers, cur_full, self._past_full, pos30)

        # Run both Stage-2 models in one go
        # with torch.autocast(self.device, dtype=torch.float16):
        out = self._run_stage2_batch(cur_crops, past_crops, pos_crops)
        xy_m, agree = out["xy_main"], out["agree_dist"]  # [3,2], [3]
        vis_main = out["vis_main"]  # None or [3]

        # keep if (agreement passes) OR (main visibility passes, when available)
        keep = (agree < self.agree_thresh) | (vis_main >= self.vis_thresh)

        if not torch.any(keep):
            return 0.0, 0.0, 0

        # choose best among kept: minimize (1 - vis) * agree
        cost = (1.0 - vis_main) * agree
        best_idx = int(torch.argmin(cost[keep]).item())
        kept_indices = torch.nonzero(keep, as_tuple=False).view(-1)
        idx = int(kept_indices[best_idx].item())

        x0, y0 = corners[idx]
        xm, ym = float(xy_m[idx, 0].item()), float(xy_m[idx, 1].item())
        gx, gy = self._crop_pred_to_global_norm((x0, y0), (xm, ym))
        return gx, gy, 1

    @torch.inference_mode()
    def _fast_step(self, last_xy: Tuple[float, float], cur_full: torch.Tensor, pos30: torch.Tensor) -> Tuple[
        float, float, int]:
        # Build single-center batch
        # with torch.autocast(self.device, dtype=torch.float16):
        cur_crops, past_crops, pos_crops, corners = self._build_crops_batch([last_xy], cur_full, self._past_full, pos30)
        # background detection
        self.is_background = self._is_background(cur_crops, past_crops)
        if self.is_background:
            return 0.0, 0.0, 0

        out = self._run_stage2_batch(cur_crops, past_crops, pos_crops)

        xy_m = out["xy_main"][0]  # [2]

        # same visibility rule as calibration (but single sample)
        visible = (out["agree_dist"][0] < self.agree_thresh) or (out["vis_main"][0] >= self.vis_thresh)
        if not visible:
            return 0.0, 0.0, 0

        x0, y0 = corners[0]
        xm, ym = float(xy_m[0].item()), float(xy_m[1].item())
        gx, gy = self._crop_pred_to_global_norm((x0, y0), (xm, ym))
        return gx, gy, 1

    @torch.inference_mode()
    def _smart_step(self, cur_full: torch.Tensor, pos30: torch.Tensor, positions_deque: deque) -> Tuple[
        float, float, int, bool]:
        """
        Smart mode:
          1) If last N invis → calibration.
          2) Else do FAST step from last visible (x,y) if available; call that p3.
          3) Triggers (based on last two entries p1,p2 from positions_deque and p3):
             - angle >= angle_thresh
             - jump  >= jump_thresh
             - p2 visible and p3 invisible
          4) If any trigger fires → calibration; else accept p3.
        """

        # (1) N consecutive invisibles?
        if self._last_n_invisible(positions_deque, self.inv_len) \
            and (self._t - self._last_n_invisible_trigger_frame) >= self.inv_len:
            gx, gy, gv = self._calibration_step(cur_full, pos30)
            self._last_n_invisible_trigger_frame = self._t
            return gx, gy, gv, False

        # (2) need last (x,y,vis) from deque
        last = positions_deque[-1]
        last_vis = int(last[2])
        if last_vis == 0:
            # can't fast from an invisible last point
            # return self._calibration_step(cur_full, pos30)
            # The other option is to return (0, 0, 0)
            return 0, 0, 0, False

        last_xy = (float(last[0]), float(last[1]))
        p3x, p3y, p3v = self._fast_step(last_xy, cur_full, pos30)

        # Background detection
        if self.is_background:
            return 0, 0, 0, True

        # Makes sure the shuttle leaving from the top is handled with stability
        if p3y <= 0.03 and all(int(v[2]) == 1 for v in list(pos30)[-3:]):
            return 0, 0, 0, False

        # (3) triggers 2,3,4 require last one or two deque entries
        last_two = self._last_two(positions_deque)
        # if last_two is not None:
        (x1, y1, v1), (x2, y2, v2) = last_two  # p1, p2
        v1 = int(v1); v2 = int(v2)

        # trigger 4: p2 visible but p3 invisible
        if v2 == 1 and p3v == 0:
            self.freshly_invisible += 1
            gx, gy, gv = self._calibration_step(cur_full, pos30)
            return gx, gy, gv, True
        if v2 == 1 and p3v == 1:
            d2 = np.array([p3x - x2, p3y - y2], dtype=np.float32)
            # trigger 3: jump too large
            jump = float(np.linalg.norm(d2))
            if jump >= self.jump_thresh:
                self.big_jump += 1
                gx, gy, gv = self._calibration_step(cur_full, pos30)
                return gx, gy, gv, True
            # triggers 2 when p1 and p2 are visible, and we got a visible p3
            if v1 == 1:
                d1 = np.array([x2 - x1, y2 - y1], dtype=np.float32)
                # trigger 2: angle between d1 and d2 too large
                ang = self._angle_between(d1, d2)
                if ang >= self.angle_thresh:
                    self.big_angle += 1
                    gx, gy, gv = self._calibration_step(cur_full, pos30)
                    return gx, gy, gv, True

        # otherwise accept p3 (visible or invisible)
        return p3x, p3y, p3v, False

    # --------------------------------------------------------------------------
    # ------------------------------- Public API -------------------------------
    # --------------------------------------------------------------------------

    # @torch.inference_mode()
    # def warmup(self, n_iters: int = 8):
    #     """
    #     Lightweight warmup to let cuDNN pick best kernels for our fixed shapes.
    #     Uses all-black frames and zeros positions (doesn't affect any running stats).
    #     """
    #     # minimal black current frame and pos
    #     cur_full = self._zeros_full
    #     pos30 = torch.zeros(30, 3, device=self.device)
    #     # fill past with black
    #     self._past_full = [self._zeros_full, self._zeros_full, self._zeros_full]
    #     if self.use_amp:
    #         amp_ctx = torch.autocast(self.device, dtype=torch.float16)
    #     else:
    #         amp_ctx = torch.nullcontext()
    #     with amp_ctx:
    #         # 1x Stage-1
    #         _ = self._stage1_infer_xy(cur_full, self._past_full, pos30)
    #         # 1x Stage-2 pair at center
    #         c = (0.5, 0.5)
    #         _ = self._run_stage2_at_center(c, cur_full, self._past_full, pos30)
    #         # clear buffer after warmup
    #     self._past_full = [None, None, None]
    #     self._t = 0

    @torch.inference_mode()
    def step(self, frame_rgb_uint8: np.ndarray, positions_deque: deque) -> Tuple[float, float, int, bool]:
        """
        Process ONE video frame. Returns global-normalized (x,y,vis).

        positions_deque: deque of last up-to-30 triplets (x,y,vis) with the LAST being the most recent.
          - On very first frame, caller should pass an empty or zero-filled deque. We always calibrate first frame.

        Note: the engine does NOT mutate the deque; your video->csv loop should append the returned (x,y,vis).
        """
        # 0) Convert current frame
        cur_full = self._to_torch_chw01(frame_rgb_uint8)
        pos30 = self._pos_deque_to_tensor(positions_deque)

        # 1) First frame => always calibration
        if self._t == 0:
            gx, gy, gv = self._calibration_step(cur_full, pos30)
            self._roll_past(cur_full)
            self._t += 1
            return gx, gy, gv, False

        # 2) Mode routing
        if self.mode == "calib":
            gx, gy, gv = self._calibration_step(cur_full, pos30)
            self._roll_past(cur_full)
            self._t += 1
            return gx, gy, gv, False
        elif self.mode == "fast":
            # Calibrate every K frames regardless
            if (self._t % self.fast_calib_interval) == 0:
                gx, gy, gv = self._calibration_step(cur_full, pos30)
                self._roll_past(cur_full)
                self._t += 1
                return gx, gy, gv, False
            else:
                # Use last if visible; else calibration
                if int(positions_deque[-1][2]) == 1:
                    last_xy = (float(positions_deque[-1][0]), float(positions_deque[-1][1]))
                    gx, gy, gv = self._fast_step(last_xy, cur_full, pos30)
                else:
                    gx, gy, gv = (0, 0, 0)
                self._roll_past(cur_full)
                self._t += 1
                return gx, gy, gv, False
        else:  # smart
            gx, gy, gv, calib_triggered = self._smart_step(cur_full, pos30, positions_deque)
            self._roll_past(cur_full)
            self._t += 1
            return gx, gy, gv, calib_triggered

    # --------------------------------------------------------------------------
    # ------------------------------ Internals ---------------------------------
    # --------------------------------------------------------------------------

    @torch.inference_mode()
    def _roll_past(self, cur_full: torch.Tensor):
        """Roll frame buffer: oldest <- mid <- newest <- current."""
        self._past_full = [self._past_full[1], self._past_full[2], cur_full]

    @torch.inference_mode()
    def _pos_deque_to_tensor(self, positions_deque: deque) -> torch.Tensor:
        """
        Convert deque of length 30 to [30,3] on device.
        """
        arr = np.asarray(positions_deque, dtype=np.float32)  # shape [30,3]
        return torch.from_numpy(arr).to(self.device, non_blocking=True)