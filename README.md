# Badminton Shuttle Head Tracking from Video

---

An independent computer vision project culminating in a system capable of tracking a badminton shuttle from match footage. 
With TensorRT, the system achieves real-time speeds of 86 FPS.

Once improved with a larger dataset, the project will become the foundation for badminton analytics and shot prediction.

# Project Overview
>[Click here](https://billthetsar.github.io/Shuttle-Detection/videos/) to look at 5 unseen video
snippets labeled by the system.
>
>In the [Project Report](https://billthetsar.github.io/Shuttle-Detection/), we explain the objectives, challenges 
overcome and insights gained from multiple model redesigns. It covers everything below and much more!

---

## Key Achievements

- **Independently executed the full ML development cycle**, from data creation, preprocessing 
  and model architecture design to training, iterative redesigns, and deployment using TensorRT.
- **Built a computer vision system that competently localizes shuttle heads** from video footage and 
  hits inference speeds of 86 FPS, comfortably surpassing the 60 FPS requirement for live video processing.

## Tools Used and Learned

- **PyTorch** (learned)
- **ONNX and TensorRT** (learned)
- **EfficientNet B0, EfficientNetv2 B0 and B3** (learned)
- **Paperspace Gradient** (learned)
- **Label Studio** (learned)
- **Hugging Face** (learned)
- **TensorBoard** (learned)
- **Pandas, numpy and matplotlib** (used)

## Skills Picked Up

- **End-to-end pipeline design of a two-stage system**
- **Iterative model redesigns**
- **Tensor manipulation up to rank 5**
- **Using Label Studio to manually annotate 5800 video frames**
- **Learned to export PyTorch state dicts to ONNX, and then to TensorRT .engine files**
- **Learned to adapt training for Paperspace Gradient**

---

# Installation and Usage
Under [requirements.txt](https://github.com/BillTheTsar/Shuttle-Detection/blob/main/requirements.txt), it is recommended 
to install only the necessary packages; though we show the optional packages as well.

Now we've installed the necessary packages, we can label our videos with shuttle head positions frame by frame.
**Note:** input videos must currently be 1920x1080 resolution; resize if necessary.

1. Using your favorite CLI, navigate to the project root directory.

2. To get the csv, run 
`python scripts/video_inferencer.py --video_path path/to/input.mp4 --csv_path path/to/output.csv`.

3. To annotate the video with the csv labels, run `python scripts/video_labeler.py 
--video path/to/input.mp4 --csv path/to/output.csv --out path/to/annotated.mp4`.

<details>
<summary>video_inferencer.py --help</summary>

```bash
PS D:\Shuttle Detection Model> python scripts/video_inferencer.py --help
usage: video_inferencer.py [-h] [--video VIDEO] [--csv CSV] [--device DEVICE] [--mode {calib,fast,smart}]

Label every frame of a video with shuttle head predictions.

options:
  -h, --help            show this help message and exit
  --video VIDEO         Path to the input video file to be processed.
  --csv CSV             Path to the output CSV file where predictions will be saved.
  --device DEVICE       Device to run inference on (e.g. 'cuda' for GPU or 'cpu' for CPU).
  --mode {calib,fast,smart}
                        Inference mode: 'calib' - calibration mode, usually slower but useful for debugging; 'fast' -
                        prioritize speed over accuracy; 'smart' - balanced mode (recommended).
```
By default, device="cuda" and mode="smart".

</details>

<details>
<summary>video_labeler.py --help</summary>

```bash
PS D:\Shuttle Detection Model> python scripts/video_labeler.py --help
usage: video_labeler.py [-h] [--video VIDEO] [--csv CSV] [--out OUT]

Overlay predicted shuttle positions onto a video.

options:
  -h, --help     show this help message and exit
  --video VIDEO  Path to input video file.
  --csv CSV      Path to CSV file with predictions (columns: frame, x, y, vis).
  --out OUT      Path to save annotated video.
```

</details>

<br><br>