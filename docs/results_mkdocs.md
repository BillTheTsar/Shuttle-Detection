# 5. Results

## Quantitative results

- On average, our stage-1 v4 model can pinpoint the shuttle head’s position to within 0.01 normalized units of the 
  ground truth. To put that in perspective, if each frame were divided into a 50×50 grid, the model would correctly 
  identify the square containing the shuttle head, assuming it’s visible.
- The stage-2 v4 model, specifically the main model in our voting ensemble, refines this even further. It achieves an 
  average positional error of just 0.0081 normalized units relative to its 224×224 input crop. When mapped back to 
  the full 1920×1080 frame, this corresponds to a deviation of only 0.00168 units.

While these results reflect training performance, we expect slightly higher deviations for unseen footage. Also, we 
omit discussing visibility losses from stage 2 as the visibility predictor is very unstable and queued for immediate 
update; it would be dishonest to praise its low training losses (suspected overfitting) when this doesn’t transfer to 
inference.

---

## Qualitative results
We show 5 snippets of unseen footage labeled by our system. Click below to be directed to the gallery.

[![Watch the video](labeled_videos/thumbnails/1.jpg)](videos.md)

---

## Limitations

Below we list the main limitations of the inference pipeline with what I believe to be good solutions.

**Limitation 1:** the inference pipeline only performs well when the camera is positioned close to behind the center of the court, from where the net should be almost perfectly rectangular. Performance is particularly bad if the camera is placed near corners of the court.

**Solution to limitation 1:** all game footage the model trained on was taken from a point close to behind the center of the court. Thus, we need to enlarge our dataset by including footage taken from a variety of angles. This should not only reduce the impact of limitation 1, but also make our models more robustly sensitive to “shuttle shapes” in general.

---

**Limitation 2:** as mentioned multiple times, the visibility predictor in the stage-2 v4 model is not effective. So far, it is a affine function involving the mean and max values of the heatmap. However, the logits of the heatmap are not normalized, leading to drastically different mean and max values from similar crops.

**Solution to limitation 2:** firstly, normalize the heatmap logits to lie within [0, 1]. Apply a sequence of 1x1 and 3x3 convolutions on the heatmaps produced by both the main and support stage-2 models to produce a visibility score. A convolution-based approach was chosen because the spatial distribution of the heatmap often reveals whether the shuttle head is visible; something that is difficult to capture with affine methods alone.

---

**Limitation 3:** the inference pipeline is highly sensitive to video scale. One can infer from the snippets of unseen footage what scale range the models perform reliably in.

**Solution to limitation 3:** as with limitation 1, the models were trained on footage at more or less the same video scale. While it might seem reasonable to include samples at multiple scales to improve generalization, I would not recommend incorporating smaller scales; at such resolutions, shuttle heads become featureless blobs, making learning and detection meaningless. Instead, it is recommended to record at a similar scale (and thus distance from the court) to that shown in the snippets.

---

**Limitation 4:** the inference pipeline fails entirely on low-quality footage. Common symptoms include frequent dropped frames (surprisingly common from equipment that claims to be 60 FPS but can’t actually achieve it), poor interpolation methods that cause the shuttle head to vanish intermittently, and constant refocusing that disrupts visual consistency. 

**Solution 4:** none. It is unrealistic to expect an inference pipeline with CNN backbones to adapt to impossible image data. Instead, use recording equipment that reliably produces high-quality footage.

---

## Reflections

Below, we list the major lessons learnt.

1. Before designing a model for a task, I should conduct more research into existing methods. It would have saved weeks of model development had I learnt about the BiFPN → heatmap method.
2. Before building the dataset, it's crucial to define the target scenarios for model deployment and ensure the data adequately represents each of them.

---

## Work under progress

1. Develop a separate visibility head as described in [Limitations](#limitations).
2. Enlarge the dataset to improve performance on wider camera angles and on frames during shuttle impact.
3. Deploy the models on Hugging Face Space (or similar platform) to enable drag-and-drop video inference for users.

<br><br>