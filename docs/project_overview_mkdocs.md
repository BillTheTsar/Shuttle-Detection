# 2. Project Overview

## Key information

- **Duration:** 2 months, 13/07/2025 - 12/09/2025
- **Project type:** independent research and development
- **Domain:** supervised learning in computer vision
- **Primary language:** Python

## Achievements

- **Independently executed the full ML development cycle**, from data creation (5800 labeled samples), preprocessing 
  and model architecture design to training, iterative redesigns (11 versioned models), and deployment using TensorRT.
- **Compiled a custom-labeled dataset of 5800 frames**, where final-stage predictions deviated by only 3-4 pixels on 
  average from the manually labeled shuttle head positions; an impressive level of precision given the object’s small 
  size and extreme speed.
- **Built a two-stage computer vision system that competently solves the shuttle localization task** proposed in the 
  problem statement. The system achieves real-time inference at 86 FPS, comfortably surpassing the 60 FPS requirement 
  for live video processing.
- **Delivered an inference engine running on consumer-grade GPUs** using `.engine` model exports. The optimized 
  deployment pipeline reduces startup latency from 9 to <1 seconds and doubles throughput with layer fusion and FP16 
  precision.
---

## Tools used and learned during the project

### Deep learning and model deployment
- **PyTorch** (learned): for implementing multiple model architecture designs, model training scripts and deployment 
  (though inference using TensorRT engine exports is 2.5 times faster).
- **ONNX and TensorRT** (learned): for accelerated deployment by exporting .pth model weight files to .onnx and .engine 
  files respectively. With layer fusion and mixed precision (almost all in fp16), inference jumped from 42 FPS to 86 
  FPS on average.
- **EfficientNet B0, EfficientNetv2 B0 and B3** (learned): these formed the CNN backbones of both stages, from which 
  FPN and BiFPN techniques were applied on layers of varying strides to produce heat maps. These were chosen for their 
  light parameter counts and low FLOPs compared to ResNet and YOLO-based methods.

### Data infrastructure and training
- **Paperspace Gradient** (learned): a cloud-based workflow for executing model training on rented A4000 GPU in the 
  Notebooks environment with persistent storage before acquiring a laptop with RTX 5080 GPU.
- **Label Studio** (learned): for importing raw data hosted on GitHub, facilitating hand-labeling and exporting the 
  labeled samples into a csv file for downstream cleaning.
- **Hugging Face** (learned): for hosting the entire processed dataset to be imported into the Paperspace Gradient 
  notebook (direct uploading, even zipped, was impossible).

### Data processing and visualization
- **TensorBoard** (learned): for visualizing training and validation losses during training, identifying model 
  overfitting and monitoring parameters of interest.
- **Pandas, numpy and matplotlib** (used): pandas was used for data cleaning, converting the csv files exported from 
  Label Studio to the processed dataset directly used for model training. numpy and matplotlib were used for verifying 
  the correctness of data augmentation techniques, visualizing model heatmap outputs and prototype adjustments.

---

## Skills picked up during the project

- **End-to-end pipeline design of a two-stage system**, incorporating modular training, hard example mining for stage 2 
  and a tailored method of integrating the two stages for stable inference.
- **Iterative model redesigns** based on TensorBoard log patterns and inference stability. From developing 11 versioned 
  models, I have seen improvements from complete paradigm shifts e.g. switching from MLP on global average pooling to 
  heatmap-based inference to subtle yet impactful changes e.g. adjusting Gaussian noise magnitudes.
- **Tensor manipulation up to rank 5**, including reshaping, broadcasting, bit-masking, and squeezing.
- **Using Label Studio to manually annotate 5800 video frames** with shuttle head positions, then using pandas with 
  anomaly handling to produce training-ready datasets.
- **Learned to export PyTorch state dict models to ONNX, and then to TensorRT .engine files** with optimized 
  performance settings and batch-size flexibility during inference.
- **Learned to adapt training for Paperspace Gradient** with its CLI-based model versioning and environment setup under 
  compute and memory restraints.

<br><br>