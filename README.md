# Bachelor Thesis: Automated Defect Detection in Wooden Surfaces

Practical deep learning project focused on real-world quality control in the wood industry.

## At a Glance

- Built and evaluated an end-to-end defect detection benchmark for wooden surfaces.
- Compared three modern object detectors and one multimodal vision-language model.
- Analyzed the trade-off between detection accuracy and inference speed for industrial use.
- Produced a full scientific thesis and reproducible LaTeX documentation.

## Problem

Manual surface inspection is time-consuming, expensive, and inconsistent at scale.  
The goal of this thesis was to evaluate whether modern AI models can detect wood defects accurately enough for production environments.

## What I Did

- Designed and executed a comparative evaluation pipeline for defect detection.
- Worked with a dataset of **4,000 images** across **8 defect classes**.
- Trained and evaluated:
  - **YOLOv11**
  - **Co-DETR**
  - **RT-DETR**
  - **Qwen2.5-VL-7B** (multimodal LLM baseline)
- Measured performance using:
  - **mAP@50** (detection quality)
  - **Inference latency** (real-time suitability)
- Interpreted model behavior and practical deployment trade-offs.

## Key Results

- **Best accuracy:** Co-DETR with **mAP@50 = 62.8**
- **Best real-time profile:** YOLOv11 and RT-DETR (~100 ms inference)
- **Important finding:** Qwen2.5-VL-7B was not reliable for precise spatial defect localization in this task

## Why This Matters

This work demonstrates how model choice depends on business priorities:

- Choose **Co-DETR** when maximizing detection quality is the top priority.
- Choose **YOLOv11/RT-DETR** when low latency and throughput are critical.

The thesis provides decision support for applying AI-based visual quality control in industrial settings.

## Tech Stack

- **Python / Deep Learning frameworks** for model experimentation and evaluation
- **Computer Vision / Object Detection** workflows
- **LaTeX** for scientific writing and final thesis production