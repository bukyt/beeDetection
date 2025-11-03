# Project members:  
Mihkel Kulu, Norman Tolmats, Markus Kivimäe, Joonas Tiitson  

# Bee Type Detection for Smart Beehive Monitoring

## Problem Statement
Beekeeping plays a vital role in agriculture, and monitoring bee activity helps assess hive health and productivity. The **Gratheon project** aims to build a smart beehive equipped with an entrance observer that can **detect and classify bees entering or leaving the hive**.

The goal is to automatically identify:
- **Drones (male bees)**
- **Worker bees (female)**
- **Worker bees with pollen**
- **Bees infected with varroa mites**

Current detection methods lack accuracy and generalization.

## Objectives
- Detect and classify bee types from video data.
- Annotate or enhance existing datasets as needed.
- Train a model capable of **real-time or near-real-time inference** on **edge devices**.
- Evaluate performance using **precision**, **accuracy**, and **F1-score**.

## Data
The dataset is sourced from the **Gratheon project** and other public repositories:
- [Raw video data](https://drive.google.com/drive/folders/105PmxDKFUR6NCPLHBkXGdkfcZwWf9ABI)
- [Project dataset description](https://gratheon.com/research/Datasets/)
- [Public annotated bee/varroa datasets](https://universe.roboflow.com/search?q=varroa)

We will prioritize high-quality, well-labeled footage. If needed, additional data will be annotated using **CVAT** and expanded via **data augmentation** (brightness, rotation, noise, flips, etc.) to improve robustness.

## Methodology
Given the hardware constraints of edge devices, model selection must balance **accuracy** and **speed**.

1. **Primary approach:** Object detection with **YOLOv8** or **YOLO-NAS**, known for high efficiency and real-time performance.
2. **Alternative experiments:** Models like **Faster R-CNN** for comparison.
3. **Optimization:** Explore deployment techniques using **TensorRT** or **TensorFlow Lite** for reduced latency and size.

## Evaluation
Model performance will be assessed using:
- **Accuracy**
- **Precision**
- **F1-score** (primary metric)

## Challenges
- Multiple bees in a single frame
- Variable lighting and weather conditions
- Non-bee insects causing false detections

## Tools and Resources
- **Frameworks:** PyTorch, TensorFlow  
- **Annotation:** CVAT  
- **Development:** Local machine & Google Colab (GPU)  
- **Model families:** YOLO (preferred), Faster R-CNN

## Further Guidance
Suggestions are welcome for:
- **Model optimization** for edge deployment  
- **Lightweight architectures** or **quantization techniques** to reduce inference time and memory footprint
