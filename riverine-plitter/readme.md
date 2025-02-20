# Riverine Plastic Detection Project

## Overview
This project adapts the pLitter detection pipeline (based on YOLOv5) to enhance the detection of plastic litter in riverine environments. Our goal is to develop a self-contained system that uses water-specific preprocessing to improve detection in challenging water scenes, along with an interactive video detection pipeline that outputs both qualitative (visual overlays and screenshots) and quantitative (detection counts) results.

---
## Project Objectives

### 1. Enhance Detection in Riverine Settings
- Adapt the existing YOLOv5-based pLitter model to work better in water environments by addressing issues like glare, low contrast, and reflections.

### 2. Domain-Specific Preprocessing
- Implement water-specific preprocessing (using techniques such as CLAHE, saturation boosting, and fast denoising) to improve the input quality for detection.

### 3. Self-Contained Pipeline
- Integrate the YOLOv5 model (including custom weights like `best_model.pt` for riverine settings) with a video processing pipeline that supports interactive parameter selection, live display, and saving outputs.

### 4. Results Generation
- Provide both quantitative (detection counts, processing times) and qualitative (visual overlays, screenshots) results for analysis.

---
## Preprocessing
### Water-Specific Preprocessing
The `preprocess_water.py` module implements preprocessing steps tailored for water conditions:

#### 1. HSV Conversion & CLAHE
- The image is converted to HSV.
- CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied to the V (value) channel to improve contrast.

#### 2. Saturation Boost
- The saturation channel is slightly boosted to enhance the vividness of colors, helping the model better distinguish plastic objects from the water.

#### 3. Denoising
- A fast denoising filter (such as median blur or fast NL-means) is applied to reduce noise while preserving edges.

#### Rationale
Water scenes often have issues like glare, low contrast, and reflections. The preprocessing improves image quality for the detection model, potentially leading to better detection performance.

---
## Video Detection Pipeline

### Key Features
- **Frame Skipping**: Users can specify a frame skip value to increase processing speed.
- **Live Display and Screenshot Options**: The script prompts whether to show live detection and save screenshots.
- **Preprocessing Option**: If enabled, frames are preprocessed before detection. Preprocessed outputs are saved in a dedicated `preprocessed` folder.
- **Dynamic Overlay**: Detection counts (unique detections and frame detections) are overlaid on the output video.
- **Speed Optimization**: Frames are resized for inference and restored to original resolution for output.

#### Rationale
Resizing frames for inference greatly speeds up processing, which is essential for real-time video detection, while resizing back preserves the output quality.

---
## Evaluation & Results

### 1. Quantitative Results
While a full quantitative evaluation (like mAP calculation) would require a well-annotated dataset and additional scripts, our pipeline outputs detection counts and processing times as basic quantitative measures.

### 2. Qualitative Results
- The final output video shows bounding boxes over detected objects.
- Detection counts are overlaid to provide clear visual evidence of the system’s performance.

#### Rationale
In the absence of a full training dataset for fine-tuning, these qualitative and basic quantitative results help demonstrate the effectiveness of the adapted pipeline.

---
## Running the Model
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/riverine-plastic-detection.git
cd riverine-plastic-detection
```

### 2. Clone YOLOv5
```bash
git clone https://github.com/ultralytics/yolov5.git
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Video Detection
```bash
python video_detector.py --input path/to/video.mp4 --output path/to/output.mp4 --type river --frame_skip 2 --live --preprocess
```

### 5. Optional: Save Preprocessed Output
```bash
python video_detector.py --input path/to/video.mp4 --output path/to/output.mp4 --type river --frame_skip 2 --preprocess --save_preprocessed
```

---
## Summary & Conclusion

### What We Did
- Adapted the pLitter/YOLOv5 pipeline for riverine plastic litter detection.
- Implemented water-specific preprocessing, custom model weights, and a flexible video processing script.

### Why We Did It
- Riverine environments pose unique challenges (e.g., glare, reflections, low contrast) that standard plastic litter detectors might not handle well.

### Outcome
- The system processes videos in near real-time, overlays detection information dynamically, and saves both the final output and preprocessed versions.
- Although we have not re-trained the model due to dataset constraints, the adapted pipeline demonstrates qualitative improvements and basic quantitative performance metrics.

---
## Future Work
- **Dataset Expansion & Fine-Tuning**: Once a sufficient riverine dataset is available, the model can be fine-tuned further for improved accuracy.
- **Advanced Evaluation**: Implement a full quantitative evaluation using the COCO API.
- **Real-Time Deployment**: Explore asynchronous processing or batch inference to boost real-time performance.

---
## Before and After Preprocessing
Below are sample images showcasing the impact of preprocessing on detection quality:

| Before Preprocessing | After Preprocessing |
|----------------------|--------------------|
| ![Before](original\plitter\data\river_sample_data_output\screenshots\frame_000001_new_1_preprocessed.jpg) | ![After](original\plitter\data\river_sample_data_output\preprocessed\screenshots\frame_000001_new_1_preprocessed.jpg) |


