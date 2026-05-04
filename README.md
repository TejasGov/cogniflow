# CogniFlow
### A Multimodal Pipeline for Real-Time Cognitive Load and Task Abandonment Prediction in ADHD Patients

**CSE 474 — Introduction to Machine Learning | Spring 2026 | University at Buffalo**

**Team:** Ishita Ramrayka · Tejas Govind · Ayudh Pratap Singh

---

## What is CogniFlow?

People with ADHD don't struggle with attention in a fixed, predictable way — their cognitive capacity shifts throughout the day depending on the task, how long they've been working, and how much context switching they've done. Existing tools like to-do lists, time blocks, and screen time managers treat every user the same at every moment. CogniFlow is built to do the opposite.

CogniFlow is a three-stage multimodal pipeline that combines:
- **LLM-based task analysis** (how hard is this task?)
- **Real-time webcam behavioral monitoring** (how is the user doing right now?)
- **XGBoost fusion classifier** (are they at risk of abandoning this task?)

The output is a continuous **Abandonment Risk Score** between 0 and 1, updated in real time, with a live SHAP-based explanation showing exactly which signals are driving the prediction.

---

## Results at a Glance

| Metric | Value |
|---|---|
| AUC-ROC | **0.958** |
| 5-Fold CV AUC | **0.956 ± 0.008** |
| YOLOv8 throughput | **40.3 FPS** |
| DAiSEE gaze variance p-value | **0.0044** |
| Monotonic profile ordering | Confirmed |
| Parse success rate (Stage 1) | **400/400 (100%)** |

---

## Pipeline Overview

```
Task Description (text)
        │
        ▼
┌─────────────────────┐
│  Stage 1: LLaMA-3   │  → complexity, steps, priority
│  Task Analysis      │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Stage 2: YOLOv8    │  → avg_gaze, avg_head_pose,
│  Behavioral Signals │     avg_eye_openness, gaze_variance
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Stage 3: XGBoost   │  → Abandonment Risk Score [0, 1]
│  Fusion Classifier  │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Live SHAP Overlay  │  → Real-time per-feature explanation
│  on Webcam Feed     │     on your face
└─────────────────────┘
```

---

## Repository Structure

```
cogniflow/
│
├── README.md
├── requirements.txt
│
├── stage1_task_analysis/
│   └── cogniflow_stage1.ipynb      # LLaMA-3 O*NET annotation pipeline
│
├── stage2_behavioral/
│   ├── cogniflow_stage2.ipynb      # YOLOv8 behavioral signal extraction
│   └── cogniflow_daisee_v3.ipynb   # DAiSEE validation (234 clips)
│
├── stage3_fusion/
│   └── cogniflow_stage3.ipynb      # XGBoost fusion + SHAP analysis
│
├── demo/
│   └── cogniflow_webcam.py         # Live webcam SHAP demo
│
├── models/
│   └── cogniflow_xgboost.pkl       # Trained XGBoost classifier
│
└── figures/
    ├── cogniflow_gantt.png
    ├── stage2_boxplots.png
    ├── stage3_roc_curve.png
    ├── stage3_feature_importance.png
    ├── stage3_profile_risk_scores.png
    ├── stage3_confusion_matrix.png
    ├── cogniflow_screenshot_1.png   # Distracted state (Risk: 0.994)
    ├── cogniflow_screenshot_2.png   # Focused state (Risk: 0.241)
    └── cogniflow_screenshot_3.png   # Transitional state (Risk: 0.503)
```

---

## Setup & Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Running the Live Demo
1. Download `cogniflow_xgboost.pkl` from the `models/` folder
2. Put it in the same directory as `cogniflow_webcam.py`
3. Run:
```bash
python demo/cogniflow_webcam.py
```
- Press **S** to save a screenshot
- Press **Q** to quit

The demo opens your webcam, extracts facial keypoints using YOLOv8-Pose, computes the 7 features every 5 seconds, queries the XGBoost model, and overlays the SHAP values on your face in real time. The skeleton mesh color changes from green (safe) → orange (moderate) → red (at risk) based on the current risk score.

---

## Running the Notebooks

All notebooks are designed to run on **Kaggle** with a T4 GPU. Each stage is independent.

### Stage 1 — Task Analysis
- Upload `stage1_task_analysis/cogniflow_stage1.ipynb` to Kaggle
- Add your HuggingFace token via Kaggle Secrets (`HF_TOKEN`)
- Upload the O*NET `Task Statements.xlsx` as a Kaggle dataset
- Run All → produces `task_dataset.csv`

### Stage 2 — Behavioral Signal Extraction
- Upload `stage2_behavioral/cogniflow_stage2.ipynb` to Kaggle
- Add `yolov8n-pose.pt` as a Kaggle dataset
- Run All → produces `visual_features.csv`

### Stage 2b — DAiSEE Validation
- Upload `stage2_behavioral/cogniflow_daisee_v3.ipynb` to Kaggle
- Add your DAiSEE video subset and `yolov8-pose` as datasets
- Run All → produces `daisee_features.csv`, `daisee_stats.csv`, validation plots

### Stage 3 — Fusion & Classification
- Upload `stage3_fusion/cogniflow_stage3.ipynb` to Kaggle
- Add `task_dataset.csv` and `visual_features.csv` as a dataset
- No GPU needed — runs on CPU in under 2 minutes
- Run All → produces `cogniflow_xgboost.pkl`, ROC curve, feature importance, SHAP plots

---

## Datasets Used

| Dataset | Source | Used For |
|---|---|---|
| O*NET Task Statements | [onetcenter.org](https://www.onetcenter.org/database.html) | Stage 1 task annotation |
| DAiSEE | [iitm.ac.in](https://people.iith.ac.in/vineethnb/resources/daisee/index.html) | Stage 2 behavioral validation |
| Synthetic sessions | Generated via calibrated Gaussian profiles | Stage 3 training |

> **Note:** DAiSEE requires access approval from the authors. The O*NET dataset is publicly available. Neither dataset is included in this repository.

---

## Models Used

| Model | Purpose | Where |
|---|---|---|
| LLaMA-3.1-8B-Instruct | Task complexity scoring | Kaggle T4 GPU (4-bit quantized) |
| YOLOv8n-Pose | Facial keypoint extraction | Local + Kaggle |
| XGBoost | Multimodal fusion classifier | Included in `models/` |
| SHAP TreeExplainer | Real-time explainability | Live demo |

---

## Key Findings

- **Task complexity** is the strongest single predictor (XGBoost gain: 0.29)
- **Gaze variance** is the most important behavioral feature (gain: 0.25) and the only DAiSEE-validated signal (p = 0.0044)
- Static gaze direction and head pose alone are not sufficient — what matters is the *variability* of gaze over time
- The fusion model correctly orders risk across all four behavioral profiles: highly focused (0.370) < moderate (0.922) < fatigued (0.996) < easily distracted (0.997)

---

## Acknowledgements

This project was completed as part of CSE 474 Introduction to Machine Learning at the University at Buffalo under Dr. Das Bhattacharjee. We used Kaggle Notebooks for GPU compute and HuggingFace for model access.
