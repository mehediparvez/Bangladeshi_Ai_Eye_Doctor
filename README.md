# Bangladeshi AI Eye Doctor

AI-powered retinal disease detection system designed for Bangladeshi patients. Combines deep learning-based fundus image classification with a bilingual (English/Bengali) conversational diagnostic agent.

## Overview

This project uses an **EfficientNet-B0** model trained on retinal fundus photographs to classify 10 eye conditions. It also includes a conversational AI agent that collects patient symptoms in English or Bengali, analyzes uploaded fundus images, and provides preliminary diagnostic guidance.

### Disease Classes

| # | Class | Test Support |
|---|-------|-------------|
| 1 | Central Serous Chorioretinopathy (CSC) | 101 |
| 2 | Diabetic Retinopathy | 1,509 |
| 3 | Disc Edema | 127 |
| 4 | Glaucoma | 1,349 |
| 5 | Healthy | 1,024 |
| 6 | Macular Scar | 444 |
| 7 | Myopia | 500 |
| 8 | Pterygium | 17 |
| 9 | Retinal Detachment | 125 |
| 10 | Retinitis Pigmentosa | 139 |

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 75.37% |
| Weighted F1 | 0.757 |
| Macro F1 | 0.775 |
| CV F1 Mean (3-fold) | 0.705 ± 0.020 |

### Per-Class F1 Scores

| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| Pterygium | 1.000 | 1.000 | **1.000** |
| Diabetic Retinopathy | 0.979 | 0.791 | **0.875** |
| Retinal Detachment | 0.728 | 0.984 | **0.837** |
| Disc Edema | 0.693 | 0.961 | **0.805** |
| Retinitis Pigmentosa | 0.624 | 0.993 | **0.767** |
| Healthy | 0.753 | 0.746 | **0.749** |
| Macular Scar | 0.676 | 0.741 | **0.707** |
| Myopia | 0.571 | 0.854 | **0.684** |
| Glaucoma | 0.745 | 0.600 | **0.665** |
| CSC | 0.497 | 0.970 | **0.658** |

## Architecture

```
Input Image (224x224 RGB)
        │
        ▼
  EfficientNet-B0 (timm)
   - Pretrained backbone
   - Dropout: 0.3
        │
        ▼
  Classification Head
   - 10-class softmax
        │
        ▼
  Disease Prediction + Confidence
```

Training uses 3-fold stratified cross-validation with weighted random sampling to handle class imbalance. The best fold checkpoint is exported as the deployment model.

## Project Structure

```
Bangladeshi_Ai_Eye_Doctor/
├── app.py                          # Gradio web interface
├── run_demo.py                     # CLI demo + dashboard generator
├── visualization.html              # Interactive results dashboard
├── requirements.txt                # Python dependencies
├── final_model_for_deployment.pth  # Trained model weights (best fold)
├── final_model_metrics.json        # Evaluation metrics + predictions
├── model_dashboard.png             # Generated results dashboard
│
├── Eye Model Training code.ipynb   # Model training notebook
├── eye2025 (4).ipynb               # Training notebook (alternate)
├── 3_conversational_ai_ENHANCED.ipynb  # Conversational AI notebook
├── synthetic_data_generation.ipynb # Patient query dataset generator
│
├── Trained model/                  # Cross-validation fold checkpoints
│   ├── efficientnet_b0_fold1_best.pth
│   ├── efficientnet_b0_fold2_best.pth
│   └── efficientnet_b0_fold3_best.pth
│
├── eye_images/                     # Sample fundus images for testing
│   ├── Healthy[1-3].jpg
│   ├── Macular Scar[1-3].jpg
│   └── Retinitis Pigmentosa[1-3].jpg
│
├── text_data/
│   └── patient_queries.csv         # 10K bilingual patient queries
│
├── Diagram/                        # Architecture & results diagrams
│   ├── system_architecture_diagram.png
│   ├── training_workflow_diagram.png
│   ├── conversational_ai_flowchart.png
│   ├── final_confusion_matrix.png
│   ├── final_roc_curves.png
│   └── ...
│
└── EyeDiseaseDetectionSubmissionDoc_FINAL_WITH_DIAGRAMS.docx
```

## Installation

```bash
# Clone the repository
git clone https://github.com/mehediparvez/Bangladeshi_Ai_Eye_Doctor.git
cd Bangladeshi_Ai_Eye_Doctor

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, ~1 GB disk for PyTorch (CPU). GPU is optional and auto-detected via CUDA.

## Usage

### CLI Demo

Loads the model, runs inference on sample images, and generates `model_dashboard.png`:

```bash
python run_demo.py

# Text-only (skip plot generation)
python run_demo.py --no-plot
```

### Web Interface

Launches a Gradio app for uploading fundus images and getting predictions:

```bash
python app.py
```

Open the URL printed in the terminal (default: `http://localhost:7860`).

### Interactive Dashboard

Open `visualization.html` in any browser for an interactive results dashboard with confusion matrix, per-class metrics, error analysis, and patient query statistics.

## Datasets

**Fundus Images:** Retinal fundus photographs sourced from Kaggle, covering 10 disease categories with significant class imbalance (17 to 1,509 samples per class in the test set).

**Patient Queries:** 10,000 synthetic bilingual queries (`text_data/patient_queries.csv`) spanning 10 disease categories, 5 urgency levels, with an equal English/Bengali split. Generated for training the conversational diagnostic agent.

## Tech Stack

- **Deep Learning:** PyTorch, torchvision, timm (EfficientNet-B0)
- **Web Interface:** Gradio
- **Evaluation:** scikit-learn, matplotlib, seaborn
- **Data Processing:** pandas, NumPy, Pillow

## License

TBD
