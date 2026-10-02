import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def update_notebook():
    nb_path = os.path.join(PROJECT_ROOT, 'notebooks', '2. sentiment_analysis.ipynb')
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Update Cell 1 code
    c1_code = """# =========================================================================
# STEP 0: GOOGLE COLAB & ENVIRONMENT SETUP (VS Code + Colab Compatible)
# =========================================================================
import sys
import os

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("⚡ Google Colab environment detected!")
    print("Verifying / installing required dependencies...")
    !pip install -q transformers datasets accelerate scikit-learn seaborn matplotlib scipy
else:
    print("💻 Local environment detected.")

import gc
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats

# -------------------------------------------------------------------------
# HARDWARE ACCELERATION SETUP (CUDA / GPU vs. CPU)
# -------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\\nHardware Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU Name:      {torch.cuda.get_device_name(0)}")
    print(f"  VRAM Memory:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"  CUDA Version:  {torch.version.cuda}")
else:
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"  CPU Threads:   {torch.get_num_threads()}")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# -------------------------------------------------------------------------
# ROBUST DATA PATH RESOLUTION (Local Windows, VS Code, Google Drive, Colab)
# -------------------------------------------------------------------------
PROJECT_ROOT = os.getcwd()
if 'notebooks' in PROJECT_ROOT:
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

CANDIDATE_DATA_PATHS = [
    # 1. Local Windows & Workspace paths (if running locally in VS Code)
    r"D:\\Aspect-Based-Sentimental-Analysis-on-Food-Reviews\\modified_annotations_2000_corrected.csv",
    r"D:\\Aspect-Based-Sentimental-Analysis-on-Food-Reviews\\final_aspect_evaluation\\modified_annotations_2000_corrected.csv",
    os.path.join(PROJECT_ROOT, 'final_aspect_evaluation', 'modified_annotations_2000_corrected.csv'),
    os.path.join(PROJECT_ROOT, 'modified_annotations_2000_corrected.csv'),
    'modified_annotations_2000_corrected.csv',
    '../modified_annotations_2000_corrected.csv',
    
    # 2. Colab Cloud Root paths (if uploaded to Colab /content/)
    '/content/modified_annotations_2000_corrected.csv',
    '/content/final_aspect_evaluation/modified_annotations_2000_corrected.csv',
    '/content/Aspect-Based-Sentimental-Analysis-on-Food-Reviews/final_aspect_evaluation/modified_annotations_2000_corrected.csv',
    '/content/Aspect-Based-Sentimental-Analysis-on-Food-Reviews/modified_annotations_2000_corrected.csv',
    
    # 3. Google Drive paths (if Drive is mounted in Colab)
    '/content/drive/MyDrive/modified_annotations_2000_corrected.csv',
    '/content/drive/MyDrive/Aspect-Based-Sentimental-Analysis-on-Food-Reviews/modified_annotations_2000_corrected.csv',
    '/content/drive/MyDrive/Aspect-Based-Sentimental-Analysis-on-Food-Reviews/final_aspect_evaluation/modified_annotations_2000_corrected.csv'
]

DATA_PATH = next((p for p in CANDIDATE_DATA_PATHS if os.path.exists(p)), None)

# If in Colab and not found yet, try mounting Google Drive automatically
if DATA_PATH is None and IN_COLAB:
    print("\\nℹ️ File not found in /content/. Checking Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        DATA_PATH = next((p for p in CANDIDATE_DATA_PATHS if os.path.exists(p)), None)
    except Exception as e:
        print(f"Drive mount skipped: {e}")

if DATA_PATH is None:
    raise FileNotFoundError(
        "\\n❌ modified_annotations_2000_corrected.csv not found!\\n"
        "How to provide the file:\\n"
        "1. If running VS Code connected to Colab: Place the CSV in your Google Drive (MyDrive/modified_annotations_2000_corrected.csv) "
        "OR open Google Colab in your browser once and drag the CSV into the left /content/ files panel.\\n"
        "2. If running VS Code locally: Ensure you select a local Python kernel (not remote Colab) so it reads from your D: drive directly."
    )

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'sentiment_training_2000')
DISTILBERT_DIR = os.path.join(OUTPUT_DIR, 'distilbert')
os.makedirs(DISTILBERT_DIR, exist_ok=True)

print(f"\\n✅ Project Root:   {PROJECT_ROOT}")
print(f"✅ Data File Found: {DATA_PATH}")
print(f"✅ DistilBERT Dir:  {DISTILBERT_DIR}")
"""
    nb.cells[1].source = c1_code

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Notebook Cell 1 updated successfully.")

if __name__ == '__main__':
    update_notebook()
