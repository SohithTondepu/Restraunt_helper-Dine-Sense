import os
import nbformat

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb_path = os.path.join(PROJECT_ROOT, 'notebooks', '2. sentiment_analysis.ipynb')

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

c1_code = """# =========================================================================
# STEP 0: GOOGLE COLAB & ENVIRONMENT SETUP (VS Code + Colab Compatible)
# =========================================================================
import sys
import os
import glob

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
# AUTO-DETECT CSV FILE (Searches /content/, Drive, and Local Windows)
# -------------------------------------------------------------------------
PROJECT_ROOT = os.getcwd()
if 'notebooks' in PROJECT_ROOT:
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

DATA_PATH = None

# Priority 1: Check standard explicit file locations
candidates = [
    # Colab uploaded files
    '/content/modified_annotations_2000_corrected.csv',
    '/content/modified_annotations_2000.csv',
    # Drive locations
    '/content/drive/MyDrive/modified_annotations_2000_corrected.csv',
    '/content/drive/MyDrive/modified_annotations_2000.csv',
    # Local Windows paths
    r"D:\\Aspect-Based-Sentimental-Analysis-on-Food-Reviews\\modified_annotations_2000_corrected.csv",
    r"D:\\Aspect-Based-Sentimental-Analysis-on-Food-Reviews\\final_aspect_evaluation\\modified_annotations_2000_corrected.csv",
    r"D:\\Aspect-Based-Sentimental-Analysis-on-Food-Reviews\\final_aspect_evaluation\\modified_annotations_2000.csv",
    os.path.join(PROJECT_ROOT, 'final_aspect_evaluation', 'modified_annotations_2000_corrected.csv'),
    os.path.join(PROJECT_ROOT, 'modified_annotations_2000_corrected.csv')
]

for p in candidates:
    if os.path.exists(p):
        DATA_PATH = p
        break

# Priority 2: Wildcard search in /content/ if in Colab (handles renamed uploads)
if DATA_PATH is None and os.path.exists('/content'):
    print("\\nListing all files in /content/:", [f for f in os.listdir('/content') if not f.startswith('.')])
    content_matches = glob.glob('/content/*annotations*.csv') + glob.glob('/content/*2000*.csv')
    if content_matches:
        DATA_PATH = content_matches[0]
        print(f"Found matched file via search: {DATA_PATH}")

# Priority 3: Mount Google Drive if not found yet
if DATA_PATH is None and IN_COLAB:
    print("\\nChecking Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        drive_matches = glob.glob('/content/drive/MyDrive/*annotations*.csv') + glob.glob('/content/drive/MyDrive/*2000*.csv')
        if drive_matches:
            DATA_PATH = drive_matches[0]
    except Exception as e:
        print(f"Drive mount error: {e}")

if DATA_PATH is None:
    available_files = os.listdir('/content') if os.path.exists('/content') else os.listdir('.')
    raise FileNotFoundError(
        f"\\n❌ Could not find an annotations CSV file!\\n"
        f"Files currently in directory: {available_files}\\n"
        f"Please check the filename of the uploaded file."
    )

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'sentiment_training_2000')
DISTILBERT_DIR = os.path.join(OUTPUT_DIR, 'distilbert')
os.makedirs(DISTILBERT_DIR, exist_ok=True)

print(f"\\n✅ Project Root:    {PROJECT_ROOT}")
print(f"✅ Data File Loaded: {DATA_PATH}")
print(f"✅ DistilBERT Dir:   {DISTILBERT_DIR}")
"""

nb.cells[1].source = c1_code

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Updated notebook Cell 1 with intelligent wildcard search.")
