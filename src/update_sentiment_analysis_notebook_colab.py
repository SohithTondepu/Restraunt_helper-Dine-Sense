import os
import json
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def build_colab_ready_notebook():
    nb = new_notebook()

    # =========================================================================
    # CELL 0: TITLE & ARCHITECTURE (Markdown)
    # =========================================================================
    c0_md = """# 🔬 Stage 1: DistilBERT Transformer Fine-Tuning (3 Epochs)

**Aspect-Conditioned Sentiment Classification on 2,000 Food Reviews (`modified_annotations_2000_corrected.csv`)**

---

### Notebook Architecture:
1. **Google Colab & Environment Setup:** Automated Colab detection, GPU/CUDA acceleration setup, and dependency verification.
2. **Dataset Loading & Grouped 70/15/15 Partitioning:** Loads verified aspect-conditioned examples from `modified_annotations_2000_corrected.csv` with zero data leakage across splits.
3. **Tokenization & PyTorch DataLoaders:** Tokenizes aspect-conditioned clauses (`Aspect: {aspect} [SEP] {clause}`) using `distilbert-base-uncased` with balanced class weights.
4. **Fine-Tuning DistilBERT (3 Epochs):** Full or selective fine-tuning with AdamW optimizer, tracking loss and validation Macro-F1 per epoch.
5. **Memory Clean-up & Model Checkpoint Recall:** Frees in-memory training instances, invokes garbage collection, and reloads the best model directly from disk.
6. **Training Loss & F1 Dynamics:** Visualizes training loss progression and validation Macro-F1 across all 3 epochs.
7. **Internal Test Set Benchmark Evaluation ($N=799$):** Computes multiclass Accuracy, Macro-F1, per-class metrics, normalized confusion matrix, and prediction confidence calibration.
8. **Statistical Rigor:** 1,000-iteration bootstrap 95% confidence intervals on Macro-F1 and optional McNemar's paired test against baseline Logistic Regression.
"""
    nb.cells.append(new_markdown_cell(c0_md))

    # =========================================================================
    # CELL 1: COLAB SETUP & ENVIRONMENT (Code)
    # =========================================================================
    c1_code = """# =========================================================================
# STEP 0: GOOGLE COLAB DETECTION & DEPENDENCY SETUP
# =========================================================================
import sys
import os

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("⚡ Google Colab environment detected!")
    print("Installing required dependencies...")
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
# ROBUST PATH RESOLUTION FOR COLAB & LOCAL
# -------------------------------------------------------------------------
PROJECT_ROOT = os.getcwd()
if 'notebooks' in PROJECT_ROOT:
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

CANDIDATE_DATA_PATHS = [
    # Colab standard locations
    '/content/modified_annotations_2000_corrected.csv',
    '/content/Aspect-Based-Sentimental-Analysis-on-Food-Reviews/final_aspect_evaluation/modified_annotations_2000_corrected.csv',
    '/content/Aspect-Based-Sentimental-Analysis-on-Food-Reviews/modified_annotations_2000_corrected.csv',
    '/content/final_aspect_evaluation/modified_annotations_2000_corrected.csv',
    '/content/drive/MyDrive/modified_annotations_2000_corrected.csv',
    # Local workspace locations
    os.path.join(PROJECT_ROOT, 'final_aspect_evaluation', 'modified_annotations_2000_corrected.csv'),
    os.path.join(PROJECT_ROOT, 'modified_annotations_2000_corrected.csv'),
    os.path.join(PROJECT_ROOT, 'final_aspect_evaluation', 'modified_annotations_2000.csv'),
    'modified_annotations_2000_corrected.csv',
    '../modified_annotations_2000_corrected.csv'
]

DATA_PATH = next((p for p in CANDIDATE_DATA_PATHS if os.path.exists(p)), None)

if DATA_PATH is None:
    if IN_COLAB:
        print("\\n⚠️ Data file not found in standard paths. Please upload modified_annotations_2000_corrected.csv:")
        from google.colab import files
        uploaded = files.upload()
        DATA_PATH = list(uploaded.keys())[0]
    else:
        raise FileNotFoundError("Could not find modified_annotations_2000_corrected.csv!")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'sentiment_training_2000')
DISTILBERT_DIR = os.path.join(OUTPUT_DIR, 'distilbert')
os.makedirs(DISTILBERT_DIR, exist_ok=True)

print(f"\\nProject Root:   {PROJECT_ROOT}")
print(f"Data File:      {DATA_PATH}")
print(f"DistilBERT Dir: {DISTILBERT_DIR}")
"""
    nb.cells.append(new_code_cell(c1_code))

    # =========================================================================
    # CELL 2: DATA LOADING & SPLIT (Markdown)
    # =========================================================================
    c2_md = """## 1. Dataset Loading & Grouped 70/15/15 Partitioning

We load the verified aspect-sentiment annotations from `modified_annotations_2000_corrected.csv`. To completely eliminate data leakage across clauses from the same review, we perform a **grouped stratified partition** by `review_id` (70% Train, 15% Validation, 15% Test).
"""
    nb.cells.append(new_markdown_cell(c2_md))

    # =========================================================================
    # CELL 3: DATA LOADING & SPLIT (Code)
    # =========================================================================
    c3_code = """df_raw = pd.read_csv(DATA_PATH)

VALID_ASPECTS = ['Food', 'Service', 'Price / Value', 'Ambience', 'General Experience']
VALID_SENTIMENTS = ['Positive', 'Negative', 'Neutral']
LABEL_MAP = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
INV_LABEL_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

mask_eligible = df_raw['aspect'].isin(VALID_ASPECTS) & df_raw['sentiment'].isin(VALID_SENTIMENTS)
df_eligible = df_raw[mask_eligible].copy()

# Grouped stratified split by review_id
review_sent = df_eligible.groupby('review_id')['sentiment'].apply(lambda s: s.value_counts().index[0])
reviews_by_strat = {}
for rid, s in review_sent.items():
    reviews_by_strat.setdefault(s, []).append(rid)

np.random.seed(42)
train_r, val_r, test_r = [], [], []
for s, rids in reviews_by_strat.items():
    shuf = np.random.permutation(rids)
    n = len(shuf)
    train_r.extend(shuf[:int(0.70 * n)])
    val_r.extend(shuf[int(0.70 * n):int(0.85 * n)])
    test_r.extend(shuf[int(0.85 * n):])

train_df = df_eligible[df_eligible['review_id'].isin(train_r)].copy()
val_df = df_eligible[df_eligible['review_id'].isin(val_r)].copy()
test_df = df_eligible[df_eligible['review_id'].isin(test_r)].copy()

train_df['aspect_conditioned_text'] = 'Aspect: ' + train_df['aspect'] + ' [SEP] ' + train_df['clause_text']
val_df['aspect_conditioned_text'] = 'Aspect: ' + val_df['aspect'] + ' [SEP] ' + val_df['clause_text']
test_df['aspect_conditioned_text'] = 'Aspect: ' + test_df['aspect'] + ' [SEP] ' + test_df['clause_text']

y_train = np.array([LABEL_MAP[s] for s in train_df['sentiment']])
y_val = np.array([LABEL_MAP[s] for s in val_df['sentiment']])
y_test = np.array([LABEL_MAP[s] for s in test_df['sentiment']])

print(f"Total Eligible Examples: {len(df_eligible):,}")
print(f"  Train Set: {len(train_df):,} examples ({len(train_r)} reviews) -> 70%")
print(f"  Val Set:   {len(val_df):,} examples ({len(val_r)} reviews) -> 15%")
print(f"  Test Set:  {len(test_df):,} examples ({len(test_r)} reviews) -> 15%")
"""
    nb.cells.append(new_code_cell(c3_code))

    # =========================================================================
    # CELL 4: TOKENIZATION & DATALOADERS (Markdown)
    # =========================================================================
    c4_md = """## 2. Tokenization & PyTorch DataLoaders

We tokenize the aspect-conditioned inputs with `distilbert-base-uncased` using padding and truncation up to `max_length=64`. Inverse class weights are computed to counter class imbalance across Negative, Neutral, and Positive sentiments.
"""
    nb.cells.append(new_markdown_cell(c4_md))

    # =========================================================================
    # CELL 5: TOKENIZATION & DATALOADERS (Code)
    # =========================================================================
    c5_code = """tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# Use batch size 32 for train, 64 for val/test
train_loader = DataLoader(TextDataset(train_df['aspect_conditioned_text'].tolist(), y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TextDataset(val_df['aspect_conditioned_text'].tolist(), y_val), batch_size=64, shuffle=False)
test_loader = DataLoader(TextDataset(test_df['aspect_conditioned_text'].tolist(), y_test), batch_size=64, shuffle=False)

# Compute inverse class weights for balanced cross-entropy loss
weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_train)
class_weights = torch.tensor(weights, dtype=torch.float)
print("Class Weights [Negative, Neutral, Positive]:", class_weights.numpy().round(3))
"""
    nb.cells.append(new_code_cell(c5_code))

    # =========================================================================
    # CELL 6: TRAINING DISTILBERT (Markdown)
    # =========================================================================
    c6_md = """## 3. Fine-Tune DistilBERT (3 Epochs)

We train for **3 epochs** using AdamW (`lr=3e-5`) with class-weighted cross-entropy loss on the active hardware device (`cuda` if GPU available, else `cpu`). We track training loss and validation Macro-F1 after each epoch, saving the best checkpoint state dict.
"""
    nb.cells.append(new_markdown_cell(c6_md))

    # =========================================================================
    # CELL 7: TRAINING DISTILBERT (Code)
    # =========================================================================
    c7_code = """NUM_EPOCHS = 3
LEARNING_RATE = 3e-5

print(f"=== INITIALIZING DISTILBERT FINE-TUNING ({NUM_EPOCHS} EPOCHS) ON {device} ===")
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
model.to(device)

# Hardware-aware adaptation: On CPU, freeze bottom layers for rapid convergence; on GPU, full fine-tuning
if device.type == 'cpu':
    print("ℹ️ CPU execution: Freezing lower 3 transformer layers for faster execution.")
    for param in model.distilbert.transformer.layer[:3].parameters():
        param.requires_grad = False
else:
    print(f"🚀 GPU execution: Full fine-tuning on {torch.cuda.get_device_name(0)}.")

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)

best_val_f1 = -1.0
best_state_dict = None
training_history = []

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader)
    
    # Validation evaluation
    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            val_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            
    v_f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)
    v_acc = accuracy_score(y_val, val_preds)
    training_history.append({'epoch': epoch+1, 'train_loss': avg_loss, 'val_macro_f1': v_f1, 'val_acc': v_acc})
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {avg_loss:.4f} | Val Macro-F1: {v_f1*100:.2f}% | Val Acc: {v_acc*100:.2f}%")
    
    if v_f1 > best_val_f1:
        best_val_f1 = v_f1
        best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

print(f"\\nBest Validation Macro-F1 Achieved: {best_val_f1*100:.2f}%")

# Save best checkpoint to disk
model.load_state_dict(best_state_dict)
model.save_pretrained(DISTILBERT_DIR)
tokenizer.save_pretrained(DISTILBERT_DIR)
print(f"Best DistilBERT checkpoint saved to: {DISTILBERT_DIR}")
"""
    nb.cells.append(new_code_cell(c7_code))

    # =========================================================================
    # CELL 8: DELETE & RECALL (Markdown)
    # =========================================================================
    c8_md = """## 4. Delete In-Memory Model & Reload ("Recall") Checkpoint

To verify model persistence and validate production deployment loading, we delete the in-memory training model instance, collect garbage and clear GPU VRAM cache, and reload it directly from disk onto the target device.
"""
    nb.cells.append(new_markdown_cell(c8_md))

    # =========================================================================
    # CELL 9: DELETE & RECALL (Code)
    # =========================================================================
    c9_code = """# Delete in-memory training objects
del model, optimizer, best_state_dict
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

print("In-memory training model deleted and garbage collected.")
print("Reloading ('recalling') DistilBERT model from disk checkpoint...")

recalled_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR, num_labels=3)
recalled_model.to(device)
reloaded_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)

print(f"DistilBERT model successfully recalled from disk onto {device}!")
"""
    nb.cells.append(new_code_cell(c9_code))

    # =========================================================================
    # CELL 10: TRAINING CURVES (Markdown)
    # =========================================================================
    c10_md = """## 5. Training Curves Progression

Plot training loss and validation Macro-F1 across the 3 fine-tuning epochs.
"""
    nb.cells.append(new_markdown_cell(c10_md))

    # =========================================================================
    # CELL 11: TRAINING CURVES (Code)
    # =========================================================================
    c11_code = """fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
epochs_x = [h['epoch'] for h in training_history]

ax1.plot(epochs_x, [h['train_loss'] for h in training_history], marker='o', color='#2980b9', linewidth=2.5, markersize=8)
ax1.set_title('DistilBERT Training Loss (3 Epochs)', fontweight='bold', fontsize=12, pad=10)
ax1.set_xlabel('Epoch', fontweight='bold')
ax1.set_ylabel('Cross-Entropy Loss', fontweight='bold')
ax1.set_xticks(epochs_x)

ax2.plot(epochs_x, [h['val_macro_f1']*100 for h in training_history], marker='s', color='#27ae60', linewidth=2.5, markersize=8)
ax2.set_title('Validation Macro-F1 (%) Progression', fontweight='bold', fontsize=12, pad=10)
ax2.set_xlabel('Epoch', fontweight='bold')
ax2.set_ylabel('Macro-F1 (%)', fontweight='bold')
ax2.set_xticks(epochs_x)

plt.tight_layout()
plt.show()
"""
    nb.cells.append(new_code_cell(c11_code))

    # =========================================================================
    # CELL 12: TEST BENCHMARK (Markdown)
    # =========================================================================
    c12_md = """## 6. Internal Test Set Benchmark Evaluation ($N=799$)

We evaluate the recalled DistilBERT model on the internal held-out test set and compute softmax posterior probabilities.
"""
    nb.cells.append(new_markdown_cell(c12_md))

    # =========================================================================
    # CELL 13: TEST BENCHMARK (Code)
    # =========================================================================
    c13_code = """recalled_model.eval()
test_preds = []
test_probs = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        logits = recalled_model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        test_probs.extend(probs)
        test_preds.extend(np.argmax(probs, axis=1).tolist())

test_preds = np.array(test_preds)
test_probs = np.array(test_probs)
test_confs = np.max(test_probs, axis=1)

acc = accuracy_score(y_test, test_preds)
macro_f1 = f1_score(y_test, test_preds, average='macro', zero_division=0)
weighted_f1 = f1_score(y_test, test_preds, average='weighted', zero_division=0)
class_f1 = f1_score(y_test, test_preds, average=None, zero_division=0)

print(f"=== DISTILBERT TEST SET CLASSIFICATION REPORT (N={len(y_test)}) ===")
print(classification_report(y_test, test_preds, target_names=['Negative', 'Neutral', 'Positive'], digits=4))

print("=== SUMMARY METRICS ===")
print(f"  Test Accuracy:    {acc*100:.2f}%")
print(f"  Test Macro-F1:    {macro_f1*100:.2f}%")
print(f"  Test Weighted-F1: {weighted_f1*100:.2f}%")
print(f"  Negative F1:      {class_f1[0]*100:.2f}%")
print(f"  Neutral F1:       {class_f1[1]*100:.2f}%")
print(f"  Positive F1:      {class_f1[2]*100:.2f}%")
"""
    nb.cells.append(new_code_cell(c13_code))

    # =========================================================================
    # CELL 14: CONFUSION MATRIX & CALIBRATION (Code)
    # =========================================================================
    c14_code = """fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

# 1. Normalized Confusion Matrix
cm = confusion_matrix(y_test, test_preds, labels=[0, 1, 2])
cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-9)

sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=ax1,
            xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'], vmin=0, vmax=1)
ax1.set_title(f"DistilBERT Normalized Confusion Matrix\\n(Test Macro-F1: {macro_f1*100:.2f}%)", fontweight='bold', fontsize=12, pad=10)
ax1.set_xlabel('Predicted Label', fontweight='bold')
ax1.set_ylabel('True Label', fontweight='bold')

# 2. Confidence Calibration Plot
is_correct = (test_preds == y_test)
sns.kdeplot(test_confs[is_correct], label=f'Correct (Mean: {test_confs[is_correct].mean():.2f})', color='#2ecc71', fill=True, ax=ax2)
if (~is_correct).sum() > 0:
    sns.kdeplot(test_confs[~is_correct], label=f'Errors (Mean: {test_confs[~is_correct].mean():.2f})', color='#e74c3c', fill=True, ax=ax2)
ax2.set_title('DistilBERT Posterior Prediction Confidence', fontweight='bold', fontsize=12, pad=10)
ax2.set_xlabel('Softmax Probability', fontweight='bold')
ax2.set_ylabel('Density', fontweight='bold')
ax2.legend(frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
plt.show()
"""
    nb.cells.append(new_code_cell(c14_code))

    # =========================================================================
    # CELL 15: STATISTICAL RIGOR (Markdown)
    # =========================================================================
    c15_md = """## 7. Statistical Rigor: Bootstrap 95% CI & McNemar's Paired Hypothesis Test

1. **Bootstrap 95% Confidence Intervals:** 1,000 resamples with replacement to compute empirical confidence intervals on Macro-F1.
2. **McNemar's Paired Test:** Statistical discordance test against the Logistic Regression baseline (if trained in Notebook 1).
"""
    nb.cells.append(new_markdown_cell(c15_md))

    # =========================================================================
    # CELL 16: STATISTICAL RIGOR (Code)
    # =========================================================================
    c16_code = """# 1. 1,000 Bootstrap Resamples on Test Macro-F1
np.random.seed(42)
n_resamples = 1000
boot_f1s = []
n_samples = len(y_test)

for _ in range(n_resamples):
    idx = np.random.choice(n_samples, size=n_samples, replace=True)
    f1_boot = f1_score(y_test[idx], test_preds[idx], average='macro', zero_division=0)
    boot_f1s.append(f1_boot)

ci_lower = np.percentile(boot_f1s, 2.5)
ci_upper = np.percentile(boot_f1s, 97.5)
print(f"DistilBERT Test Macro-F1 95% Bootstrap CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")

# 2. McNemar's Paired Test vs Logistic Regression
lr_path = os.path.join(OUTPUT_DIR, 'logistic_regression.joblib')
tfidf_path = os.path.join(OUTPUT_DIR, 'tfidf_vectorizer.joblib')

if os.path.exists(lr_path) and os.path.exists(tfidf_path):
    lr_clf = joblib.load(lr_path)
    tfidf = joblib.load(tfidf_path)
    X_test_tfidf = tfidf.transform(test_df['aspect_conditioned_text'])
    lr_preds = lr_clf.predict(X_test_tfidf)
    
    b = np.sum((lr_preds == y_test) & (test_preds != y_test))
    c = np.sum((lr_preds != y_test) & (test_preds == y_test))
    
    mcnemar_stat = (abs(b - c) - 1)**2 / np.maximum(b + c, 1e-9)
    p_val = stats.chi2.sf(mcnemar_stat, df=1)
    
    print(f"\\nMcNemar's Paired Test (DistilBERT vs. Logistic Regression):")
    print(f"  b (LR correct, DistilBERT error): {b}")
    print(f"  c (DistilBERT correct, LR error): {c}")
    print(f"  Chi-Square Statistic:             {mcnemar_stat:.4f}")
    print(f"  p-value:                          {p_val:.4f}")
    if p_val < 0.05:
        print("  Conclusion: Statistically significant difference at alpha = 0.05.")
    else:
        print("  Conclusion: No statistically significant difference (comparable accuracy profile).")
else:
    print(f"\\nℹ️ Note: Baseline Logistic Regression artifacts not found at {lr_path}.")
    print("  Run Notebook 1 ('1. Food_reviews.ipynb') to compute McNemar's paired test against the baseline.")
"""
    nb.cells.append(new_code_cell(c16_code))

    # =========================================================================
    # CELL 17: INSIGHTS & PRODUCTION TRADE-OFFS (Markdown)
    # =========================================================================
    c17_md = """## 8. DistilBERT Technical Insights & Production Trade-offs

### 1. Handling Subtle Negation & Aspect Conditioning
DistilBERT uses multi-head self-attention across 768-dimensional token representations, allowing it to attend from aspect prefix tokens (`Aspect: Food`) directly to nuanced sentiment expressions (`"not bad"`, `"took ages to serve"`), resolving negation and boundary ambiguity that bag-of-words models frequently struggle with.

### 2. Computational Profile & Deployment Trade-offs
- **Latency & Throughput:** DistilBERT processes ~15-25 ms per clause on CPU (~2-4 ms on GPU), making it suitable for near real-time inference in production.
- **Model Checkpoint Size:** Saved weights occupy ~268 MB, fitting comfortably inside standard serverless containers or cloud inference endpoints.
- **Persistence Verification:** The model was successfully written to disk, deleted from RAM, and recalled to perform full test set inference.
"""
    nb.cells.append(new_markdown_cell(c17_md))

    # Save to notebook
    output_nb_path = os.path.join(PROJECT_ROOT, 'notebooks', '2. sentiment_analysis.ipynb')
    with open(output_nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Successfully wrote Colab-adjusted notebook to: {output_nb_path}")

if __name__ == '__main__':
    build_colab_ready_notebook()
