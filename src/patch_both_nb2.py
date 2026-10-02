import os
import nbformat

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files to update
targets = [
    os.path.join(PROJECT_ROOT, 'notebooks', '2_sentiment_analysis.ipynb'),
    os.path.join(PROJECT_ROOT, 'notebooks', '2. sentiment_analysis.ipynb')
]

c9_code = """# =========================================================================
# STEP 4: SAFE IN-MEMORY CLEANUP & CHECKPOINT RECALL
# =========================================================================
import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Safely delete in-memory training objects without crashing if already deleted
for var_name in ['model', 'optimizer', 'best_state_dict']:
    if var_name in globals():
        del globals()[var_name]

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

if 'device' not in globals():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if 'DISTILBERT_DIR' not in globals():
    DISTILBERT_DIR = 'outputs/sentiment_training_2000/distilbert'

print("In-memory training model cleared and garbage collected.")
print(f"Reloading ('recalling') DistilBERT model from disk checkpoint at: {DISTILBERT_DIR}...")

recalled_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR, num_labels=3)
recalled_model.to(device)
reloaded_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)

print(f"\\n✅ DistilBERT model successfully recalled from disk onto {device}!")
"""

for target in targets:
    if os.path.exists(target):
        with open(target, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Find cell with del model
        found = False
        for idx, cell in enumerate(nb.cells):
            if 'del model' in cell.source or 'STEP 4:' in cell.source:
                cell.source = c9_code
                found = True
                print(f"Patched Cell {idx} in {os.path.basename(target)}")
                break
        
        with open(target, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
    else:
        # If one doesn't exist, copy from the other
        src = targets[0] if os.path.exists(targets[0]) else targets[1]
        with open(src, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        with open(target, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Created {os.path.basename(target)} from template.")

print("All target notebooks updated successfully.")
