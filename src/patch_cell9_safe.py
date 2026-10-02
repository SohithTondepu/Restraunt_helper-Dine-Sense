import os
import nbformat

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb_path = os.path.join(PROJECT_ROOT, 'notebooks', '2. sentiment_analysis.ipynb')

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

c9_code = """# =========================================================================
# STEP 4: SAFE IN-MEMORY CLEANUP & CHECKPOINT RECALL
# =========================================================================
# 1. Safely delete in-memory training objects without crashing if already deleted
for var_name in ['model', 'optimizer', 'best_state_dict']:
    if var_name in globals():
        del globals()[var_name]

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

print("In-memory training model deleted and garbage collected.")
print(f"Checking disk checkpoint at: {DISTILBERT_DIR}...")

# 2. Verify files exist on disk before loading
if not os.path.exists(DISTILBERT_DIR) or not os.listdir(DISTILBERT_DIR):
    raise FileNotFoundError(
        f"❌ No checkpoint found in {DISTILBERT_DIR}!\\n"
        "Please ensure the training cell (Step 3 / Cell 7) finished all epochs successfully."
    )

print("Files in checkpoint directory:", os.listdir(DISTILBERT_DIR))
print(f"Reloading ('recalling') DistilBERT model onto {device}...")

recalled_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR, num_labels=3)
recalled_model.to(device)
reloaded_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)

print(f"\\n✅ DistilBERT model successfully recalled from disk onto {device}!")
"""

# Cell 9 is the delete & recall cell
nb.cells[9].source = c9_code

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Cell 9 updated with safe cleanup and robust checkpoint loading.")
