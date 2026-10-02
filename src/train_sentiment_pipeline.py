import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set thread count for CPU acceleration
torch.set_num_threads(16)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'final_aspect_evaluation', 'modified_annotations_2000.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'sentiment_training_2000')
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_ASPECTS = ['Food', 'Service', 'Price / Value', 'Ambience', 'General Experience']
VALID_SENTIMENTS = ['Positive', 'Negative', 'Neutral']
LABEL_MAP = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
INV_LABEL_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

def run_pipeline():
    print("="*75)
    print("1. DATASET INSPECTION & ELIGIBLE EXAMPLE EXTRACTION")
    print("="*75)
    df_raw = pd.read_csv(DATA_PATH)
    total_raw_rows = len(df_raw)
    unique_reviews_raw = df_raw['review_id'].nunique()
    unique_clauses_raw = df_raw['clause_id'].nunique()

    print(f"Total raw rows:       {total_raw_rows:,}")
    print(f"Unique reviews:       {unique_reviews_raw:,}")
    print(f"Unique clauses:       {unique_clauses_raw:,}")

    # Exclusions
    mask_no_aspect = (df_raw['annotation_status'] == 'No Aspect Opinion') | df_raw['aspect'].isna()
    mask_mixed = df_raw['sentiment'] == 'Mixed'
    mask_null_sentiment = df_raw['sentiment'].isna()
    mask_invalid_aspect = ~df_raw['aspect'].isin(VALID_ASPECTS) & df_raw['aspect'].notna()

    mask_eligible = df_raw['aspect'].isin(VALID_ASPECTS) & df_raw['sentiment'].isin(VALID_SENTIMENTS)
    df_eligible = df_raw[mask_eligible].copy()

    print(f"Excluded No Aspect Opinion rows:        {mask_no_aspect.sum():,}")
    print(f"Excluded Null sentiment rows:           {mask_null_sentiment.sum():,}")
    print(f"Excluded Mixed sentiment rows:          {mask_mixed.sum():,}")
    print(f"Total excluded rows:                    {total_raw_rows - len(df_eligible):,}")
    print(f"Total ELIGIBLE training examples:       {len(df_eligible):,}")
    print(f"Unique reviews in eligible dataset:     {df_eligible['review_id'].nunique():,}")
    print(f"Unique clauses in eligible dataset:     {df_eligible['clause_id'].nunique():,}")

    print("\nEligible Sentiment Distribution:")
    for s, c in df_eligible['sentiment'].value_counts().items():
        print(f"  - {s:<10}: {c:>4} ({c/len(df_eligible)*100:.2f}%)")

    print("\nEligible Aspect Distribution:")
    for a, c in df_eligible['aspect'].value_counts().items():
        print(f"  - {a:<20}: {c:>4} ({c/len(df_eligible)*100:.2f}%)")

    # ==========================================
    # 2. GROUPED STRATIFIED SPLITTING (70/15/15)
    # ==========================================
    print("\n" + "="*75)
    print("2. REPRODUCIBLE GROUPED SPLIT (BY REVIEW_ID)")
    print("="*75)
    review_sent = df_eligible.groupby('review_id')['sentiment'].apply(lambda s: s.value_counts().index[0])
    reviews_by_strat = {}
    for rid, s in review_sent.items():
        reviews_by_strat.setdefault(s, []).append(rid)

    np.random.seed(42)
    train_r, val_r, test_r = [], [], []
    for s, rids in reviews_by_strat.items():
        shuf = np.random.permutation(rids)
        n = len(shuf)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        train_r.extend(shuf[:n_train])
        val_r.extend(shuf[n_train:n_train+n_val])
        test_r.extend(shuf[n_train+n_val:])

    train_df = df_eligible[df_eligible['review_id'].isin(train_r)].copy()
    val_df = df_eligible[df_eligible['review_id'].isin(val_r)].copy()
    test_df = df_eligible[df_eligible['review_id'].isin(test_r)].copy()

    # Input construction: Aspect: {aspect} [SEP] {clause}
    train_df['aspect_conditioned_text'] = 'Aspect: ' + train_df['aspect'] + ' [SEP] ' + train_df['clause_text']
    val_df['aspect_conditioned_text'] = 'Aspect: ' + val_df['aspect'] + ' [SEP] ' + val_df['clause_text']
    test_df['aspect_conditioned_text'] = 'Aspect: ' + test_df['aspect'] + ' [SEP] ' + test_df['clause_text']

    # Save splits
    train_csv = os.path.join(OUTPUT_DIR, 'train.csv')
    val_csv = os.path.join(OUTPUT_DIR, 'validation.csv')
    test_csv = os.path.join(OUTPUT_DIR, 'test.csv')
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    split_manifest = {
        'seed': 42,
        'train_review_count': len(train_r),
        'val_review_count': len(val_r),
        'test_review_count': len(test_r),
        'train_row_count': len(train_df),
        'val_row_count': len(val_df),
        'test_row_count': len(test_df),
        'train_sentiment_distribution': train_df['sentiment'].value_counts().to_dict(),
        'val_sentiment_distribution': val_df['sentiment'].value_counts().to_dict(),
        'test_sentiment_distribution': test_df['sentiment'].value_counts().to_dict(),
        'train_review_ids': sorted(list(train_r)),
        'val_review_ids': sorted(list(val_r)),
        'test_review_ids': sorted(list(test_r))
    }
    with open(os.path.join(OUTPUT_DIR, 'split_manifest.json'), 'w') as f:
        json.dump(split_manifest, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, 'label_mapping.json'), 'w') as f:
        json.dump(LABEL_MAP, f, indent=2)

    print(f"Train split: {len(train_r)} reviews, {len(train_df)} rows ({len(train_df)/len(df_eligible)*100:.1f}%)")
    print(f"Val split:   {len(val_r)} reviews, {len(val_df)} rows ({len(val_df)/len(df_eligible)*100:.1f}%)")
    print(f"Test split:  {len(test_r)} reviews, {len(test_df)} rows ({len(test_df)/len(df_eligible)*100:.1f}%)")
    print(f"Saved splits and split_manifest.json to {OUTPUT_DIR}")

    # ==========================================
    # 3. TF-IDF VECTORIZATION
    # ==========================================
    print("\n" + "="*75)
    print("3. TF-IDF VECTORIZATION (TRAIN-ONLY FIT)")
    print("="*75)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True, min_df=2)
    X_train = tfidf.fit_transform(train_df['aspect_conditioned_text'])
    X_val = tfidf.transform(val_df['aspect_conditioned_text'])
    X_test = tfidf.transform(test_df['aspect_conditioned_text'])

    joblib.dump(tfidf, os.path.join(OUTPUT_DIR, 'tfidf_vectorizer.joblib'))
    print(f"Fitted TF-IDF vocabulary size: {len(tfidf.vocabulary_):,} features")

    y_train = np.array([LABEL_MAP[s] for s in train_df['sentiment']])
    y_val = np.array([LABEL_MAP[s] for s in val_df['sentiment']])
    y_test = np.array([LABEL_MAP[s] for s in test_df['sentiment']])

    models = {}
    val_f1s = {}
    train_times = {}

    # ==========================================
    # 4. MODEL 1: LOGISTIC REGRESSION
    # ==========================================
    print("\n" + "="*75)
    print("4. TRAINING MODEL 1: LOGISTIC REGRESSION (TUNED ON VALIDATION)")
    print("="*75)
    best_c, best_f1, best_lr = None, -1, None
    t0 = time.time()
    for c in [0.1, 0.5, 1.0, 2.0, 5.0]:
        clf = LogisticRegression(C=c, class_weight='balanced', max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        pred_val = clf.predict(X_val)
        f1 = f1_score(y_val, pred_val, average='macro')
        print(f"  C={c:<4} -> Val Macro-F1 = {f1*100:.2f}%")
        if f1 > best_f1:
            best_f1 = f1
            best_c = c
            best_lr = clf
    train_times['Logistic Regression'] = time.time() - t0
    val_f1s['Logistic Regression'] = best_f1
    models['Logistic Regression'] = best_lr
    joblib.dump(best_lr, os.path.join(OUTPUT_DIR, 'logistic_regression.joblib'))
    print(f"Selected Best LR (C={best_c}): Val Macro-F1 = {best_f1*100:.2f}%")

    # ==========================================
    # 5. MODEL 2: LINEAR SVM
    # ==========================================
    print("\n" + "="*75)
    print("5. TRAINING MODEL 2: LINEAR SVM (TUNED ON VALIDATION)")
    print("="*75)
    best_c, best_f1, best_svm = None, -1, None
    t0 = time.time()
    for c in [0.05, 0.1, 0.2, 0.5, 1.0]:
        clf = LinearSVC(C=c, class_weight='balanced', max_iter=2000, random_state=42)
        clf.fit(X_train, y_train)
        pred_val = clf.predict(X_val)
        f1 = f1_score(y_val, pred_val, average='macro')
        print(f"  C={c:<4} -> Val Macro-F1 = {f1*100:.2f}%")
        if f1 > best_f1:
            best_f1 = f1
            best_c = c
            best_svm = clf
    train_times['Linear SVM'] = time.time() - t0
    val_f1s['Linear SVM'] = best_f1
    models['Linear SVM'] = best_svm
    joblib.dump(best_svm, os.path.join(OUTPUT_DIR, 'linear_svm.joblib'))
    print(f"Selected Best Linear SVM (C={best_c}): Val Macro-F1 = {best_f1*100:.2f}%")

    # ==========================================
    # 6. MODEL 3: RANDOM FOREST
    # ==========================================
    print("\n" + "="*75)
    print("6. TRAINING MODEL 3: RANDOM FOREST (TUNED ON VALIDATION)")
    print("="*75)
    best_rf_params, best_f1, best_rf = None, -1, None
    t0 = time.time()
    for d in [10, 20]:
        for n in [100, 200]:
            clf = RandomForestClassifier(n_estimators=n, max_depth=d, class_weight='balanced', random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)
            pred_val = clf.predict(X_val)
            f1 = f1_score(y_val, pred_val, average='macro')
            print(f"  depth={str(d):<4} n_est={n:<4} -> Val Macro-F1 = {f1*100:.2f}%")
            if f1 > best_f1:
                best_f1 = f1
                best_rf_params = (d, n)
                best_rf = clf
    train_times['Random Forest'] = time.time() - t0
    val_f1s['Random Forest'] = best_f1
    models['Random Forest'] = best_rf
    joblib.dump(best_rf, os.path.join(OUTPUT_DIR, 'random_forest.joblib'))
    print(f"Selected Best Random Forest (depth={best_rf_params[0]}, n={best_rf_params[1]}): Val Macro-F1 = {best_f1*100:.2f}%")

    # ==========================================
    # 7. MODEL 4: XGBOOST
    # ==========================================
    print("\n" + "="*75)
    print("7. TRAINING MODEL 4: XGBOOST (TUNED ON VALIDATION)")
    print("="*75)
    sample_weights_train = compute_sample_weight('balanced', y_train)
    best_xgb_params, best_f1, best_xgb = None, -1, None
    t0 = time.time()
    for lr in [0.1, 0.2]:
        for d in [4, 6]:
            clf = XGBClassifier(n_estimators=150, max_depth=d, learning_rate=lr, random_state=42, n_jobs=-1, objective='multi:softprob')
            clf.fit(X_train, y_train, sample_weight=sample_weights_train)
            pred_val = clf.predict(X_val)
            f1 = f1_score(y_val, pred_val, average='macro')
            print(f"  lr={lr:<4} depth={d:<4} -> Val Macro-F1 = {f1*100:.2f}%")
            if f1 > best_f1:
                best_f1 = f1
                best_xgb_params = (lr, d)
                best_xgb = clf
    train_times['XGBoost'] = time.time() - t0
    val_f1s['XGBoost'] = best_f1
    models['XGBoost'] = best_xgb
    joblib.dump(best_xgb, os.path.join(OUTPUT_DIR, 'xgboost.joblib'))
    print(f"Selected Best XGBoost (lr={best_xgb_params[0]}, depth={best_xgb_params[1]}): Val Macro-F1 = {best_f1*100:.2f}%")

    # ==========================================
    # 8. MODEL 5: DISTILBERT
    # ==========================================
    print("\n" + "="*75)
    print("8. TRAINING MODEL 5: DISTILBERT (CHECKPOINT SELECTION ON VALIDATION MACRO-F1)")
    print("="*75)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

    # Freeze lower 3 layers to make CPU fine-tuning fast & stable
    for param in model.distilbert.transformer.layer[:3].parameters():
        param.requires_grad = False

    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(texts, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
            self.labels = torch.tensor(labels)
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item['labels'] = self.labels[idx]
            return item

    train_loader = DataLoader(TextDataset(train_df['aspect_conditioned_text'].tolist(), y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TextDataset(val_df['aspect_conditioned_text'].tolist(), y_val), batch_size=64, shuffle=False)
    test_loader = DataLoader(TextDataset(test_df['aspect_conditioned_text'].tolist(), y_test), batch_size=64, shuffle=False)

    weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5)

    best_distil_f1 = -1
    best_distil_state = None
    distil_history = []

    for epoch in range(2):
        model.train()
        total_loss = 0
        ep_t0 = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            loss = criterion(out.logits, batch['labels'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        ep_time = time.time() - ep_t0

        # Validate
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
                val_preds.extend(torch.argmax(logits, dim=1).tolist())
        v_f1 = f1_score(y_val, val_preds, average='macro')
        v_acc = accuracy_score(y_val, val_preds)
        distil_history.append({'epoch': epoch+1, 'train_loss': avg_loss, 'val_macro_f1': v_f1, 'val_accuracy': v_acc, 'epoch_time': ep_time})
        print(f"  Epoch {epoch+1} ({ep_time:.1f}s) -> Loss: {avg_loss:.4f}, Val Macro-F1: {v_f1*100:.2f}%, Val Acc: {v_acc*100:.2f}%")

        if v_f1 > best_distil_f1:
            best_distil_f1 = v_f1
            best_distil_state = {k: v.cpu() for k, v in model.state_dict().items()}

    train_times['DistilBERT'] = time.time() - t0
    val_f1s['DistilBERT'] = best_distil_f1

    # Load best checkpoint
    model.load_state_dict(best_distil_state)
    distilbert_dir = os.path.join(OUTPUT_DIR, 'distilbert')
    os.makedirs(distilbert_dir, exist_ok=True)
    model.save_pretrained(distilbert_dir)
    tokenizer.save_pretrained(distilbert_dir)
    print(f"Selected Best DistilBERT checkpoint saved to {distilbert_dir} (Val Macro-F1 = {best_distil_f1*100:.2f}%)")

    # Predict test set for DistilBERT
    model.eval()
    distil_test_preds = []
    distil_test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
            probs = torch.softmax(logits, dim=1).numpy()
            distil_test_probs.extend(probs)
            distil_test_preds.extend(np.argmax(probs, axis=1).tolist())

    # ==========================================
    # 9. EVALUATION ON INTERNAL TEST SET
    # ==========================================
    print("\n" + "="*75)
    print("9. FINAL EVALUATION ON FIXED INTERNAL TEST SET (N=799)")
    print("="*75)
    model_names = ['Logistic Regression', 'Linear SVM', 'Random Forest', 'XGBoost', 'DistilBERT']
    test_results = []
    all_test_preds = {}
    all_test_confs = {}

    for name in model_names:
        if name == 'DistilBERT':
            preds = np.array(distil_test_preds)
            confs = np.max(distil_test_probs, axis=1)
        else:
            clf = models[name]
            preds = clf.predict(X_test)
            if hasattr(clf, 'predict_proba'):
                confs = np.max(clf.predict_proba(X_test), axis=1)
            elif hasattr(clf, 'decision_function'):
                # Calibrate decision scores via softmax for SVM confidence
                df_scores = clf.decision_function(X_test)
                exp_s = np.exp(df_scores - np.max(df_scores, axis=1, keepdims=True))
                probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
                confs = np.max(probs, axis=1)
            else:
                confs = np.ones(len(preds))

        all_test_preds[name] = preds
        all_test_confs[name] = confs

        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average='macro')
        weighted_f1 = f1_score(y_test, preds, average='weighted')
        class_f1 = f1_score(y_test, preds, average=None)

        test_results.append({
            'Model': name,
            'Val_Macro_F1': round(val_f1s[name], 4),
            'Test_Accuracy': round(acc, 4),
            'Test_Macro_F1': round(macro_f1, 4),
            'Test_Weighted_F1': round(weighted_f1, 4),
            'Negative_F1': round(class_f1[0], 4),
            'Neutral_F1': round(class_f1[1], 4),
            'Positive_F1': round(class_f1[2], 4),
            'Training_Time_s': round(train_times[name], 1)
        })

    df_results = pd.DataFrame(test_results)
    print(df_results.to_string(index=False))

    df_results.to_csv(os.path.join(OUTPUT_DIR, 'metrics_comparison.csv'), index=False)
    with open(os.path.join(OUTPUT_DIR, 'metrics_comparison.json'), 'w') as f:
        json.dump(test_results, f, indent=2)

    # ==========================================
    # 10. ERROR ANALYSIS DATASET CREATION
    # ==========================================
    print("\n" + "="*75)
    print("10. CREATING ERROR ANALYSIS DATASET")
    print("="*75)
    df_err = test_df[['review_id', 'clause_id', 'aspect', 'clause_text', 'sentiment']].copy()
    df_err.rename(columns={'sentiment': 'true_sentiment'}, inplace=True)

    for name in model_names:
        df_err[f'pred_{name.lower().replace(" ", "_")}'] = [INV_LABEL_MAP[p] for p in all_test_preds[name]]
        df_err[f'conf_{name.lower().replace(" ", "_")}'] = [round(float(c), 3) for c in all_test_confs[name]]

    error_csv = os.path.join(OUTPUT_DIR, 'error_analysis.csv')
    df_err.to_csv(error_csv, index=False)
    print(f"Saved test error analysis table ({len(df_err)} rows) to: {error_csv}")

    # ==========================================
    # 11. GENERATE VISUALIZATIONS
    # ==========================================
    print("\n" + "="*75)
    print("11. SAVING VISUALIZATIONS & CHARTS")
    print("="*75)
    # 1. Confusion Matrices
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.8))
    for idx, name in enumerate(model_names):
        cm = confusion_matrix(y_test, all_test_preds[name], labels=[0, 1, 2])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[idx],
                    xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'])
        axes[idx].set_title(f"{name}\\n(Macro-F1: {test_results[idx]['Test_Macro_F1']*100:.1f}%)", fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        if idx == 0:
            axes[idx].set_ylabel('True')
        else:
            axes[idx].set_ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'), dpi=300)
    plt.close()

    # 2. Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(model_names))
    w = 0.35
    ax.bar(x - w/2, [r['Test_Accuracy']*100 for r in test_results], w, label='Accuracy (%)', color='#3498db')
    ax.bar(x + w/2, [r['Test_Macro_F1']*100 for r in test_results], w, label='Macro-F1 (%)', color='#e67e22')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontweight='bold')
    ax.set_ylabel('Score (%)', fontweight='bold')
    ax.set_title('Sentiment Classification Performance Comparison (Test Set)', fontweight='bold', pad=12)
    ax.set_ylim(0, 110)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison_chart.png'), dpi=300)
    plt.close()

    # 3. Per-Class F1 Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(model_names))
    w = 0.25
    ax.bar(x - w, [r['Negative_F1']*100 for r in test_results], w, label='Negative F1', color='#e74c3c')
    ax.bar(x, [r['Neutral_F1']*100 for r in test_results], w, label='Neutral F1', color='#95a5a6')
    ax.bar(x + w, [r['Positive_F1']*100 for r in test_results], w, label='Positive F1', color='#2ecc71')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontweight='bold')
    ax.set_ylabel('F1 Score (%)', fontweight='bold')
    ax.set_title('Per-Class F1 Comparison across Sentiment Models', fontweight='bold', pad=12)
    ax.set_ylim(0, 110)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'per_class_f1_comparison.png'), dpi=300)
    plt.close()

    # 4. DistilBERT Training Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [h['epoch'] for h in distil_history]
    ax1.plot(epochs, [h['train_loss'] for h in distil_history], marker='o', color='#2980b9', linewidth=2)
    ax1.set_title('DistilBERT Training Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_xticks(epochs)

    ax2.plot(epochs, [h['val_macro_f1']*100 for h in distil_history], marker='s', color='#27ae60', linewidth=2)
    ax2.set_title('DistilBERT Validation Macro-F1 (%)', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro-F1 (%)')
    ax2.set_xticks(epochs)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'distilbert_training_curve.png'), dpi=300)
    plt.close()

    print(f"All 4 visual charts saved to {OUTPUT_DIR}")

    # ==========================================
    # 12. SAVE COMPLETE TRAINING CONFIGURATION
    # ==========================================
    config = {
        'task': 'aspect-conditioned clause-level 3-class sentiment classification',
        'input_format': 'Aspect: {aspect} [SEP] {clause}',
        'classes': ['Negative', 'Neutral', 'Positive'],
        'random_seed': 42,
        'split_ratio': {'train': 0.70, 'val': 0.15, 'test': 0.15},
        'eligible_dataset_size': len(df_eligible),
        'hardware_used': '16-Core CPU (Intel/AMD x86_64)',
        'models_trained': model_names,
        'hyperparameters': {
            'Logistic Regression': {'C': best_c, 'class_weight': 'balanced', 'solver': 'lbfgs'},
            'Linear SVM': {'C': best_c, 'class_weight': 'balanced'},
            'Random Forest': {'n_estimators': best_rf_params[1], 'max_depth': best_rf_params[0], 'class_weight': 'balanced'},
            'XGBoost': {'learning_rate': best_xgb_params[0], 'max_depth': best_xgb_params[1], 'n_estimators': 150},
            'DistilBERT': {'learning_rate': 3e-5, 'batch_size': 32, 'max_length': 64, 'epochs': 2, 'frozen_layers': 3}
        },
        'test_metrics': test_results
    }
    with open(os.path.join(OUTPUT_DIR, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "="*75)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*75)

if __name__ == '__main__':
    run_pipeline()
