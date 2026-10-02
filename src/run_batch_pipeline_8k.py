"""
Batch Inference Pipeline for Remaining 8,000 Restaurant Reviews
================================================================

This script processes the 8,000 out-of-sample reviews from the raw 10,000 reviews corpus
through the complete DineSense AI ABSA pipeline:
1. Discourse Clause Segmentation (RST concessive satellites & contrastive nuclei)
2. Hybrid Aspect Matching (Tier 1 exact lexicon + Tier 2 dense semantic embeddings)
3. Deep Learning Sentiment Inference (Fine-Tuned DistilBERT with neutral boosting, batched)
4. Consolidation into Full 10,000-Review Assertions Dataset
"""

import os
import sys
import time
import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.aspect_engine import (
    split_into_clauses,
    match_aspect_hybrid,
    get_sentence_model,
    ASPECT_LEXICON,
    ASPECT_DESCRIPTIONS,
    _aspect_names,
    _match_keyword,
    _has_service_latency,
    _GENERIC_SENTIMENT_ONLY,
    _FACTUAL_CONTEXT_REGEX
)
from src.analytics import VALID_ASPECTS

RAW_REVIEWS_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'Restaurant reviews.csv')
ANNOTATIONS_2K_PATH = os.path.join(PROJECT_ROOT, 'final_aspect_evaluation', 'modified_annotations_2000_corrected_mixed.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'distilbert_neutral_boosted')
if not os.path.exists(MODEL_DIR):
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'interview_presentation_package', '05_ml_sentiment_training_notebook', 'saved_models', 'distilbert_neutral_boosted')

OUTPUT_8K_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'predicted_assertions_remaining_8000.csv')
FULL_10K_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'full_10000_reviews_assertions.csv')
PKG_10K_PATH = os.path.join(PROJECT_ROOT, 'interview_presentation_package', '01_datasets_and_preprocessing', 'full_10000_reviews_assertions.csv')

INV_LABEL_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}


def run_batch_pipeline():
    start_time = time.time()
    print("=" * 70)
    print("DineSense AI: Full 8,000 Reviews Batch Inference Pipeline")
    print("=" * 70)

    # 1. Load Datasets
    print(f"\n[Step 1/5] Ingesting review corpus and identifying remaining 8,000 reviews...")
    raw_df = pd.read_csv(RAW_REVIEWS_PATH)
    raw_df['original_row_id'] = raw_df.index
    print(f"Total raw reviews in corpus: {len(raw_df):,}")

    df_2k = pd.read_csv(ANNOTATIONS_2K_PATH)
    annotated_ids = set(df_2k[df_2k['original_row_id'] >= 0]['original_row_id'].unique())
    print(f"Reviews already in 2,000 annotated set: {len(annotated_ids):,}")

    remaining_df = raw_df.loc[~raw_df['original_row_id'].isin(annotated_ids)].copy().reset_index(drop=True)
    print(f"Remaining reviews to process: {len(remaining_df):,}")

    # 2. Discourse Clause Segmentation
    print(f"\n[Step 2/5] Performing RST discourse clause segmentation on {len(remaining_df):,} reviews...")
    t_seg = time.time()
    segmented_records = []

    for _, row in tqdm(remaining_df.iterrows(), total=len(remaining_df), desc="Segmenting reviews"):
        rev_text = str(row.get('Review', '')).strip()
        orig_id = int(row['original_row_id'])
        rest = str(row.get('Restaurant', 'Unknown'))
        rating = row.get('Rating', np.nan)
        time_val = str(row.get('Time', ''))
        reviewer = str(row.get('Reviewer', 'Anonymous'))
        rev_id = f"rev_rem_{orig_id}"

        if not rev_text or rev_text.lower() == 'nan':
            continue

        clauses = split_into_clauses(rev_text)
        for c_idx, clause in enumerate(clauses):
            segmented_records.append({
                'review_id': rev_id,
                'original_row_id': orig_id,
                'clause_id': f"{rev_id}_c{c_idx}",
                'Restaurant': rest,
                'star_rating': rating,
                'Time': time_val,
                'Reviewer': reviewer,
                'clause_text': clause
            })

    df_clauses = pd.DataFrame(segmented_records)
    print(f"Clause segmentation completed in {time.time() - t_seg:.2f}s.")
    print(f"Total extracted clauses across 8,000 reviews: {len(df_clauses):,}")

    # 3. Hybrid Aspect Classification
    print(f"\n[Step 3/5] Classifying operational aspects (Tier 1 Lexicon + Tier 2 Semantic Cosine)...")
    t_asp = time.time()

    matched_assertions = []
    unmatched_for_tier2 = []

    # Fast Tier 1
    for idx, row in tqdm(df_clauses.iterrows(), total=len(df_clauses), desc="Aspect Tier 1 Lexicon"):
        clause = row['clause_text']
        clause_lower = clause.lower()
        words = set(re.findall(r'\b[a-zA-Z]{3,}\b', clause_lower))
        detected = set()

        if _has_service_latency(clause_lower):
            detected.add('Service')

        for aspect, keywords in ASPECT_LEXICON.items():
            if aspect == 'Service' and 'Service' in detected:
                continue
            if any(_match_keyword(kw, words, clause_lower) for kw in keywords):
                detected.add(aspect)

        if detected:
            for asp in detected:
                rec = dict(row)
                rec['aspect'] = asp
                matched_assertions.append(rec)
        else:
            unmatched_for_tier2.append(row)

    print(f"Tier 1 Lexicon matched {len(matched_assertions):,} assertions across clauses.")
    print(f"Clauses passed to Tier 2 Semantic Cosine: {len(unmatched_for_tier2):,}")

    # Tier 2 Semantic Cosine Matcher (Batched)
    if unmatched_for_tier2:
        sent_model, aspect_vecs = get_sentence_model()
        if sent_model is not None and aspect_vecs is not None:
            t2_clauses = [r['clause_text'] for r in unmatched_for_tier2]
            print("Encoding Tier 2 clauses with SentenceTransformer...")
            t2_vecs = sent_model.encode(t2_clauses, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
            sim_matrix = np.dot(t2_vecs, aspect_vecs.T)  # shape: (N, 5)

            for i, row in enumerate(unmatched_for_tier2):
                best_idx = int(np.argmax(sim_matrix[i]))
                best_score = float(sim_matrix[i, best_idx])
                if best_score >= 0.22:
                    best_aspect = _aspect_names[best_idx]
                    cleaned_clause = row['clause_text'].strip()
                    if best_aspect == 'General Experience':
                        if not _GENERIC_SENTIMENT_ONLY.match(cleaned_clause) and not _FACTUAL_CONTEXT_REGEX.match(cleaned_clause):
                            rec = dict(row)
                            rec['aspect'] = best_aspect
                            matched_assertions.append(rec)
                    else:
                        rec = dict(row)
                        rec['aspect'] = best_aspect
                        matched_assertions.append(rec)

    df_assertions_8k = pd.DataFrame(matched_assertions)
    print(f"Total aspect-matched assertions from 8,000 reviews: {len(df_assertions_8k):,}")
    print(f"Aspect matching completed in {time.time() - t_asp:.2f}s.")

    # 4. Deep Learning Sentiment Inference (Batched DistilBERT)
    print(f"\n[Step 4/5] Running fine-tuned DistilBERT sentiment classification (batch size=128)...")
    t_sent = time.time()

    print(f"Loading transformer model from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    all_texts = df_assertions_8k['clause_text'].tolist()
    batch_size = 128
    pred_labels = []
    pred_confs = []

    for i in tqdm(range(0, len(all_texts), batch_size), desc="DistilBERT Inference"):
        batch_texts = all_texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            confs, pred_ids = torch.max(probs, dim=1)

        for p_id, conf in zip(pred_ids.cpu().numpy(), confs.cpu().numpy()):
            pred_labels.append(INV_LABEL_MAP[int(p_id)])
            pred_confs.append(round(float(conf), 3))

    df_assertions_8k['sentiment'] = pred_labels
    df_assertions_8k['confidence'] = pred_confs
    df_assertions_8k['assertion_id'] = [f"ass_rem_{idx}" for idx in range(len(df_assertions_8k))]

    print(f"Sentiment inference completed in {time.time() - t_sent:.2f}s ({len(all_texts)/(time.time() - t_sent):.1f} assertions/sec).")

    # Clean timestamps and YearMonth
    df_assertions_8k['Review_Date'] = pd.to_datetime(df_assertions_8k['Time'], errors='coerce')
    df_assertions_8k['YearMonth'] = df_assertions_8k['Review_Date'].dt.to_period('M').astype(str).replace('NaT', np.nan)

    # Save isolated 8k predictions
    os.makedirs(os.path.dirname(OUTPUT_8K_PATH), exist_ok=True)
    df_assertions_8k.to_csv(OUTPUT_8K_PATH, index=False)
    print(f"Saved 8k assertions to: {OUTPUT_8K_PATH}")

    # 5. Consolidate into Full 10,000 Reviews Dataset
    print(f"\n[Step 5/5] Consolidating 2,000 annotated + 8,000 inferred into Full 10k Assertions Dataset...")

    # Standardize 2k annotations columns
    common_cols = ['assertion_id', 'review_id', 'original_row_id', 'Restaurant', 'clause_text',
                   'aspect', 'sentiment', 'confidence', 'star_rating', 'Time', 'Review_Date', 'YearMonth', 'Reviewer']

    # Ensure 2k dataset has star_rating and confidence
    if 'star_rating' not in df_2k.columns and 'Rating' in df_2k.columns:
        df_2k['star_rating'] = df_2k['Rating']
    if 'confidence' not in df_2k.columns:
        df_2k['confidence'] = 1.0

    # Ensure timestamps for 2k
    if 'Review_Date' not in df_2k.columns:
        raw_meta = raw_df.set_index('original_row_id')
        df_2k['Time'] = df_2k['original_row_id'].map(raw_meta['Time'])
        df_2k['Review_Date'] = pd.to_datetime(df_2k['Time'], errors='coerce')
        df_2k['YearMonth'] = df_2k['Review_Date'].dt.to_period('M').astype(str).replace('NaT', np.nan)
        df_2k['Reviewer'] = df_2k['original_row_id'].map(raw_meta['Reviewer']).fillna('Anonymous')

    df_2k_sub = df_2k[df_2k['aspect'].isin(VALID_ASPECTS)].copy()
    if 'Restaurant' not in df_2k_sub.columns and 'establishment_id' in df_2k_sub.columns:
        df_2k_sub['Restaurant'] = df_2k_sub['establishment_id']

    # Align columns
    for c in common_cols:
        if c not in df_2k_sub.columns:
            df_2k_sub[c] = np.nan
        if c not in df_assertions_8k.columns:
            df_assertions_8k[c] = np.nan

    df_full_10k = pd.concat([df_2k_sub[common_cols], df_assertions_8k[common_cols]], ignore_index=True)

    # Save full 10k assertions
    df_full_10k.to_csv(FULL_10K_PATH, index=False)
    print(f"Saved Full 10k Dataset to: {FULL_10K_PATH}")

    os.makedirs(os.path.dirname(PKG_10K_PATH), exist_ok=True)
    df_full_10k.to_csv(PKG_10K_PATH, index=False)
    print(f"Saved Full 10k Dataset to Presentation Package: {PKG_10K_PATH}")

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("BATCH INFERENCE PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total Pipeline Execution Time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Total Unique Reviews Covered: {df_full_10k['review_id'].nunique():,}")
    print(f"Total Aspect Assertions Produced: {len(df_full_10k):,}")
    print("=" * 70)


if __name__ == '__main__':
    run_batch_pipeline()
