import os
import sys
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

PROJECT_ROOT = r'd:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews'
ANN_CSV = os.path.join(PROJECT_ROOT, 'annotations', 'combined_2000', 'annotations_2000_final.csv')
PRED_CSV = os.path.join(PROJECT_ROOT, 'aspect_evaluation', 'aspect_predictions_2000.csv')
OUT_CSV = os.path.join(PROJECT_ROOT, 'hybrid_aligned_annotations.csv')
OUT_CSV_ALT = os.path.join(PROJECT_ROOT, 'annotation_dataset', 'hybrid_aligned_annotations.csv')

ASPECT_CATEGORIES = ['Food', 'Service', 'Price / Value', 'Ambience', 'General Experience']

def main():
    print("Loading human annotations and hybrid predictions...")
    df_ann = pd.read_csv(ANN_CSV)
    df_pred = pd.read_csv(PRED_CSV)

    print(f"Original annotations: {len(df_ann)} rows across {df_ann['clause_id'].nunique()} unique clauses.")
    print(f"Hybrid predictions: {len(df_pred)} rows across {df_pred['clause_id'].nunique()} unique clauses.")

    # Build predictions lookup by (review_id, clause_id)
    pred_dict = {}
    for idx, row in df_pred.iterrows():
        key = (str(row['review_id']), str(row['clause_id']))
        pred_dict[key] = json.loads(row['predicted_aspect_labels'])

    # Group annotations by (review_id, clause_id) preserving encounter order
    ann_by_clause = {}
    for idx, row in df_ann.iterrows():
        key = (str(row['review_id']), str(row['clause_id']))
        if key not in ann_by_clause:
            ann_by_clause[key] = []
        ann_by_clause[key].append(row.to_dict())

    # Generate aligned annotation records
    aligned_records = []
    
    # Detailed change tracking
    clauses_identical = 0
    clauses_modified = 0
    
    clause_change_breakdown = Counter()
    label_additions = Counter()
    label_removals = Counter()
    label_kept = Counter()
    
    for key, orig_rows in ann_by_clause.items():
        rev_id, cid = key
        orig_aspects = set(r['aspect'] for r in orig_rows if pd.notna(r['aspect']))
        preds = pred_dict[key]
        pred_aspects = set(preds)

        # Check if identical
        if orig_aspects == pred_aspects:
            clauses_identical += 1
            for a in orig_aspects:
                label_kept[a] += 1
        else:
            clauses_modified += 1
            # Track additions and removals
            added = pred_aspects - orig_aspects
            removed = orig_aspects - pred_aspects
            kept = orig_aspects & pred_aspects

            for a in added:
                label_additions[a] += 1
            for a in removed:
                label_removals[a] += 1
            for a in kept:
                label_kept[a] += 1

            if len(orig_aspects) == 0 and len(pred_aspects) > 0:
                clause_change_breakdown['No Aspect Opinion -> One or More Aspects'] += 1
            elif len(orig_aspects) > 0 and len(pred_aspects) == 0:
                clause_change_breakdown['One or More Aspects -> No Aspect Opinion'] += 1
            else:
                clause_change_breakdown['Aspect Reassignment / Multi-Aspect Shift'] += 1

        # Build output records
        if len(preds) == 0:
            # Single row with No Aspect Opinion
            base_row = orig_rows[0].copy()
            base_row['assertion_id'] = f"{cid}_A01"
            base_row['aspect'] = np.nan
            base_row['annotation_status'] = 'No Aspect Opinion'
            # Preserve all other fields (review_text, clause_text, offsets, sentiment, etc.)
            aligned_records.append(base_row)
        else:
            for i, asp in enumerate(preds):
                assertion_id = f"{cid}_A{i+1:02d}"
                # If an original row matches this aspect, inherit its specific spans
                matched_orig = next((r for r in orig_rows if r.get('aspect') == asp), None)
                if matched_orig is not None:
                    row = matched_orig.copy()
                else:
                    row = orig_rows[0].copy()
                
                row['assertion_id'] = assertion_id
                row['aspect'] = asp
                row['annotation_status'] = 'Annotated'
                # Sentiment and all other fields are preserved from row
                aligned_records.append(row)

    df_aligned = pd.DataFrame(aligned_records)

    # Reorder columns to exactly match original
    df_aligned = df_aligned[df_ann.columns]

    # Save aligned dataset
    df_aligned.to_csv(OUT_CSV, index=False, encoding='utf-8')
    df_aligned.to_csv(OUT_CSV_ALT, index=False, encoding='utf-8')
    print(f"Saved aligned dataset to: {OUT_CSV}")
    print(f"Also saved copy to: {OUT_CSV_ALT}")
    print(f"Total aligned rows: {len(df_aligned)} (original was {len(df_ann)})")

    # ========================================================
    # RUN EVALUATION ON THE ALIGNED DATASET
    # ========================================================
    print("\n" + "="*60)
    print("RUNNING EVALUATION ON HYBRID ALIGNED DATASET")
    print("="*60)

    # Aggregate clause-level reference aspect labels from aligned dataset
    aligned_clause_labels = {}
    for idx, row in df_aligned.iterrows():
        cid = str(row['clause_id'])
        if cid not in aligned_clause_labels:
            aligned_clause_labels[cid] = set()
        if pd.notna(row['aspect']):
            aligned_clause_labels[cid].add(row['aspect'])

    # Evaluate against predictions across all 11,631 clauses
    category_stats = {}
    for cat in ASPECT_CATEGORIES:
        tp = fp = fn = tn = 0
        for idx, row in df_pred.iterrows():
            cid = str(row['clause_id'])
            ref_aspects = aligned_clause_labels[cid]
            pred_aspects = set(json.loads(row['predicted_aspect_labels']))

            in_ref = cat in ref_aspects
            in_pred = cat in pred_aspects

            if in_ref and in_pred:
                tp += 1
            elif not in_ref and in_pred:
                fp += 1
            elif in_ref and not in_pred:
                fn += 1
            else:
                tn += 1

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        support = tp + fn

        category_stats[cat] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'Support': support
        }

    # No Aspect Opinion
    tp_no = fp_no = fn_no = tn_no = 0
    exact_matches = 0
    jaccard_scores = []

    for idx, row in df_pred.iterrows():
        cid = str(row['clause_id'])
        ref_aspects = aligned_clause_labels[cid]
        pred_aspects = set(json.loads(row['predicted_aspect_labels']))

        is_ref_no = len(ref_aspects) == 0
        is_pred_no = len(pred_aspects) == 0

        if is_ref_no and is_pred_no:
            tp_no += 1
        elif not is_ref_no and is_pred_no:
            fp_no += 1
        elif is_ref_no and not is_pred_no:
            fn_no += 1
        else:
            tn_no += 1

        if ref_aspects == pred_aspects:
            exact_matches += 1

        if not ref_aspects and not pred_aspects:
            jaccard = 1.0
        elif not ref_aspects or not pred_aspects:
            jaccard = 0.0
        else:
            jaccard = len(ref_aspects & pred_aspects) / len(ref_aspects | pred_aspects)
        jaccard_scores.append(jaccard)

    prec_no = tp_no / (tp_no + fp_no) if (tp_no + fp_no) > 0 else 0.0
    rec_no = tp_no / (tp_no + fn_no) if (tp_no + fn_no) > 0 else 0.0
    f1_no = (2 * prec_no * rec_no) / (prec_no + rec_no) if (prec_no + rec_no) > 0 else 0.0

    category_stats['No Aspect Opinion'] = {
        'TP': tp_no,
        'FP': fp_no,
        'FN': fn_no,
        'TN': tn_no,
        'Precision': round(prec_no, 4),
        'Recall': round(rec_no, 4),
        'F1-Score': round(f1_no, 4),
        'Support': tp_no + fn_no
    }

    # Macro and Micro averages for 5 aspects
    macro_prec = np.mean([category_stats[c]['Precision'] for c in ASPECT_CATEGORIES])
    macro_rec = np.mean([category_stats[c]['Recall'] for c in ASPECT_CATEGORIES])
    macro_f1 = np.mean([category_stats[c]['F1-Score'] for c in ASPECT_CATEGORIES])

    total_tp = sum(category_stats[c]['TP'] for c in ASPECT_CATEGORIES)
    total_fp = sum(category_stats[c]['FP'] for c in ASPECT_CATEGORIES)
    total_fn = sum(category_stats[c]['FN'] for c in ASPECT_CATEGORIES)
    total_support = sum(category_stats[c]['Support'] for c in ASPECT_CATEGORIES)

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    print("\n--- DETAILED EVALUATION METRICS ---")
    metrics_summary = []
    for cat in ASPECT_CATEGORIES + ['No Aspect Opinion']:
        s = category_stats[cat]
        metrics_summary.append({
            'Category': cat,
            'TP': s['TP'],
            'FP': s['FP'],
            'FN': s['FN'],
            'TN': s['TN'],
            'Precision': s['Precision'],
            'Recall': s['Recall'],
            'F1-Score': s['F1-Score'],
            'Support': s['Support']
        })
        print(f"{cat:<20}: P={s['Precision']:.4f}, R={s['Recall']:.4f}, F1={s['F1-Score']:.4f}, Supp={s['Support']} (TP={s['TP']}, FP={s['FP']}, FN={s['FN']}, TN={s['TN']})")

    print(f"\nMacro Average (5 Aspects): P={macro_prec:.4f}, R={macro_rec:.4f}, F1={macro_f1:.4f}, Support={total_support}")
    print(f"Micro Average (5 Aspects): P={micro_prec:.4f}, R={micro_rec:.4f}, F1={micro_f1:.4f}, Support={total_support}")
    print(f"Exact Match Ratio: {exact_matches / len(df_pred):.4f} ({exact_matches}/{len(df_pred)})")
    print(f"Mean Jaccard Similarity: {np.mean(jaccard_scores):.4f}")

    # Output CSV for metrics
    report_df = pd.DataFrame(metrics_summary)
    report_df.loc[len(report_df)] = {
        'Category': 'Macro Average (5 Aspects)',
        'TP': total_tp, 'FP': total_fp, 'FN': total_fn, 'TN': '',
        'Precision': round(macro_prec, 4), 'Recall': round(macro_rec, 4), 'F1-Score': round(macro_f1, 4),
        'Support': total_support
    }
    report_df.loc[len(report_df)] = {
        'Category': 'Micro Average (5 Aspects)',
        'TP': total_tp, 'FP': total_fp, 'FN': total_fn, 'TN': '',
        'Precision': round(micro_prec, 4), 'Recall': round(micro_rec, 4), 'F1-Score': round(micro_f1, 4),
        'Support': total_support
    }
    metrics_csv = os.path.join(PROJECT_ROOT, 'aspect_evaluation', 'csv', 'aligned_aspect_classification_report.csv')
    report_df.to_csv(metrics_csv, index=False, encoding='utf-8')
    print(f"\nSaved aligned metrics report to: {metrics_csv}")

    # Print summary statistics on modifications
    print("\n" + "="*60)
    print("MODIFICATION STATISTICS SUMMARY")
    print("="*60)
    print(f"Total Unique Clauses: {len(df_pred)}")
    print(f"Clauses Identical: {clauses_identical} ({clauses_identical/len(df_pred)*100:.2f}%)")
    print(f"Clauses Modified:  {clauses_modified} ({clauses_modified/len(df_pred)*100:.2f}%)")
    print("\nClause Modification Breakdown:")
    for k, v in clause_change_breakdown.items():
        print(f"  - {k}: {v} ({v/len(df_pred)*100:.2f}%)")
    
    print("\nAssertion Row Counts:")
    print(f"  - Original Assertion Rows: {len(df_ann)}")
    print(f"  - Aligned Assertion Rows:  {len(df_aligned)} (net change: +{len(df_aligned) - len(df_ann)})")
    
    print("\nAspect Label Changes by Category:")
    all_cats = ASPECT_CATEGORIES
    print(f"{'Category':<20} | {'Kept':<8} | {'Added':<8} | {'Removed':<8} | {'Orig Supp':<10} | {'New Supp':<10}")
    print("-" * 75)
    
    # Original supports
    orig_supp = Counter()
    for idx, row in df_ann.iterrows():
        if pd.notna(row['aspect']):
            orig_supp[row['aspect']] += 1

    for cat in all_cats:
        k = label_kept[cat]
        a = label_additions[cat]
        r = label_removals[cat]
        osup = orig_supp[cat]
        nsup = category_stats[cat]['Support']
        print(f"{cat:<20} | {k:<8} | {a:<8} | {r:<8} | {osup:<10} | {nsup:<10}")

    print("\nSentiment Label Verification:")
    print(f"  - Original sentiment distribution:\n{df_ann['sentiment'].value_counts(dropna=False)}")
    print(f"  - Aligned sentiment distribution:\n{df_aligned['sentiment'].value_counts(dropna=False)}")

if __name__ == '__main__':
    main()
