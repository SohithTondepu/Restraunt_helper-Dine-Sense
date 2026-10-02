import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

INPUT_CLAUSES_CSV = os.path.join(CURRENT_DIR, 'clause_evaluation_input_2000.csv')
PREDICTIONS_CSV = os.path.join(CURRENT_DIR, 'predicted_aspects_2000.csv')
REPORT_CSV = os.path.join(CURRENT_DIR, 'aspect_classification_report.csv')
SUMMARY_CSV = os.path.join(CURRENT_DIR, 'aspect_overall_summary.csv')

ASPECT_CATEGORIES = ['Food', 'Service', 'Price / Value', 'Ambience', 'General Experience']

def evaluate(recompute=False):
    if recompute:
        from src.aspect_engine import match_aspect_hybrid, get_sentence_model
        print(f"Loading input clauses from: {INPUT_CLAUSES_CSV}")
        df = pd.read_csv(INPUT_CLAUSES_CSV)
        get_sentence_model()
        print(f"Running inference across {len(df)} clauses...")
        preds_list = []
        for idx, row in df.iterrows():
            text = str(row['clause_text'])
            preds = sorted(list(match_aspect_hybrid(text)))
            preds_list.append(preds)
        df_eval = df.copy()
        df_eval['predicted_aspect_labels'] = [json.dumps(p) for p in preds_list]
        df_eval['predicted_aspects_display'] = ['; '.join(p) if p else 'No Aspect Opinion' for p in preds_list]
    else:
        print(f"Loading existing predictions from: {PREDICTIONS_CSV}")
        df_eval = pd.read_csv(PREDICTIONS_CSV)

    print(f"Total clauses evaluated: {len(df_eval)}")

    # Parse reference and predicted sets
    ref_sets = [set(json.loads(r)) for r in df_eval['reference_aspect_labels']]
    pred_sets = [set(json.loads(p)) for p in df_eval['predicted_aspect_labels']]

    # Category-level metrics
    cat_stats = []
    total_tp = total_fp = total_fn = 0
    f1_scores = []
    prec_scores = []
    rec_scores = []
    total_support = 0

    for cat in ASPECT_CATEGORIES:
        tp = fp = fn = tn = 0
        for r, p in zip(ref_sets, pred_sets):
            in_ref = cat in r
            in_pred = cat in p
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
        supp = tp + fn

        cat_stats.append({
            'Category': cat,
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'Support': supp
        })
        f1_scores.append(f1)
        prec_scores.append(prec)
        rec_scores.append(rec)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_support += supp

    # No Aspect Opinion
    tp_no = fp_no = fn_no = tn_no = 0
    for r, p in zip(ref_sets, pred_sets):
        is_ref_no = len(r) == 0
        is_pred_no = len(p) == 0
        if is_ref_no and is_pred_no:
            tp_no += 1
        elif not is_ref_no and is_pred_no:
            fp_no += 1
        elif is_ref_no and not is_pred_no:
            fn_no += 1
        else:
            tn_no += 1

    prec_no = tp_no / (tp_no + fp_no) if (tp_no + fp_no) > 0 else 0.0
    rec_no = tp_no / (tp_no + fn_no) if (tp_no + fn_no) > 0 else 0.0
    f1_no = (2 * prec_no * rec_no) / (prec_no + rec_no) if (prec_no + rec_no) > 0 else 0.0

    cat_stats.append({
        'Category': 'No Aspect Opinion',
        'TP': tp_no, 'FP': fp_no, 'FN': fn_no, 'TN': tn_no,
        'Precision': round(prec_no, 4),
        'Recall': round(rec_no, 4),
        'F1-Score': round(f1_no, 4),
        'Support': tp_no + fn_no
    })

    # Macro & Micro averages across 5 aspects
    macro_prec = np.mean(prec_scores)
    macro_rec = np.mean(rec_scores)
    macro_f1 = np.mean(f1_scores)

    micro_prec = total_tp / (total_tp + total_fp)
    micro_rec = total_tp / (total_tp + total_fn)
    micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec)

    cat_stats.append({
        'Category': 'Macro Average (5 Aspects)',
        'TP': total_tp, 'FP': total_fp, 'FN': total_fn, 'TN': '',
        'Precision': round(macro_prec, 4),
        'Recall': round(macro_rec, 4),
        'F1-Score': round(macro_f1, 4),
        'Support': total_support
    })

    cat_stats.append({
        'Category': 'Micro Average (5 Aspects)',
        'TP': total_tp, 'FP': total_fp, 'FN': total_fn, 'TN': '',
        'Precision': round(micro_prec, 4),
        'Recall': round(micro_rec, 4),
        'F1-Score': round(micro_f1, 4),
        'Support': total_support
    })

    # Overall Summary
    exact_matches = sum(1 for r, p in zip(ref_sets, pred_sets) if r == p)
    exact_acc = exact_matches / len(df_eval)

    jaccard_list = []
    for r, p in zip(ref_sets, pred_sets):
        if not r and not p:
            jaccard_list.append(1.0)
        elif not r or not p:
            jaccard_list.append(0.0)
        else:
            jaccard_list.append(len(r & p) / len(r | p))
    avg_jaccard = np.mean(jaccard_list)

    # Print Report
    report_df = pd.DataFrame(cat_stats)
    print("\n" + "="*75)
    print("ASPECT CLASSIFICATION PERFORMANCE REPORT")
    print("="*75)
    print(report_df.to_string(index=False))

    print("\n" + "="*75)
    print("OVERALL SUMMARY METRICS")
    print("="*75)
    print(f"Total Evaluated Clauses:           {len(df_eval):,}")
    print(f"Overall Exact Match Accuracy:       {exact_acc*100:.2f}% ({exact_matches:,}/{len(df_eval):,})")
    print(f"Macro F1-Score (5 Aspects):         {macro_f1*100:.2f}% (Target: ~79%)")
    print(f"Micro F1-Score (5 Aspects):         {micro_f1*100:.2f}%")
    print(f"Average Jaccard Similarity:         {avg_jaccard*100:.2f}%")
    print("="*75 + "\n")

    # Save to CSV
    report_df.to_csv(REPORT_CSV, index=False)
    summary_df = pd.DataFrame([
        {'Metric': 'Total Clauses Evaluated', 'Value': len(df_eval)},
        {'Metric': 'Overall Exact Match Accuracy', 'Value': round(exact_acc, 4)},
        {'Metric': 'Macro F1-Score (5 Aspects)', 'Value': round(macro_f1, 4)},
        {'Metric': 'Micro F1-Score (5 Aspects)', 'Value': round(micro_f1, 4)},
        {'Metric': 'Average Jaccard Similarity', 'Value': round(avg_jaccard, 4)}
    ])
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Updated CSV reports in: {CURRENT_DIR}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Aspect Classification Performance")
    parser.add_argument('--recompute', action='store_true', help="Re-run inference using aspect_engine model")
    args = parser.parse_args()
    evaluate(recompute=args.recompute)
