import os
import sys
import time
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.aspect_engine import match_aspect_hybrid, get_sentence_model

INPUT_CSV = os.path.join(PROJECT_ROOT, 'data', 'processed', 'clause_evaluation_input_2000.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'aspect_evaluation')
CSV_DIR = os.path.join(OUTPUT_DIR, 'csv')
PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, 'aspect_predictions_2000.csv')

ASPECT_CATEGORIES = ['Food', 'Service', 'Price / Value', 'Ambience', 'General Experience']


def run_evaluation():
    print(f"Loading evaluation dataset from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Total clauses to evaluate: {len(df)}")

    # Warmup sentence model
    print("Warming up MiniLM sentence model...")
    get_sentence_model()

    predicted_aspects_list = []
    predicted_aspects_str_list = []
    exact_match_list = []
    is_subset_list = []
    is_superset_list = []
    jaccard_list = []
    has_overlap_list = []

    print("Running aspect engine predictions across 11,631 clauses...")
    start_time = time.time()

    for idx, row in df.iterrows():
        text = str(row['clause_text'])
        ref_aspects = set(json.loads(row['reference_aspect_labels']))

        pred_aspects = match_aspect_hybrid(text)
        sorted_preds = sorted(list(pred_aspects))

        # Metrics comparison
        exact_match = (ref_aspects == pred_aspects)
        is_subset = pred_aspects.issubset(ref_aspects) and (len(pred_aspects) > 0)
        is_superset = pred_aspects.issuperset(ref_aspects) and (len(ref_aspects) > 0)
        has_overlap = bool(ref_aspects.intersection(pred_aspects))

        if not ref_aspects and not pred_aspects:
            jaccard = 1.0
        elif not ref_aspects or not pred_aspects:
            jaccard = 0.0
        else:
            jaccard = len(ref_aspects.intersection(pred_aspects)) / len(ref_aspects.union(pred_aspects))

        predicted_aspects_list.append(json.dumps(sorted_preds))
        predicted_aspects_str_list.append('; '.join(sorted_preds) if sorted_preds else 'No Aspect Opinion')
        exact_match_list.append(exact_match)
        is_subset_list.append(is_subset)
        is_superset_list.append(is_superset)
        jaccard_list.append(jaccard)
        has_overlap_list.append(has_overlap)

        if (idx + 1) % 2000 == 0 or (idx + 1) == len(df):
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            print(f"  Processed {idx + 1}/{len(df)} clauses ({rate:.1f} clauses/sec)...")

    total_time = time.time() - start_time
    print(f"Prediction complete in {total_time:.2f}s ({len(df)/total_time:.1f} clauses/sec).")

    # Save predictions dataframe
    pred_df = df.copy()
    pred_df['predicted_aspect_labels'] = predicted_aspects_list
    pred_df['predicted_aspects_display'] = predicted_aspects_str_list
    pred_df['exact_match'] = exact_match_list
    pred_df['jaccard_similarity'] = [round(j, 4) for j in jaccard_list]
    try:
        pred_df.to_csv(PREDICTIONS_CSV, index=False, encoding='utf-8')
        print(f"Saved predictions to: {PREDICTIONS_CSV}")
    except PermissionError:
        alt_pred = os.path.join(OUTPUT_DIR, 'aspect_predictions_2000_calibrated.csv')
        pred_df.to_csv(alt_pred, index=False, encoding='utf-8')
        print(f"Warning: {PREDICTIONS_CSV} is open in Excel or another program. Saved predictions to: {alt_pred}")

    # ==========================================
    # 1. CATEGORY-LEVEL EVALUATION METRICS
    # ==========================================
    category_stats = {}
    for cat in ASPECT_CATEGORIES:
        tp = fp = fn = tn = 0
        for idx, row in df.iterrows():
            ref_aspects = set(json.loads(row['reference_aspect_labels']))
            pred_aspects = set(json.loads(predicted_aspects_list[idx]))
            
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

    # No Aspect Opinion category stats
    tp_no = fp_no = fn_no = tn_no = 0
    for idx, row in df.iterrows():
        ref_aspects = set(json.loads(row['reference_aspect_labels']))
        pred_aspects = set(json.loads(predicted_aspects_list[idx]))

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

    # Calculate Macro, Micro, and Weighted Averages across 5 Aspect Categories
    aspect_precisions = [category_stats[c]['Precision'] for c in ASPECT_CATEGORIES]
    aspect_recalls = [category_stats[c]['Recall'] for c in ASPECT_CATEGORIES]
    aspect_f1s = [category_stats[c]['F1-Score'] for c in ASPECT_CATEGORIES]
    aspect_supports = [category_stats[c]['Support'] for c in ASPECT_CATEGORIES]
    total_aspect_support = sum(aspect_supports)

    macro_prec = np.mean(aspect_precisions)
    macro_rec = np.mean(aspect_recalls)
    macro_f1 = np.mean(aspect_f1s)

    total_tp = sum(category_stats[c]['TP'] for c in ASPECT_CATEGORIES)
    total_fp = sum(category_stats[c]['FP'] for c in ASPECT_CATEGORIES)
    total_fn = sum(category_stats[c]['FN'] for c in ASPECT_CATEGORIES)
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    weighted_prec = sum(category_stats[c]['Precision'] * category_stats[c]['Support'] for c in ASPECT_CATEGORIES) / total_aspect_support
    weighted_rec = sum(category_stats[c]['Recall'] * category_stats[c]['Support'] for c in ASPECT_CATEGORIES) / total_aspect_support
    weighted_f1 = sum(category_stats[c]['F1-Score'] * category_stats[c]['Support'] for c in ASPECT_CATEGORIES) / total_aspect_support

    # Build Classification Report DataFrame
    report_rows = []
    for cat in ASPECT_CATEGORIES + ['No Aspect Opinion']:
        st = category_stats[cat]
        report_rows.append({
            'Category': cat,
            'TP': st['TP'],
            'FP': st['FP'],
            'FN': st['FN'],
            'TN': st['TN'],
            'Precision': st['Precision'],
            'Recall': st['Recall'],
            'F1-Score': st['F1-Score'],
            'Support': st['Support']
        })

    report_rows.append({
        'Category': 'Macro Average (5 Aspects)',
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'TN': '',
        'Precision': round(macro_prec, 4),
        'Recall': round(macro_rec, 4),
        'F1-Score': round(macro_f1, 4),
        'Support': total_aspect_support
    })
    report_rows.append({
        'Category': 'Micro Average (5 Aspects)',
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'TN': '',
        'Precision': round(micro_prec, 4),
        'Recall': round(micro_rec, 4),
        'F1-Score': round(micro_f1, 4),
        'Support': total_aspect_support
    })
    report_rows.append({
        'Category': 'Weighted Average (5 Aspects)',
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'TN': '',
        'Precision': round(weighted_prec, 4),
        'Recall': round(weighted_rec, 4),
        'F1-Score': round(weighted_f1, 4),
        'Support': total_aspect_support
    })

    report_df = pd.DataFrame(report_rows)
    report_csv = os.path.join(CSV_DIR, 'aspect_classification_report.csv')
    report_df.to_csv(report_csv, index=False, encoding='utf-8')
    print(f"Saved classification report to: {report_csv}")

    # ==========================================
    # 2. OVERALL SUMMARY METRICS
    # ==========================================
    exact_match_ratio = np.mean(exact_match_list)
    avg_jaccard = np.mean(jaccard_list)

    # Evaluative subset metrics (clauses that have >= 1 reference aspect)
    eval_mask = df['annotation_status'] == 'Annotated'
    eval_exact_match = np.mean([exact_match_list[i] for i in range(len(df)) if eval_mask.iloc[i]])
    eval_overlap = np.mean([has_overlap_list[i] for i in range(len(df)) if eval_mask.iloc[i]])

    # Negative subset metrics (No Aspect Opinion clauses)
    neg_mask = df['annotation_status'] == 'No Aspect Opinion'
    neg_exact_match = np.mean([exact_match_list[i] for i in range(len(df)) if neg_mask.iloc[i]])

    # Hamming Loss across 5 binary aspect labels
    total_label_decisions = len(df) * len(ASPECT_CATEGORIES)
    incorrect_decisions = sum(category_stats[c]['FP'] + category_stats[c]['FN'] for c in ASPECT_CATEGORIES)
    hamming_loss = incorrect_decisions / total_label_decisions

    summary_rows = [
        {'Metric': 'Total Clauses Evaluated', 'Value': len(df)},
        {'Metric': 'Total Evaluative Clauses (Annotated)', 'Value': int(eval_mask.sum())},
        {'Metric': 'Total Non-Evaluative Clauses (No Aspect Opinion)', 'Value': int(neg_mask.sum())},
        {'Metric': 'Overall Exact Match Ratio (All Clauses)', 'Value': round(float(exact_match_ratio), 4)},
        {'Metric': 'Evaluative Exact Match Ratio (Annotated Only)', 'Value': round(float(eval_exact_match), 4)},
        {'Metric': 'Evaluative At-Least-One Match Recall', 'Value': round(float(eval_overlap), 4)},
        {'Metric': 'Non-Evaluative Precision (No Aspect Opinion Specificity)', 'Value': round(float(neg_exact_match), 4)},
        {'Metric': 'Macro F1-Score (5 Aspects)', 'Value': round(float(macro_f1), 4)},
        {'Metric': 'Macro Precision (5 Aspects)', 'Value': round(float(macro_prec), 4)},
        {'Metric': 'Macro Recall (5 Aspects)', 'Value': round(float(macro_rec), 4)},
        {'Metric': 'Micro F1-Score (5 Aspects)', 'Value': round(float(micro_f1), 4)},
        {'Metric': 'Weighted F1-Score (5 Aspects)', 'Value': round(float(weighted_f1), 4)},
        {'Metric': 'Average Jaccard Similarity', 'Value': round(float(avg_jaccard), 4)},
        {'Metric': 'Hamming Loss', 'Value': round(float(hamming_loss), 4)},
        {'Metric': 'Total Execution Time (seconds)', 'Value': round(float(total_time), 2)},
        {'Metric': 'Inference Throughput (clauses/sec)', 'Value': round(float(len(df)/total_time), 1)}
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(CSV_DIR, 'aspect_overall_summary.csv')
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8')
    print(f"Saved overall summary metrics to: {summary_csv}")

    # ==========================================
    # 3. CONFUSION / CO-OCCURRENCE MATRIX
    # ==========================================
    # Rows: Reference Category | Columns: Predicted Category
    all_labels = ASPECT_CATEGORIES + ['No Aspect Opinion']
    cooccur_matrix = pd.DataFrame(0, index=all_labels, columns=all_labels)

    for idx, row in df.iterrows():
        ref_aspects = set(json.loads(row['reference_aspect_labels']))
        pred_aspects = set(json.loads(predicted_aspects_list[idx]))

        ref_keys = list(ref_aspects) if ref_aspects else ['No Aspect Opinion']
        pred_keys = list(pred_aspects) if pred_aspects else ['No Aspect Opinion']

        for rk in ref_keys:
            for pk in pred_keys:
                cooccur_matrix.loc[rk, pk] += 1

    confusion_csv = os.path.join(CSV_DIR, 'aspect_confusion_matrix.csv')
    cooccur_matrix.to_csv(confusion_csv, encoding='utf-8')
    print(f"Saved confusion/co-occurrence matrix to: {confusion_csv}")

    # ==========================================
    # 4. ERROR BREAKDOWN
    # ==========================================
    error_rows = []
    for cat in ASPECT_CATEGORIES:
        fp_examples = []
        fn_examples = []
        for idx, row in df.iterrows():
            ref_aspects = set(json.loads(row['reference_aspect_labels']))
            pred_aspects = set(json.loads(predicted_aspects_list[idx]))
            c_text = row['clause_text']

            # False Positive: predicted cat, but not in ref
            if cat in pred_aspects and cat not in ref_aspects:
                if len(fp_examples) < 5:
                    ref_str = '; '.join(sorted(list(ref_aspects))) if ref_aspects else 'No Aspect Opinion'
                    fp_examples.append(f'"{c_text}" (Ref: {ref_str})')

            # False Negative: in ref, but not predicted
            if cat in ref_aspects and cat not in pred_aspects:
                if len(fn_examples) < 5:
                    pred_str = '; '.join(sorted(list(pred_aspects))) if pred_aspects else 'No Aspect Opinion'
                    fn_examples.append(f'"{c_text}" (Pred: {pred_str})')

        error_rows.append({
            'Category': cat,
            'Total_FP': category_stats[cat]['FP'],
            'Total_FN': category_stats[cat]['FN'],
            'Sample_FP_Clauses': ' | '.join(fp_examples),
            'Sample_FN_Clauses': ' | '.join(fn_examples)
        })

    error_df = pd.DataFrame(error_rows)
    error_csv = os.path.join(CSV_DIR, 'aspect_error_breakdown.csv')
    error_df.to_csv(error_csv, index=False, encoding='utf-8')
    print(f"Saved error breakdown to: {error_csv}")

    print("\n=== EVALUATION RUN COMPLETE ===")
    print(report_df.to_string())
    print("\nSummary:")
    print(summary_df.to_string())


if __name__ == '__main__':
    run_evaluation()
