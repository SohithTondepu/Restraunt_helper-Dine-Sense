import os
import sys
import json
import numpy as np
import pandas as pd
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(PROJECT_ROOT, 'aspect_evaluation', 'csv')
PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, 'aspect_evaluation', 'aspect_predictions_2000.csv')

ASPECT_CATEGORIES = ['Food', 'Service', 'Price / Value', 'Ambience', 'General Experience']

def audit_and_correct():
    print(f"Loading predictions from: {PREDICTIONS_CSV}")
    df = pd.read_csv(PREDICTIONS_CSV)
    N = len(df)
    assert N == 11631, f"Expected 11631 rows, got {N}"

    # 1. Binary One-vs-Rest Evaluation across 5 Aspect Categories
    category_stats = {}
    for cat in ASPECT_CATEGORIES:
        tp = fp = fn = tn = 0
        for _, row in df.iterrows():
            ref = set(json.loads(row['reference_aspect_labels']))
            pred = set(json.loads(row['predicted_aspect_labels']))
            in_ref = cat in ref
            in_pred = cat in pred
            if in_ref and in_pred:
                tp += 1
            elif not in_ref and in_pred:
                fp += 1
            elif in_ref and not in_pred:
                fn += 1
            else:
                tn += 1

        assert tp + fn + fp + tn == N, "Contingency table does not sum to N"
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        support = tp + fn
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        category_stats[cat] = {
            'Category': cat,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'Specificity': round(specificity, 4),
            'Support': support,
            'Total_Evaluated': N
        }

    # Macro, Micro, Weighted across 5 Aspects ONLY
    supports = [category_stats[c]['Support'] for c in ASPECT_CATEGORIES]
    total_aspect_support = sum(supports)

    macro_prec = np.mean([category_stats[c]['Precision'] for c in ASPECT_CATEGORIES])
    macro_rec = np.mean([category_stats[c]['Recall'] for c in ASPECT_CATEGORIES])
    macro_f1 = np.mean([category_stats[c]['F1-Score'] for c in ASPECT_CATEGORIES])

    total_tp = sum(category_stats[c]['TP'] for c in ASPECT_CATEGORIES)
    total_fp = sum(category_stats[c]['FP'] for c in ASPECT_CATEGORIES)
    total_fn = sum(category_stats[c]['FN'] for c in ASPECT_CATEGORIES)
    micro_prec = total_tp / (total_tp + total_fp)
    micro_rec = total_tp / (total_tp + total_fn)
    micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec)

    weighted_prec = sum(category_stats[c]['Precision'] * category_stats[c]['Support'] for c in ASPECT_CATEGORIES) / total_aspect_support
    weighted_rec = sum(category_stats[c]['Recall'] * category_stats[c]['Support'] for c in ASPECT_CATEGORIES) / total_aspect_support
    weighted_f1 = sum(category_stats[c]['F1-Score'] * category_stats[c]['Support'] for c in ASPECT_CATEGORIES) / total_aspect_support

    # Save 5-Aspect Classification Report (Excluding No Aspect Opinion from the 5-aspect average)
    report_rows = [category_stats[c] for c in ASPECT_CATEGORIES]
    report_rows.append({
        'Category': 'Macro Average (5 Aspects)',
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'TN': '',
        'Precision': round(macro_prec, 4),
        'Recall': round(macro_rec, 4),
        'F1-Score': round(macro_f1, 4),
        'Specificity': '',
        'Support': total_aspect_support,
        'Total_Evaluated': N
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
        'Specificity': '',
        'Support': total_aspect_support,
        'Total_Evaluated': N
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
        'Specificity': '',
        'Support': total_aspect_support,
        'Total_Evaluated': N
    })

    report_df = pd.DataFrame(report_rows)
    report_csv = os.path.join(CSV_DIR, 'aspect_classification_report.csv')
    report_df.to_csv(report_csv, index=False, encoding='utf-8')
    print(f"Saved corrected classification report to: {report_csv}")

    # 2. Binary One-vs-Rest Contingency Tables CSV
    contingency_rows = []
    for c in ASPECT_CATEGORIES:
        st = category_stats[c]
        contingency_rows.append({
            'Aspect': c,
            'TP (Present in Ref & Pred)': st['TP'],
            'FN (Present in Ref, Absent in Pred)': st['FN'],
            'FP (Absent in Ref, Present in Pred)': st['FP'],
            'TN (Absent in Ref & Pred)': st['TN'],
            'Support (TP + FN)': st['Support'],
            'Negative_Support (FP + TN)': st['FP'] + st['TN'],
            'Total_Clauses (N)': st['Total_Evaluated'],
            'Precision': st['Precision'],
            'Recall': st['Recall'],
            'F1-Score': st['F1-Score'],
            'Specificity': st['Specificity']
        })
    contingency_df = pd.DataFrame(contingency_rows)
    contingency_csv = os.path.join(CSV_DIR, 'aspect_binary_contingency_tables.csv')
    contingency_df.to_csv(contingency_csv, index=False, encoding='utf-8')
    print(f"Saved binary contingency tables to: {contingency_csv}")

    # 3. No Aspect Opinion Correct Handling
    ref_no_aspect = df['reference_aspect_labels'].apply(lambda x: len(json.loads(x)) == 0)
    pred_no_aspect = df['predicted_aspect_labels'].apply(lambda x: len(json.loads(x)) == 0)

    n_ref_no_aspect = int(ref_no_aspect.sum())
    n_pred_no_aspect = int(pred_no_aspect.sum())
    correct_no_aspect = int((ref_no_aspect & pred_no_aspect).sum())
    false_no_aspect = int((~ref_no_aspect & pred_no_aspect).sum())
    aspects_on_ref_no_aspect = int((ref_no_aspect & ~pred_no_aspect).sum())

    # Breakdown of false alarms on No Aspect Opinion
    false_alarm_breakdown = Counter()
    for _, r in df[ref_no_aspect & ~pred_no_aspect].iterrows():
        for a in json.loads(r['predicted_aspect_labels']):
            false_alarm_breakdown[a] += 1

    # Breakdown of missed aspects in false no aspect predictions
    missed_aspect_breakdown = Counter()
    for _, r in df[~ref_no_aspect & pred_no_aspect].iterrows():
        for a in json.loads(r['reference_aspect_labels']):
            missed_aspect_breakdown[a] += 1

    no_aspect_summary = [
        {'Metric': '1. Reference clauses with no aspect (No Aspect Opinion)', 'Count': n_ref_no_aspect, 'Share_of_Total_Clauses': f"{n_ref_no_aspect/N:.2%}"},
        {'Metric': '2. Predicted clauses with no aspect (Empty Prediction)', 'Count': n_pred_no_aspect, 'Share_of_Total_Clauses': f"{n_pred_no_aspect/N:.2%}"},
        {'Metric': '3. Correct no-aspect detections (Ref=None & Pred=None)', 'Count': correct_no_aspect, 'Share_of_Total_Clauses': f"{correct_no_aspect/n_ref_no_aspect:.2%} of ref no-aspect"},
        {'Metric': '4. False no-aspect predictions (Ref has aspect, Pred=None)', 'Count': false_no_aspect, 'Share_of_Total_Clauses': f"{false_no_aspect/(N - n_ref_no_aspect):.2%} of annotated clauses"},
        {'Metric': '5. Aspect predictions made on ref No Aspect Opinion (False Alarms)', 'Count': aspects_on_ref_no_aspect, 'Share_of_Total_Clauses': f"{aspects_on_ref_no_aspect/n_ref_no_aspect:.2%} of ref no-aspect"}
    ]
    no_aspect_df = pd.DataFrame(no_aspect_summary)
    no_aspect_csv = os.path.join(CSV_DIR, 'aspect_no_aspect_audit.csv')
    no_aspect_df.to_csv(no_aspect_csv, index=False, encoding='utf-8')
    print(f"Saved No Aspect Opinion audit to: {no_aspect_csv}")

    # 4. Multi-Label Co-Occurrence / Confusion Matrix with Explicit Documentation
    # Cell (R_i, P_j) = Count of clauses having reference aspect R_i where model predicted P_j
    all_keys = ASPECT_CATEGORIES + ['No Aspect Opinion']
    cooccur_matrix = pd.DataFrame(0, index=all_keys, columns=all_keys)

    for _, row in df.iterrows():
        ref = set(json.loads(row['reference_aspect_labels']))
        pred = set(json.loads(row['predicted_aspect_labels']))
        ref_keys = list(ref) if ref else ['No Aspect Opinion']
        pred_keys = list(pred) if pred else ['No Aspect Opinion']
        for rk in ref_keys:
            for pk in pred_keys:
                cooccur_matrix.loc[rk, pk] += 1

    confusion_csv = os.path.join(CSV_DIR, 'aspect_confusion_matrix.csv')
    cooccur_matrix.to_csv(confusion_csv, encoding='utf-8')
    print(f"Saved documented co-occurrence matrix to: {confusion_csv}")

    print("\n=== AUDIT COMPLETE ===")
    print("\nClassification Report (5 Aspects):")
    print(report_df.to_string())
    print("\nNo Aspect Opinion Summary:")
    print(no_aspect_df.to_string())
    print(f"\nFalse Alarms Breakdown by Aspect: {dict(false_alarm_breakdown)}")
    print(f"Missed Aspects Breakdown: {dict(missed_aspect_breakdown)}")

if __name__ == '__main__':
    audit_and_correct()
