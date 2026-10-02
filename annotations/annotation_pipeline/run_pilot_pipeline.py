import os
import sys
import json
import pandas as pd
from typing import List, Dict, Any

from segmenter import ClauseSegmenter
from annotator import AspectAnnotator
from validator import AnnotationValidator

PROJECT_ROOT = r"d:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews"
ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "annotations")
MANIFEST_PATH = os.path.join(ANNOTATIONS_DIR, "pilot_sample_manifest.csv")
OUTPUT_CSV = os.path.join(ANNOTATIONS_DIR, "pilot_llm_annotations.csv")
OUTPUT_JSONL = os.path.join(ANNOTATIONS_DIR, "pilot_llm_annotations.jsonl")
TEMPLATE_CSV = os.path.join(ANNOTATIONS_DIR, "verified_annotations_template.csv")
SUMMARY_MD = os.path.join(ANNOTATIONS_DIR, "pilot_annotation_summary.md")

def run_pilot():
    print(f"Loading manifest: {MANIFEST_PATH}")
    manifest_df = pd.read_csv(MANIFEST_PATH, keep_default_na=False)
    
    segmenter = ClauseSegmenter()
    annotator = AspectAnnotator(annotator_version="gemini-3.8-flash-preannotator-v1.0")
    validator = AnnotationValidator()

    all_assertions: List[Dict[str, Any]] = []
    review_clause_counts = {}

    for _, row in manifest_df.iterrows():
        rev_id = row['review_id']
        est_id = row['establishment_id']
        rev_text = row['review_text']
        timestamp = row['review_timestamp']
        orig_row = row['original_row_id']
        rating = row['star_rating']

        clauses = segmenter.segment(rev_text)
        review_clause_counts[rev_id] = len(clauses)

        for c_idx, (c_start, c_end, c_text) in enumerate(clauses, 1):
            clause_id = f"{rev_id}_C{c_idx:02d}"
            assertions = annotator.annotate_clause(
                review_id=rev_id,
                clause_id=clause_id,
                clause_text=c_text,
                clause_start_char=c_start,
                clause_end_char=c_end,
                review_text=rev_text
            )

            for ass in assertions:
                # Add metadata context
                record = {
                    "review_id": rev_id,
                    "original_row_id": orig_row,
                    "establishment_id": est_id,
                    "star_rating": rating,
                    "review_timestamp": timestamp,
                    "review_text": rev_text,
                    **ass
                }
                all_assertions.append(record)

    assertions_df = pd.DataFrame(all_assertions)
    print(f"\nExtracted {len(assertions_df)} aspect-opinion assertions from {len(manifest_df)} reviews.")

    # Run validation
    print("\nRunning rigorous validation suite...")
    is_valid, errors = validator.validate_dataset(assertions_df, manifest_df)
    if not is_valid:
        print(f"VALIDATION FAILED with {len(errors)} errors:")
        for err in errors[:10]:
            print(" -", err)
        raise ValueError("Annotation dataset failed validation checks.")
    else:
        print("ALL VALIDATION CHECKS PASSED SUCCESSFULLY!")

    # 1. Export JSONL (Machine-readable source)
    print(f"\nExporting JSONL to: {OUTPUT_JSONL}")
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for record in all_assertions:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 2. Export CSV (Human-review format)
    print(f"Exporting CSV to: {OUTPUT_CSV}")
    csv_cols = [
        "assertion_id", "clause_id", "review_id", "establishment_id", "star_rating",
        "clause_text", "aspect", "aspect_target_span", "opinion_span",
        "sentiment", "theme", "annotation_status",
        "clause_start_char", "clause_end_char", "target_start_char", "target_end_char",
        "opinion_start_char", "opinion_end_char", "llm_rationale",
        "annotator_version", "review_text"
    ]
    assertions_df[csv_cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    # 3. Export Human Verification Template
    print(f"Exporting Verification Template to: {TEMPLATE_CSV}")
    template_df = assertions_df[csv_cols].copy()
    template_df["verification_status"] = "Pending"
    template_df["verified_aspect"] = ""
    template_df["verified_target_span"] = ""
    template_df["verified_opinion_span"] = ""
    template_df["verified_sentiment"] = ""
    template_df["verified_theme"] = ""
    template_df["correction_notes"] = ""
    
    template_cols = [
        "assertion_id", "clause_id", "review_id", "establishment_id", "star_rating",
        "clause_text", "aspect", "aspect_target_span", "opinion_span",
        "sentiment", "theme", "annotation_status",
        "verification_status", "verified_aspect", "verified_target_span",
        "verified_opinion_span", "verified_sentiment", "verified_theme",
        "correction_notes", "llm_rationale", "clause_start_char", "clause_end_char",
        "review_text"
    ]
    template_df[template_cols].to_csv(TEMPLATE_CSV, index=False, encoding="utf-8")

    # 4. Generate Summary Report Markdown
    total_reviews = len(manifest_df)
    total_clauses = sum(review_clause_counts.values())
    total_assertions = len(assertions_df)
    status_counts = assertions_df["annotation_status"].value_counts().to_dict()
    aspect_counts = assertions_df[assertions_df["aspect"] != ""]["aspect"].value_counts().to_dict()
    sentiment_counts = assertions_df[assertions_df["sentiment"] != ""]["sentiment"].value_counts().to_dict()

    summary_content = f"""# Pilot Annotation Summary Report (100 Reviews)

**Execution Date:** 2026-09-30  
**Model & Pipeline:** `gemini-3.8-flash-preannotator-v1.0`  
**Dataset Scope:** 100-Review Pilot Sample (1 review per establishment across 100 establishments)  

---

## 1. Executive Metrics

| Metric | Count | Notes |
| :--- | :--- | :--- |
| **Total Reviews Processed** | {total_reviews} | Exactly 1 review per establishment |
| **Total Clauses Segmented** | {total_clauses} | Mean {total_clauses / total_reviews:.2f} clauses/review |
| **Total Assertions Generated** | {total_assertions} | Mean {total_assertions / total_reviews:.2f} assertions/review |
| **Validation Checks** | **100% Passed** | Zero character offset mismatches |

---

## 2. Assertion Status Breakdown

| Annotation Status | Count | Percentage |
| :--- | :--- | :--- |
| `Annotated` | {status_counts.get('Annotated', 0)} | {status_counts.get('Annotated', 0)/total_assertions*100:.1f}% |
| `No Aspect Opinion` | {status_counts.get('No Aspect Opinion', 0)} | {status_counts.get('No Aspect Opinion', 0)/total_assertions*100:.1f}% |
| `Needs Review` | {status_counts.get('Needs Review', 0)} | {status_counts.get('Needs Review', 0)/total_assertions*100:.1f}% |

---

## 3. Aspect Distribution (Excluding 'No Aspect Opinion')

| Aspect Category | Count | Percentage of Evaluative Assertions |
| :--- | :--- | :--- |
"""
    eval_total = sum(aspect_counts.values()) if aspect_counts else 1
    for asp, cnt in aspect_counts.items():
        summary_content += f"| `{asp}` | {cnt} | {cnt/eval_total*100:.1f}% |\n"

    summary_content += f"""
---

## 4. Sentiment Polarity Distribution

| Sentiment | Count | Percentage of Evaluative Assertions |
| :--- | :--- | :--- |
"""
    sent_total = sum(sentiment_counts.values()) if sentiment_counts else 1
    for sent, cnt in sentiment_counts.items():
        summary_content += f"| `{sent}` | {cnt} | {cnt/sent_total*100:.1f}% |\n"

    summary_content += """
---

## 5. Sample Ambiguous & Boundary Cases for Verification

1. **Activity / Capability Observation (`No Aspect Opinion`):**
   - *Clause:* `"One can also chill with friends and or parents"` (`REV_00001_C05`)
   - *Classification:* `No Aspect Opinion` (no evaluative polarity or aspect rating forced).
2. **Explicit Target in Pricing (`Annotated`):**
   - *Clause:* `"had Saturday lunch , which was cost effective"` (`REV_00001_C03`)
   - *Target Span:* `"Saturday lunch"`, *Opinion:* `"cost effective"`, *Aspect:* `Price / Value`.
3. **Contrastive Sentiment Split across Clauses:**
   - *Clauses:* `"The food was awesome"` (Positive) vs `"but staff was rude"` (Negative).
4. **Subtle Mixed Sentiments within Single Clause (`Needs Review`):**
   - Records marked `Needs Review` where positive and negative opinions co-occur without a clear clause delimiter.
"""
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write(summary_content)
    print(f"Generated summary report at: {SUMMARY_MD}")

if __name__ == "__main__":
    run_pilot()
