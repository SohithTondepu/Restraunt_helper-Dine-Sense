# Full 100-Review Pilot V2 Annotation Summary Report

**Execution Date:** 2026-10-01  
**Annotation Engine:** `gemini-3.8-flash-preannotator-v2.0`  
**Scope:** Full 100-Review Pilot Dataset (Exactly 1 review per establishment across all 100 establishments)  
**Methodology:** Revised semantic ABSA with compound negation spans, context disambiguation, grounded whole-establishment categorization, and descriptive clause elaboration linking.

---

## 1. Executive Summary

| Metric | Count | Notes |
| :--- | :---: | :--- |
| **Total Reviews Processed** | 100 | 100% of pilot establishments represented |
| **Total Clauses Segmented** | 647 | Mean 6.47 clauses per review |
| **Total Assertions Generated** | 648 | Mean 6.48 assertions per review |
| **Character Offset Accuracy** | **100%** | Zero mismatches against unmodified source reviews |
| **Hallucinated AC Theme** | **0** | Pruned to zero across the entire dataset |
| **Unspecified `Other` Dump** | **0** | Replaced with grounded categories and `General / Whole Establishment` |

---

## 2. Assertion Status Breakdown

| Status | Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| `Annotated` | 208 | 32.1% | Explicit, confident aspect–opinion evaluations |
| `No Aspect Opinion` | 440 | 67.9% | Factual context, procedural steps, dish lists, and idioms |
| `Needs Review` | 0 | 0.0% | Ambiguous polarity or complex edge cases flagged for human verification |

---

## 3. Aspect Distribution (Evaluative Assertions)

| Aspect Category | Count | Percentage of Evaluative Assertions | Notes |
| :--- | :---: | :---: | :--- |
| `Food / Dining` | 89 | 42.8% | Grounded textual evidence |
| `General / Whole Establishment` | 85 | 40.9% | Grounded textual evidence |
| `Facilities / Amenities` | 14 | 6.7% | Grounded textual evidence |
| `Staff / Service` | 12 | 5.8% | Grounded textual evidence |
| `Price / Value` | 6 | 2.9% | Grounded textual evidence |
| `Cleanliness` | 2 | 1.0% | Grounded textual evidence |

---

## 4. Sentiment Polarity Breakdown (Evaluative Assertions)

| Sentiment | Count | Percentage |
| :--- | :---: | :---: |
| `Positive` | 154 | 74.0% |
| `Negative` | 54 | 26.0% |

---

## 5. Deliverable Inventory in `annotations/v2/`

1. `pilot_v2_annotations_100.csv` — Human-readable CSV containing all assertions with zero-based offsets and rationales.
2. `pilot_v2_annotations_100.jsonl` — Machine-readable structured line-delimited JSON.
3. `verified_annotations_template_100.csv` — Verification template pre-formatted with columns for `verification_status` (`Pending`, `Verified`, `Corrected`, `Rejected`), `verified_aspect`, `verified_sentiment`, `correction_notes`, etc.
4. `revised_annotation_instructions.md` — Formal guidelines documenting negation, opinion span selection, and elaboration linking.
5. `unresolved_cases.md` — Catalog of edge cases flagged for human judgment.
