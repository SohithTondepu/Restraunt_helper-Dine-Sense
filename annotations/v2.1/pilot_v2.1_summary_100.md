# Full 100-Review Pilot V2.1 Annotation Summary Report

**Execution Date:** 2026-10-01  
**Annotation Engine:** `gemini-3.8-flash-preannotator-v2.1`  
**Scope:** Full 100-Review Pilot Dataset (Exactly 1 review per establishment across all 100 establishments)  
**Taxonomy:** 5-Aspect Scheme (`Food`, `Service`, `Price / Value`, `Ambience`, `General Experience`) with robust typo normalization and clause-level elaboration linking.

---

## 1. Executive Summary

| Metric | Count | Notes |
| :--- | :---: | :--- |
| **Total Reviews Processed** | 100 | 100% of pilot establishments represented |
| **Total Clauses Segmented** | 648 | Mean 6.48 clauses per review |
| **Total Assertions Generated** | 649 | Mean 6.49 assertions per review |
| **Character Offset Accuracy** | **100%** | Zero mismatches against unmodified source reviews |
| **Spelling Variants Recovered** | **100%** | Short reviews with typos (`avarage taste`, `Worst taste`, `very quick delivery..`) properly assigned |

---

## 2. Assertion Status Breakdown

| Status | Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| `Annotated` | 258 | 39.8% | Explicit, confident aspect–opinion evaluations |
| `No Aspect Opinion` | 391 | 60.2% | Factual context, procedural steps, dish lists, and idioms |

---

## 3. Aspect Distribution Across the 5 Categories

| # | Aspect Category | Count | Percentage of Evaluative Assertions | Description |
| :-: | :--- | :---: | :---: | :--- |
| 1 | `Food` | 118 | 45.7% | Culinary taste, freshness, dishes, portions, beverages |
| 2 | `General Experience` | 67 | 26.0% | Whole-venue recommendations, repeat intent, holistic appraisal |
| 3 | `Ambience` | 26 | 10.1% | Interior decor, atmosphere, comfort, music, cleanliness |
| 4 | `Service` | 30 | 11.6% | Staff courtesy, turnaround speed, delivery, hospitality |
| 5 | `Price / Value` | 17 | 6.6% | Affordability, bill charges, value for money |

---

## 4. Sentiment Polarity Breakdown (Evaluative Assertions)

| Sentiment | Count | Percentage |
| :--- | :---: | :---: |
| `Positive` | 192 | 74.4% |
| `Negative` | 61 | 23.6% |
| `Neutral` | 5 | 1.9% |

---

## 5. Deliverable Inventory in `annotations/v2.1/`

1. `pilot_v2.1_annotations_100.csv` — Primary CSV with 5-aspect taxonomy, exact offsets, and rationales.
2. `pilot_v2.1_annotations_100.jsonl` — Line-delimited JSON format for automated pipelines.
3. `verified_annotations_template_100.csv` — Template structured with blank verification columns ready for human audit.
4. `annotation_guidelines_v2.1.md` — Formal reference guidelines and edge case definitions.
