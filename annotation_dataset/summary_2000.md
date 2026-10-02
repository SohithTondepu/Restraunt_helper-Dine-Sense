# DineSense AI: 2,000-Review Milestone Dataset Summary Report
**Total Reviews:** 2,000 (100 Establishments × 20 Reviews/Establishment)  
**Composition:** 100 Approved Pilot Reviews (`annotations/v2.2/pilot_v2.2_annotations_100.csv`) + 1,900 Batch Reviews (`annotations/batch_1900/annotations_1900.csv`)  
**Status:** Completed & 100% Offset-Verified  
**Files:**
- Final Merged CSV: [`annotations/combined_2000/annotations_2000_final.csv`](file:///d:/Aspect-Based-Sentimental-Analysis-on-Food-Reviews/annotations/combined_2000/annotations_2000_final.csv)
- Merged JSONL: [`annotations/combined_2000/annotations_2000.jsonl`](file:///d:/Aspect-Based-Sentimental-Analysis-on-Food-Reviews/annotations/combined_2000/annotations_2000.jsonl)

---

## 1. High-Level Dataset Statistics

| Metric | Approved Pilot (100) | Batch (1,900) | Combined Milestone (2,000) |
| :--- | :---: | :---: | :---: |
| **Total Reviews** | 100 | 1,900 | **2,000** |
| **Total Clauses** | 723 | 10,908 | **11,631** |
| **Total Assertions** | 727 | 10,908 | **11,635** |
| **Annotated Assertions** | 388 (53.4%) | 4,570 (41.9%) | **4,958 (42.6%)** |
| **No Aspect Opinion** | 339 (46.6%) | 6,338 (58.1%) | **6,677 (57.4%)** |
| **Needs Review** | 0 (0.0%) | 0 (0.0%) | **0 (0.0%)** |

---

## 2. Aspect Category Breakdown (Annotated: 4,958 assertions)

| Aspect Category | Assertions | Share of Annotated |
| :--- | :---: | :---: |
| **Food** | 2,101 | 42.4% |
| **General Experience** | 1,358 | 27.4% |
| **Service** | 683 | 13.8% |
| **Ambience** | 533 | 10.8% |
| **Price / Value** | 283 | 5.7% |
| **Total** | **4,958** | **100.0%** |

---

## 3. Sentiment Breakdown (Annotated: 4,958 assertions)

| Sentiment | Assertions | Share of Annotated |
| :--- | :---: | :---: |
| **Positive** | 3,926 | 79.2% |
| **Negative** | 930 | 18.8% |
| **Neutral** | 101 | 2.0% |
| **Mixed** | 1 | 0.02% |
| **Total** | **4,958** | **100.0%** |

---

## 4. Quality & Compliance Checklist

- [x] **No Overwriting of Pilot:** The approved 100-review pilot files in `annotations/v2.2/` remain completely unchanged.
- [x] **100% Verbatim Offset Matching:** Verified `review_text[start:end] == span_text` programmatically for 100% of clause, target, and opinion spans (0 errors across 7,810 spans).
- [x] **Zero Ambiguity Left Behind:** All 60 previously ambiguous clauses were evaluated in full review context and assigned to their respective aspect categories.
- [x] **No Model Training / Train-Test Splits:** Strictly complied with instructions to refrain from training or splitting data at this stage.
