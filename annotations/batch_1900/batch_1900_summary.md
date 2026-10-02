# DineSense AI: Batch 1,900 & Combined 2,000 Annotation Summary Report
**Dataset:** 1,900 Stratified Restaurant Reviews (`annotations/batch_1900/sample_manifest_1900.csv`) + 100 Approved Pilot Reviews  
**Pipeline Engine:** `Batch1900Annotator` (`annotations/batch_1900/batch_1900_annotator_engine.py`)  
**Resolution Engine:** `validate_resolutions.py` / `apply_60_resolutions.py`  
**Execution Date:** 2026-10-01  
**Deliverables:**
- Batch 1,900 CSV: [`annotations/batch_1900/annotations_1900.csv`](file:///d:/Aspect-Based-Sentimental-Analysis-on-Food-Reviews/annotations/batch_1900/annotations_1900.csv)
- Batch 1,900 JSONL: [`annotations/batch_1900/annotations_1900.jsonl`](file:///d:/Aspect-Based-Sentimental-Analysis-on-Food-Reviews/annotations/batch_1900/annotations_1900.jsonl)
- Batch 1,900 Verification Template: [`annotations/batch_1900/verified_annotations_template_1900.csv`](file:///d:/Aspect-Based-Sentimental-Analysis-on-Food-Reviews/annotations/batch_1900/verified_annotations_template_1900.csv)
- Combined 2,000 Final CSV: [`annotations/combined_2000/annotations_2000_final.csv`](file:///d:/Aspect-Based-Sentimental-Analysis-on-Food-Reviews/annotations/combined_2000/annotations_2000_final.csv)
- Combined 2,000 JSONL: [`annotations/combined_2000/annotations_2000.jsonl`](file:///d:/Aspect-Based-Sentimental-Analysis-on-Food-Reviews/annotations/combined_2000/annotations_2000.jsonl)

---

## 1. Overview & Sampling Integrity

Following approval of the 100-review pilot annotation, the next batch of **1,900 reviews** was sampled and annotated:
- **Zero Pilot Overlap:** Filtered out all 100 pilot row indices (`pilot_sample_manifest.csv`) to ensure 0% duplicate reviews.
- **Establishment Balance:** Sampled exactly **19 reviews per establishment** across all **100 food establishments** in the dataset (combined with the 1 pilot review/establishment = 20 reviews per establishment across 2,000 reviews).
- **Representative Stratification:**
  - **Star Rating:** High (4.0–5.0): 1,162 (61.2%), Low (1.0–2.5): 502 (26.4%), Mid (3.0–3.5): 236 (12.4%) — strictly preserving the ground-truth population distribution of `data/processed/cleaned_reviews.csv`.
  - **Review Length:** Medium (150–350 chars): 947 (49.8%), Short (<150 chars): 513 (27.0%), Long (>350 chars): 440 (23.2%).
  - **Temporal Range:** 2016 (9), 2017 (36), 2018 (840), 2019 (1,015).

---

## 2. Quantitative Summary (Batch 1,900 After Ambiguity Resolution)

All 60 previously ambiguous `Needs Review` cases have been definitively resolved into their proper aspects using expert LLM contextual discourse analysis:

| Metric | Count | Percentage |
| :--- | :---: | :---: |
| **Total Reviews** | 1,900 | 100.0% |
| **Total Clauses** | 10,908 | — |
| **Total Assertions** | 10,908 | — |
| **Annotated Assertions** | 4,570 | 41.9% |
| **No Aspect Opinion (Factual/Context)** | 6,338 | 58.1% |
| **Needs Review (Ambiguous Cases)** | **0** | **0.0%** |

---

## 3. Aspect & Sentiment Distributions (Batch 1,900)

### Aspect Category Breakdown (Annotated: 4,570 assertions)
| Aspect Category | Assertions | Share of Annotated |
| :--- | :---: | :---: |
| **Food** | 1,924 | 42.1% |
| **General Experience** | 1,281 | 28.0% |
| **Service** | 631 | 13.8% |
| **Ambience** | 479 | 10.5% |
| **Price / Value** | 255 | 5.6% |

### Sentiment Breakdown (Annotated: 4,570 assertions)
| Sentiment | Assertions | Share of Annotated |
| :--- | :---: | :---: |
| **Positive** | 3,653 | 79.9% |
| **Negative** | 824 | 18.0% |
| **Neutral** | 92 | 2.0% |
| **Mixed** | 1 | 0.02% |

---

## 4. Resolution of the 60 Ambiguous Cases

Using contextual review understanding, discourse coreference, and the approved guidelines:
- **Food (26 cases resolved):** Disambiguated by linking pronoun references or truncated opinions to specific dishes or menu offerings mentioned in adjacent clauses (e.g., `'Nothing special in it'` -> starter flavor; `'this can be better'` -> Exotic Vegetable Pasta; `'disappointed with the temperature of the coffees'` -> food temperature).
- **General Experience (27 cases resolved):** Captures whole-visit appraisals, return intent statements, or overall emotional summaries not confined to any individual department (e.g., `'Never disappointed us in any aspect'`, `'HIGHLY DISAPPOINTED'`, `'will not prefer to return again'`, `'total disappointed yesterday night'`).
- **Service (6 cases resolved):** Contextual service issues including delivery packing failures, missing items, uncoordinated birthday arrangements, and door entry refusals (e.g., `'only disappointed with the packaging'`, `'Extremely disappointed'` due to stag entry refusal).
- **Ambience (1 case resolved):** Atmosphere/music improvement feedback (`'can be better'` following discussion of loud music).

---

## 5. Combined 2,000-Review Canonical Dataset (Pilot 100 + Batch 1,900)

The approved 100-review pilot and the updated 1,900-review batch form the complete 2,000-review milestone dataset:
- File: [`annotations/combined_2000/annotations_2000_final.csv`](file:///d:/Aspect-Based-Sentimental-Analysis-on-Food-Reviews/annotations/combined_2000/annotations_2000_final.csv)
- JSONL: [`annotations/combined_2000/annotations_2000.jsonl`](file:///d:/Aspect-Based-Sentimental-Analysis-on-Food-Reviews/annotations/combined_2000/annotations_2000.jsonl)

### Overall 2,000-Review Statistics:
- **Total Reviews:** 2,000
- **Total Atomic Clauses:** 11,631
- **Total Assertions:** 11,635
- **Status Counts:**
  - `Annotated`: 4,958 (42.6%)
  - `No Aspect Opinion`: 6,677 (57.4%)
  - `Needs Review`: **0 (0.0%)**
- **Aspect Breakdown (Annotated: 4,958):**
  - **Food:** 2,101 assertions (42.4%)
  - **General Experience:** 1,358 assertions (27.4%)
  - **Service:** 683 assertions (13.8%)
  - **Ambience:** 533 assertions (10.8%)
  - **Price / Value:** 283 assertions (5.7%)
- **Sentiment Breakdown (Annotated: 4,958):**
  - **Positive:** 3,926 assertions (79.2%)
  - **Negative:** 930 assertions (18.8%)
  - **Neutral:** 101 assertions (2.0%)
  - **Mixed:** 1 assertion (0.02%)

---

## 6. Verification & Quality Assurance

1. **Character Offset Traceability:**
   - Programmatically validated 100% of all 11,635 assertions:
     - `review_text[clause_start:clause_end] == clause_text` (0 errors across 11,631 clauses)
     - `review_text[target_start:target_end] == aspect_target_span` (0 errors across all non-empty targets)
     - `review_text[opinion_start:opinion_end] == opinion_span` (0 errors across all non-empty opinions)
2. **Preservation of Raw Inputs:**
   - `data/processed/cleaned_reviews.csv` remains strictly untouched.
   - Punctuation, capitalization, emojis, typos, and original CRLF (`\r\n`) endings are 100% preserved.
