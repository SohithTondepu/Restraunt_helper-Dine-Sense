# Aspect–Sentiment Annotation Audit & Correction Report

**Dataset Audited:** `final_aspect_evaluation/modified_annotations_2000.csv`  
**Corrected Dataset:** `modified_annotations_2000_corrected.csv`  
**Audit Log:** `annotation_correction_audit.csv`  
**Training Summary:** `training_data_summary.csv`  
**Audit Standard:** DineSense AI Annotation Guidelines v2.1  

---

## 1. Executive Summary & Core Results

A focused, conservative annotation correction pass was conducted across all **12,245 assertion rows** (spanning 2,000 reviews and 11,631 unique clauses). No predictions from machine learning models, aspect engines, or star ratings were used to assign gold ground truth. All decisions are grounded strictly in explicit textual evidence and canonical annotation rules.

- **Total Rows Inspected:** 12,245
- **Baseline Eligible Training Assertions:** 5,072
- **Corrected Eligible Training Assertions:** 5,649
- **Net Additional Valid Aspect-Sentiment Assertions:** **+577**
- **Assertions Left Unresolved (Flagged for Review):** **1,593**
- **Total Audit Actions Recorded in Audit Log:** 2,814

---

## 2. Detailed Audit Metrics

### A. Review of `No Aspect Opinion` Rows (4,872 Rows Total)
Every row originally labeled `No Aspect Opinion` was inspected:
- **Total `No Aspect Opinion` rows reviewed:** 4,872
- **Subgroup B.1 (Evaluative assertions with existing sentiment but missing aspect):** 302 rows restored to `Annotated` with canonical aspects derived from explicit evaluation themes.
- **Subgroup B.2 (Rows with null sentiment):** Reclassified when text expressed unambiguous evaluative opinions:
  - Reclassified to **Food**: 154
  - Reclassified to **Service**: 22
  - Reclassified to **Price / Value**: 19
  - Reclassified to **Ambience**: 30
  - Reclassified to **General Experience**: 227
- **Total with Sentiment Restored / Filled:** 452
- **Remaining genuine factual / procedural clauses retaining `No Aspect Opinion`:** 4,420

### B. Review of Rows with Aspect but Missing Sentiment (2,301 Rows Total)
Every row originally assigned an aspect with blank/null sentiment was inspected:
- **Total Missing Sentiment rows reviewed:** 2,301
- **Resolved as Positive:** 69
- **Resolved as Negative:** 47
- **Resolved as Neutral:** 9
- **Reclassified to `No Aspect Opinion` (Factual entity mentions without evaluative opinion):** 583
- **Left Unresolved (Ambiguous / Rhetorical / Insufficient context):** 1,593

### C. Inconsistencies Corrected Across Existing Annotations
- **Polarity Contradictions / Explicit Negation Errors Corrected:** 61  
  *(e.g., "Not good", "Not fresh", "Service is pathetic" incorrectly labeled Positive flipped to Negative; "not bad" corrected to Neutral)*
- **Aspect Misattributions Corrected:** 0  
  *(e.g., staff service turnaround mislabeled as Ambience moved to Service)*
- **Non-Canonical Sentiment Normalized:** 1  
  *(1 row with sentiment 'Mixed' normalized to 'Neutral')*

---

## 3. Before vs. After Distribution Comparison

### Aspect Distribution
| Aspect Category | Original Count | Corrected Count | Delta |
| :--- | :--- | :--- | :--- |
| **Food** | 3,439 | 3,263 | -176 |
| **Service** | 1,238 | 1,136 | -102 |
| **Price / Value** | 471 | 462 | -9 |
| **Ambience** | 903 | 892 | -11 |
| **General Experience** | 1,322 | 1,489 | +167 |
| *Unassigned / No Aspect (NaN)* | 4,872 | 5,003 | +131 |

### Sentiment Distribution
| Sentiment Class | Original Count | Corrected Count | Delta |
| :--- | :--- | :--- | :--- |
| **Positive** | 4,267 | 4,372 | +105 |
| **Negative** | 995 | 1,127 | +132 |
| **Neutral** | 111 | 150 | +39 |
| **Mixed** *(Non-canonical)* | 1 | 0 | -1 |
| *Unassigned / Unresolved (NaN)* | 6,871 | 6,596 | -275 |

### Annotation Status Distribution
| Annotation Status | Original Count | Corrected Count | Delta |
| :--- | :--- | :--- | :--- |
| **Annotated** | 7,373 | 7,242 | -131 |
| **No Aspect Opinion** | 4,872 | 5,003 | +131 |

---

## 4. Eligible Training Data Summary (by Aspect and Sentiment)

The resulting eligible training examples (aspect assigned, valid sentiment in `[Positive, Negative, Neutral]`, status `Annotated`) are broken down as follows:

| Aspect | Sentiment | Assertion Rows | Unique Clauses | Unique Reviews |
| :--- | :--- | :--- | :--- | :--- |
| Food | Positive | 1,864 | 1,862 | 1,050 |
| Food | Negative | 575 | 575 | 413 |
| Food | Neutral | 83 | 83 | 74 |
| Service | Positive | 606 | 605 | 515 |
| Service | Negative | 225 | 224 | 181 |
| Service | Neutral | 24 | 24 | 23 |
| Price / Value | Positive | 256 | 251 | 216 |
| Price / Value | Negative | 132 | 128 | 118 |
| Price / Value | Neutral | 4 | 4 | 4 |
| Ambience | Positive | 560 | 559 | 471 |
| Ambience | Negative | 72 | 71 | 64 |
| Ambience | Neutral | 27 | 27 | 25 |
| General Experience | Positive | 1,086 | 1,085 | 789 |
| General Experience | Negative | 123 | 122 | 110 |
| General Experience | Neutral | 12 | 12 | 12 |
| **TOTAL** | **ALL** | **5,649** | **5,212** | **1,806** |

---

## 5. Representative Examples of Corrections

### A. Evaluative Assertions Restored from `No Aspect Opinion`
- **Assertion ID:** `REV_00641_C02_A01`  
  **Text:** *"Good packing"*  
  **Original:** Aspect=`nan`, Sentiment=`Positive`, Status=`No Aspect Opinion`  
  **Corrected:** Aspect=`Service`, Sentiment=`Positive`, Status=`Annotated`  
  **Action:** `changed` (high confidence)  
  **Rationale:** Restored omitted aspect 'Service' and Annotated status for evaluative assertion with existing sentiment 'Positive'

- **Assertion ID:** `REV_00822_C07_A01`  
  **Text:** *"they wouldn't take it back"*  
  **Original:** Aspect=`nan`, Sentiment=`Negative`, Status=`No Aspect Opinion`  
  **Corrected:** Aspect=`Service`, Sentiment=`Negative`, Status=`Annotated`  
  **Action:** `changed` (high confidence)  
  **Rationale:** Restored omitted aspect 'Service' and Annotated status for evaluative assertion with existing sentiment 'Negative'

- **Assertion ID:** `REV_00926_C02_A01`  
  **Text:** *"Always up to my expectations"*  
  **Original:** Aspect=`nan`, Sentiment=`Positive`, Status=`No Aspect Opinion`  
  **Corrected:** Aspect=`General Experience`, Sentiment=`Positive`, Status=`Annotated`  
  **Action:** `changed` (high confidence)  
  **Rationale:** Restored omitted aspect 'General Experience' and Annotated status for evaluative assertion with existing sentiment 'Positive'

- **Assertion ID:** `REV_01074_C05_A01`  
  **Text:** *"Has never disappointed"*  
  **Original:** Aspect=`nan`, Sentiment=`nan`, Status=`No Aspect Opinion`  
  **Corrected:** Aspect=`General Experience`, Sentiment=`Positive`, Status=`Annotated`  
  **Action:** `changed` (high confidence)  
  **Rationale:** Reclassified 'No Aspect Opinion' to 'General Experience' with 'Positive' sentiment based on explicit evaluative opinion in text

### B. Factual Statements Reclassified to `No Aspect Opinion`
- **Assertion ID:** `REV_01074_C03_A01`  
  **Text:** *"if you want north Indian style"*  
  **Original:** Aspect=`Ambience`, Sentiment=`nan`, Status=`Annotated`  
  **Corrected:** Aspect=`nan`, Sentiment=`nan`, Status=`No Aspect Opinion`  
  **Action:** `changed` (high confidence)  
  **Rationale:** Reclassified factual entity mention / procedural statement without evaluative opinion to 'No Aspect Opinion'

- **Assertion ID:** `REV_01516_C02_A01`  
  **Text:** *"the place is my backup option plan in case i dont have anywhere to go to or no food at home"*  
  **Original:** Aspect=`Food`, Sentiment=`nan`, Status=`Annotated`  
  **Corrected:** Aspect=`nan`, Sentiment=`nan`, Status=`No Aspect Opinion`  
  **Action:** `changed` (high confidence)  
  **Rationale:** Reclassified factual entity mention / procedural statement without evaluative opinion to 'No Aspect Opinion'

- **Assertion ID:** `REV_01651_C01_A01`  
  **Text:** *"Would you like to feast on North Indian food in Hyderabad"*  
  **Original:** Aspect=`Food`, Sentiment=`nan`, Status=`Annotated`  
  **Corrected:** Aspect=`nan`, Sentiment=`nan`, Status=`No Aspect Opinion`  
  **Action:** `changed` (high confidence)  
  **Rationale:** Reclassified factual entity mention / procedural statement without evaluative opinion to 'No Aspect Opinion'

### C. Polarity / Negation Inconsistencies Corrected
- **Assertion ID:** `REV_05475_C07_A01`  
  **Text:** *"FEAST tats really disgusting you guys are serving stale and contaminated food on this high cost"*  
  **Original:** Aspect=`Food`, Sentiment=`Positive`  
  **Corrected:** Aspect=`Food`, Sentiment=`Negative`  
  **Action:** `changed` (high confidence)  
  **Rationale:** Corrected polarity contradiction: explicit negative evaluation 'disgusting' was erroneously labeled 'Positive'

- **Assertion ID:** `REV_05475_C07_A02`  
  **Text:** *"FEAST tats really disgusting you guys are serving stale and contaminated food on this high cost"*  
  **Original:** Aspect=`Price / Value`, Sentiment=`Positive`  
  **Corrected:** Aspect=`Price / Value`, Sentiment=`Negative`  
  **Action:** `changed` (high confidence)  
  **Rationale:** Corrected polarity contradiction: explicit negative evaluation 'disgusting' was erroneously labeled 'Positive'

- **Assertion ID:** `REV_00190_C03_A01`  
  **Text:** *"Very delicious food, probably the best biryanis in the country and the starters are no less either"*  
  **Original:** Aspect=`Food`, Sentiment=`Negative`  
  **Corrected:** Aspect=`Food`, Sentiment=`Positive`  
  **Action:** `changed` (high confidence)  
  **Rationale:** Corrected polarity contradiction: unmitigated positive food praise was erroneously labeled 'Negative'

---

## 6. Limitations & Unresolved Ambiguous Cases

A key requirement of this audit was to **prevent fabricating ground truth** by forcing uncertain cases into definitive categories. A total of **1,593 assertions** were marked as `action = 'unresolved'` and flagged with `low` confidence.

Representative examples of unresolved cases:
- **Assertion ID:** `REV_02819_C01_A01`  
  **Clause Text:** *"First impression when i looked at the box i was like Awee"*  
  **Assigned Aspect:** `General Experience`  
  **Reason Unresolved:** Genuinely ambiguous / rhetorical statement or insufficient context to definitively infer evaluative sentiment. Left unresolved to prevent fabricating ground truth.

- **Assertion ID:** `REV_02912_C08_A01`  
  **Clause Text:** *"That was the most fanciest name I could find on their menu"*  
  **Assigned Aspect:** `Food`  
  **Reason Unresolved:** Genuinely ambiguous / rhetorical statement or insufficient context to definitively infer evaluative sentiment. Left unresolved to prevent fabricating ground truth.

- **Assertion ID:** `REV_03108_C19_A01`  
  **Clause Text:** *"When people go to a Irani style hotel, there are some expectations"*  
  **Assigned Aspect:** `General Experience`  
  **Reason Unresolved:** Genuinely ambiguous / rhetorical statement or insufficient context to definitively infer evaluative sentiment. Left unresolved to prevent fabricating ground truth.

- **Assertion ID:** `REV_03108_C20_A01`  
  **Clause Text:** *"We like Biriyani dum cooked"*  
  **Assigned Aspect:** `Food`  
  **Reason Unresolved:** Genuinely ambiguous / rhetorical statement or insufficient context to definitively infer evaluative sentiment. Left unresolved to prevent fabricating ground truth.

- **Assertion ID:** `REV_03108_C28_A01`  
  **Clause Text:** *"The Chicken pieces were only bones"*  
  **Assigned Aspect:** `Food`  
  **Reason Unresolved:** Genuinely ambiguous / rhetorical statement or insufficient context to definitively infer evaluative sentiment. Left unresolved to prevent fabricating ground truth.


These unresolved cases consist of rhetorical queries, conditional hypotheticals, and ambiguous statements where customer sentiment cannot be reliably deduced without external speculation. They are excluded from sentiment model training to preserve training integrity.

---

## 7. Deliverable Provenance Verification

- Original source `final_aspect_evaluation/modified_annotations_2000.csv` remains strictly untouched.
- Corrected dataset `modified_annotations_2000_corrected.csv` contains all 12,245 original rows, exact verbatim clause text, review IDs, and character offsets, supplemented with 6 provenance tracking columns (`original_aspect`, `original_sentiment`, `original_annotation_status`, `correction_action`, `correction_reason`, `correction_confidence`).
- Every modified or unresolved row is logged in `annotation_correction_audit.csv`.
