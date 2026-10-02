# DineSense AI: 5-Category Aspect Classification Audit & Evaluation Report

**Audit Date:** 2026-10-01  
**Target Dataset:** 2,000 Milestone Restaurant Reviews (11,631 Unique Clauses)  
**Evaluation Input Table:** `data/processed/clause_evaluation_input_2000.csv`  
**Prediction Records:** `aspect_evaluation/aspect_predictions_2000.csv`  
**Audit & Metric CSV Directory:** `aspect_evaluation/csv/`  

---

## 1. Verification of Reference-Label Construction

- **Clause-Level & Multi-Label Evaluation:**  
  The evaluation is strictly clause-level and multi-label. Each of the 11,631 clauses is evaluated independently against a ground truth set of aspect labels $Y_i \subseteq \{\text{Food}, \text{Service}, \text{Price / Value}, \text{Ambience}, \text{General Experience}\}$. If $Y_i = \emptyset$, the clause represents `No Aspect Opinion`.
- **Assertion-to-Clause Aggregation:**  
  All 11,635 assertion-level records in `annotations_2000_final.csv` were grouped by `clause_id`. For each clause, non-null aspect strings from its constituent assertions were aggregated into a unique set.
- **Reconciliation of 4,954 Clauses vs. 4,958 Aspect Supports:**  
  - Total unique clauses with at least one aspect: **4,954 clauses**.
  - Single-aspect clauses: **4,950 clauses** (contributing $4,950 \times 1 = 4,950$ aspect supports).
  - Multi-aspect clauses: exactly **4 clauses**, each containing exactly 2 co-occurring assertions ($4 \times 2 = 8$ aspect supports):
    1. `REV_01169_C01`: `"Amazing food and ambience"` $\rightarrow$ `{'Ambience', 'Food'}` (+1 Ambience, +1 Food)
    2. `REV_06351_C01`: `"Great place with nice ambience"` $\rightarrow$ `{'Ambience', 'General Experience'}` (+1 Ambience, +1 General Experience)
    3. `REV_06351_C03`: `"Mouth watering food with neat service"` $\rightarrow$ `{'Food', 'Service'}` (+1 Food, +1 Service)
    4. `REV_06755_C01`: `"Great place and great food"` $\rightarrow$ `{'Food', 'General Experience'}` (+1 Food, +1 General Experience)
  - **Sum of aspect supports:** $4,950 + 8 = 4,958$ ($2,101 + 683 + 283 + 533 + 1,358 = 4,958$).  
  **Exact mathematical identity confirmed:** the difference of $+4$ arises entirely from the 4 multi-aspect clauses.

---

## 2. Corrected Binary One-vs-Rest Evaluation Metrics

Each aspect category is evaluated as an independent binary decision across all $N = 11,631$ clauses ($TP + FN = \text{Support}$; $TP + FP + FN + TN = 11,631$).  
`No Aspect Opinion` is treated as the absence of all 5 labels and is **strictly excluded from the 5-aspect Macro/Micro averages**.

*Saved to [`aspect_evaluation/csv/aspect_classification_report.csv`](file:///d:/Aspect-Based-Sentimental-Analysis-on-Food-Reviews/aspect_evaluation/csv/aspect_classification_report.csv)*:

| Category | TP | FP | FN | TN | Precision | Recall | F1-Score | Specificity | Support ($TP+FN$) | Total ($N$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Price / Value** | 281 | 372 | 2 | 10,976 | **0.4303** | **0.9929** | **0.6004** | 0.9672 | 283 | 11,631 |
| **Ambience** | 494 | 809 | 39 | 10,289 | **0.3791** | **0.9268** | **0.5381** | 0.9271 | 533 | 11,631 |
| **Food** | 1,825 | 2,996 | 276 | 6,534 | **0.3786** | **0.8686** | **0.5273** | 0.6856 | 2,101 | 11,631 |
| **Service** | 648 | 1,290 | 35 | 9,658 | **0.3344** | **0.9488** | **0.4945** | 0.8822 | 683 | 11,631 |
| **General Experience** | 609 | 741 | 749 | 9,532 | **0.4511** | **0.4485** | **0.4498** | 0.9279 | 1,358 | 11,631 |
| **Macro Average (5 Aspects)** | 3,857 | 6,208 | 1,101 | — | **0.3947** | **0.8371** | **0.5220** | — | 4,958 | 11,631 |
| **Micro Average (5 Aspects)** | 3,857 | 6,208 | 1,101 | — | **0.3832** | **0.7779** | **0.5135** | — | 4,958 | 11,631 |
| **Weighted Average (5 Aspects)** | 3,857 | 6,208 | 1,101 | — | **0.3954** | **0.7779** | **0.5069** | — | 4,958 | 11,631 |

*Complete $2 \times 2$ contingency tables for each aspect are saved to [`aspect_evaluation/csv/aspect_binary_contingency_tables.csv`](file:///d:/Aspect-Based-Sentimental-Analysis-on-Food-Reviews/aspect_evaluation/csv/aspect_binary_contingency_tables.csv).*

---

## 3. Audit of the Confusion Matrix

### Why Original Confusion Matrix Row Totals Did Not Match Aspect Supports
In single-label multi-class classification, each sample belongs to exactly one row and one column, ensuring $\sum_j M_{ij} = \text{Support}_i$.  
In multi-label classification:
1. When a clause has 1 reference aspect (e.g. `Food`), but the classifier predicts multiple aspects (e.g. `{'Food', 'Service'}`), the cross-product counting logic previously added $+1$ to `(Food, Food)` AND $+1$ to `(Food, Service)`. This overcounted row `Food` by 225.
2. When a `No Aspect Opinion` clause had multiple false positive predictions (e.g. food + service), it added $+1$ to multiple columns in row `No Aspect Opinion`, overcounting that row by 426.

### Documented Co-Occurrence Matrix
*Saved to [`aspect_evaluation/csv/aspect_confusion_matrix.csv`](file:///d:/Aspect-Based-Sentimental-Analysis-on-Food-Reviews/aspect_evaluation/csv/aspect_confusion_matrix.csv).*  
**Counting Convention:** Cell $(R_i, P_j)$ counts the number of clauses having reference aspect $R_i$ for which the classifier predicted aspect $P_j$. A clause with multiple predicted aspects contributes to multiple column entries within its reference row.

| Reference Aspect ($R_i$) | Pred: Food | Pred: Service | Pred: Price / Value | Pred: Ambience | Pred: General Experience | Pred: No Aspect (Empty) | Total Ref Clauses |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Food** | **1,825** | 149 | 46 | 64 | 57 | 185 | 2,101 |
| **Service** | 122 | **648** | 19 | 55 | 21 | 11 | 683 |
| **Price / Value** | 80 | 41 | **281** | 6 | 9 | 0 | 283 |
| **Ambience** | 92 | 27 | 12 | **494** | 47 | 10 | 533 |
| **General Experience** | 300 | 93 | 47 | 58 | **609** | 373 | 1,358 |
| **No Aspect Opinion** | 2,405 | 981 | 248 | 628 | 609 | **2,232** | 6,677 |

### Exact Match & Subset Accuracies
- **Overall Exact Match Ratio (All 11,631 Clauses):** **46.53%** ($5,412 / 11,631$)
- **Evaluative Exact Match Ratio (4,954 Annotated Clauses Only):** **64.19%** ($3,180 / 4,954$)
- **Evaluative At-Least-One Overlap Recall:** **77.78%** ($3,853 / 4,954$)
- **Non-Evaluative Specificity (6,677 No Aspect Clauses Only):** **33.43%** ($2,232 / 6,677$)

---

## 4. No Aspect Opinion Audit (Absence of All Labels)

*Saved to [`aspect_evaluation/csv/aspect_no_aspect_audit.csv`](file:///d:/Aspect-Based-Sentimental-Analysis-on-Food-Reviews/aspect_evaluation/csv/aspect_no_aspect_audit.csv)*:

| Metric | Count | Share |
| :--- | :---: | :---: |
| **1. Reference clauses with no aspect (`No Aspect Opinion`)** | **6,677** | 57.41% of total corpus |
| **2. Predicted clauses with no aspect (Empty Prediction)** | **2,811** | 24.17% of total corpus |
| **3. Correct no-aspect detections ($Y = \emptyset \land \hat{Y} = \emptyset$)** | **2,232** | 33.43% of ref no-aspect clauses |
| **4. False no-aspect predictions ($|Y| \ge 1 \land \hat{Y} = \emptyset$)** | **579** | 11.69% of annotated clauses |
| **5. Aspect predictions made on ref `No Aspect Opinion` (False Alarms)** | **4,445** | 66.57% of ref no-aspect clauses |

### Breakdown of False No-Aspect Predictions (579 Missed Clauses)
- Missed `General Experience`: **373** clauses (implicit holistic satisfaction phrases lacking explicit keywords).
- Missed `Food`: **185** clauses (unusual dish names, typos like *"biriyani"*, or descriptive preparation).
- Missed `Service`: **11** clauses.
- Missed `Ambience`: **10** clauses.
- Missed `Price / Value`: **0** clauses (100% recall on pricing phrases against empty predictions).

### Breakdown of False Alarms on Reference `No Aspect Opinion` (4,871 Aspect Predictions across 4,445 Clauses)
- Falsely predicted `Food`: **2,405** (factual ordering statements with food nouns, e.g., *"We ordered two plates of chicken biryani"*).
- Falsely predicted `Service`: **981** (procedural check-in / table actions, e.g., *"attendant was Papiya"*, *"asked for bill"*).
- Falsely predicted `Ambience`: **628** (spatial / arrival statements, e.g., *"Have been here and it was crowded"*).
- Falsely predicted `General Experience`: **609** (non-evaluative mentions of the establishment, e.g., *"visiting this restaurant"*).
- Falsely predicted `Price / Value`: **248** (numerical bill / time mentions, e.g., *"Our bill was 636"*).

---

## 5. False Positive & False Negative Representative Examples

*Saved to [`aspect_evaluation/csv/aspect_error_breakdown.csv`](file:///d:/Aspect-Based-Sentimental-Analysis-on-Food-Reviews/aspect_evaluation/csv/aspect_error_breakdown.csv)*:

### 1. Food
- **False Positive:** `"We asked them to exchange for another drink"` (Ref: `No Aspect Opinion`; triggered by `'drink'`).
- **False Positive:** `"Food"` (Ref: `No Aspect Opinion`; conversational header word without opinion).
- **False Negative:** `"Ege biriyani is very good"` (Misspelling *"biriyani"* not in `'biryani'` lexicon; routed semantically to General Experience).
- **False Negative:** `"Main course is also pretty below average"` (*"main course"* phrase not matched by single-word food tokens).

### 2. Service
- **False Positive:** `"Had dinner at this place and the attendant was Papiya"` (Ref: `No Aspect Opinion`; triggered by `'attendant'`).
- **False Positive:** `"Nice food and nice service"` (Ref: `Food` only; co-occurring service noun triggered `Service`).
- **False Negative:** `"waters think themself like heroes here"` (Typo *"waters"* instead of *"waiters"* bypassed lexicon).
- **False Negative:** `"The packaging was good and simple"` (*"packaging"* annotated as Service in v2.2, but absent from core service lexicon).

### 3. Price / Value
- **False Positive:** `"Definitely worth a try"` (Ref: `General Experience`; triggered by `'worth'`).
- **False Positive:** `"They took 15min to print a bill 😂😂"` (Ref: `No Aspect Opinion`; triggered by `'bill'`).
- **False Negative:** `"This outlet is making a fast buck in the Hitech area"` (Idiom *"fast buck"* not captured lexically).
- **False Negative:** `"The AC temperature was a tad bit on the higher side"` (Ref: Price / Value sub-clause misrouted to Ambience/Service).

### 4. Ambience
- **False Positive:** `"Have been here quite a number of times and always find it crowded"` (Ref: `No Aspect Opinion`; triggered by `'crowded'`).
- **False Positive:** `"The staff is very friendly and the ambiance is awesome"` (Ref: `Service` assertion; ambience noun triggered `Ambience`).
- **False Negative:** `"No theme no decoration"` (*"theme"*, *"decoration"* absent from ambience lexicon).
- **False Negative:** `"Nice place in paradise gachibowli"` (Routed to General Experience rather than Ambience).

### 5. General Experience
- **False Positive:** `"i think this place is very good place"` (Ref: `Ambience`; triggered by `'place'` repetition).
- **False Positive:** `"we are visiting flehazo restaurant that's lovely place"` (Ref: `No Aspect Opinion`; triggered by `'lovely place'`).
- **False Negative:** `"Never disappoints me"` (Implicit repeat satisfaction lacking explicit "visit"/"recommend" keyword).
- **False Negative:** `"Always up to my expectations"` (Holistic satisfaction without explicit venue noun).
