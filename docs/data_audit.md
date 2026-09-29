# Data Audit & Preprocessing Report

**Dataset:** Restaurant Reviews Dataset (Hyderabad Zomato/Food Dining Reviews)  
**Total Raw Rows:** 10,000  
**Final Cleaned Rows:** 9,633  
**Splits:** 70% Train (6,727) / 15% Validation (1,439) / 15% Test (1,467)  
**Splitting Strategy:** Grouped by `Reviewer` (`GroupShuffleSplit`, random seed 42) to prevent reviewer writing-style leakage between splits.

---

## 1. Raw Data Anomalies & Cleaning Actions

1. **Rogue Header Column `7514`:**
   - *Finding:* The raw CSV contained an extra column header named `7514` with a solitary non-null float value at row 0 (`2447.0`), caused by an unescaped comma in an upstream export.
   - *Action:* Dropped column `7514`.
2. **Invalid & Non-Numeric Ratings:**
   - *Finding:* One row contained `'Rating' == 'Like'`; 38 rows had missing ratings.
   - *Action:* Coerced ratings to floats, removed rows where `Rating` or `Review` was null.
3. **Empty & Short String Reviews:**
   - *Finding:* 45 rows contained reviews shorter than 4 characters (e.g. `"."`, `"ok"`).
   - *Action:* Filtered out reviews with character length $\le 3$.
4. **Duplicate Reviews:**
   - *Finding:* 265 exact duplicate reviews for the same restaurant.
   - *Action:* Deduplicated subset `(Restaurant, Review)` to eliminate cross-split train-test contamination.
5. **Reviewer Attribution:**
   - 38 rows had missing reviewer names; imputed as `'Anonymous'`.

---

## 2. Rating & Sentiment Label Distribution

Ratings were mapped into 3 standard review-sentiment categories:
- **Negative (Class 0):** $\text{Rating} \le 2.0$ (Star ratings: 1.0, 1.5, 2.0)
- **Neutral (Class 1):** $2.5 \le \text{Rating} \le 3.5$ (Star ratings: 2.5, 3.0, 3.5)
- **Positive (Class 2):** $\text{Rating} \ge 4.0$ (Star ratings: 4.0, 4.5, 5.0)

| Sentiment Class | Rating Range | Count (Cleaned) | Percentage |
|:---|:---:|:---:|:---:|
| **Positive** | 4.0 – 5.0 | 6,031 | 62.6% |
| **Negative** | 1.0 – 2.0 | 2,408 | 25.0% |
| **Neutral** | 2.5 – 3.5 | 1,194 | 12.4% |
| **Total** | 1.0 – 5.0 | **9,633** | **100.0%** |

### Critical Observation on Neutral Class (3-Star Reviews)
Neutral reviews constitute only **12.4%** of the dataset. Manual inspection of 50 3-star reviews reveals the **"Mixed Review Dilemma"**:
> *"The chicken biryani was full of aroma and authentic spices, but the waiter took 45 minutes to get the bill and was unapologetic."*

In dining reviews, 3-star ratings rarely indicate "bland neutrality"; they reflect **conflicting aspect polarities** (Food Positive + Service Negative). This provides the core empirical motivation for Phase 4 (Aspect-Based Sentiment Analysis).

---

## 3. Review Length & Tokenization Profile

- **Average Character Length:** 188 characters
- **Median Word Count:** 24 words
- **95th Percentile Word Count:** 94 words
- **Max Word Count:** 320 words

**Token Length Setting for Transformers:**  
Since the 95th percentile of word length is 94 words (~115 subword tokens), `max_length = 128` comfortably covers >96% of full reviews without truncation, optimizing CPU/GPU memory footprint and inference latency.
