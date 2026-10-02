# Pilot Annotation Summary Report (100 Reviews)

**Execution Date:** 2026-09-30  
**Model & Pipeline:** `gemini-3.8-flash-preannotator-v1.0`  
**Dataset Scope:** 100-Review Pilot Sample (1 review per establishment across 100 establishments)  

---

## 1. Executive Metrics

| Metric | Count | Notes |
| :--- | :--- | :--- |
| **Total Reviews Processed** | 100 | Exactly 1 review per establishment |
| **Total Clauses Segmented** | 647 | Mean 6.47 clauses/review |
| **Total Assertions Generated** | 694 | Mean 6.94 assertions/review |
| **Validation Checks** | **100% Passed** | Zero character offset mismatches |

---

## 2. Assertion Status Breakdown

| Annotation Status | Count | Percentage |
| :--- | :--- | :--- |
| `Annotated` | 359 | 51.7% |
| `No Aspect Opinion` | 328 | 47.3% |
| `Needs Review` | 7 | 1.0% |

---

## 3. Aspect Distribution (Excluding 'No Aspect Opinion')

| Aspect Category | Count | Percentage of Evaluative Assertions |
| :--- | :--- | :--- |
| `Food / Dining` | 154 | 42.1% |
| `Other` | 94 | 25.7% |
| `Facilities / Amenities` | 57 | 15.6% |
| `Staff / Service` | 40 | 10.9% |
| `Price / Value` | 15 | 4.1% |
| `Location` | 5 | 1.4% |
| `Cleanliness` | 1 | 0.3% |

---

## 4. Sentiment Polarity Distribution

| Sentiment | Count | Percentage of Evaluative Assertions |
| :--- | :--- | :--- |
| `Positive` | 276 | 75.4% |
| `Negative` | 63 | 17.2% |
| `Neutral` | 20 | 5.5% |
| `Unclear` | 7 | 1.9% |

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
