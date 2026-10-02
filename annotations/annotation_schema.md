# DineSense AI Annotation Schema Specification

**Document Version:** 1.0.0  
**Effective Date:** 2026-09-30  
**Target Domain:** Food and Hospitality Reviews (Establishment-Level)  
**Schema Level:** Review → Clause → Assertion Hierarchy (One row per aspect–opinion assertion)

---

## 1. Structural Hierarchy

1. **Review Level (`review_id`, `establishment_id`, `review_timestamp`, `review_text`)**:
   - Represents the source document.
   - Text must be preserved **exactly as originally ingested**, including case, typos, punctuation, and CRLF (`\r\n`) line endings.
   - Review text must never be truncated, paraphrased, or normalized.

2. **Clause Level (`clause_id`, `clause_text`, `clause_start_char`, `clause_end_char`)**:
   - A review is segmented into one or more syntactically or semantically coherent clauses.
   - Offsets are zero-based, end-exclusive (`[clause_start_char, clause_end_char)`), referencing exact coordinates in `review_text`.
   - Every clause satisfies: `review_text[clause_start_char:clause_end_char] == clause_text`.

3. **Assertion Level (`assertion_id`, `aspect`, `aspect_target_span`, `target_start_char`, `target_end_char`, `opinion_span`, `opinion_start_char`, `opinion_end_char`, `sentiment`, `theme`, `annotation_status`, `llm_rationale`, `annotator_version`)**:
   - An assertion is a single aspect–opinion tuple evaluated within a clause.
   - A single clause may spawn zero, one, or multiple assertions:
     - If the clause expresses multiple distinct aspects or opinions (e.g. *"Great food and quick service"*), create separate assertion rows (`REV_XXXXX_C01_A01`, `REV_XXXXX_C01_A02`).
     - If the clause contains no evaluative opinion about any aspect (e.g. *"We reached at 8 PM"*), create one record with `annotation_status = 'No Aspect Opinion'`.

---

## 2. Field Definitions

| Field Name | Type | Nullable | Description & Invariants |
| :--- | :--- | :--- | :--- |
| `review_id` | String | No | Deterministic unique review identifier (e.g., `REV_00073`). |
| `original_row_id` | Integer | No | Exact 0-indexed row position in `cleaned_reviews.csv`. |
| `establishment_id` | String | No | Name / identifier of the establishment (e.g., `Beyond Flavours`). |
| `review_timestamp` | String | No | Ingested timestamp from source review. |
| `review_text` | String | No | Exact, unmodified source review text. |
| `clause_id` | String | No | Deterministic clause identifier (e.g., `REV_00073_C01`). |
| `clause_text` | String | No | Exact substring of the clause. |
| `clause_start_char` | Integer | No | 0-based character index where clause begins in `review_text`. |
| `clause_end_char` | Integer | No | 0-based character index where clause ends (exclusive) in `review_text`. |
| `assertion_id` | String | No | Unique assertion identifier (e.g., `REV_00073_C01_A01`). |
| `annotation_status` | Enum | No | `Annotated`, `Needs Review`, or `No Aspect Opinion`. |
| `aspect` | Enum | Yes | Aspect category. Required when `Annotated`. Must be `null` / empty when `annotation_status == 'No Aspect Opinion'`. |
| `aspect_target_span` | String | Yes | Exact text identifying the target entity/aspect within the clause. Empty/null if implicit or `No Aspect Opinion`. Never invent entities. |
| `target_start_char` | Integer | Yes | 0-based start offset of `aspect_target_span` in `review_text`. |
| `target_end_char` | Integer | Yes | 0-based end offset (exclusive) of `aspect_target_span` in `review_text`. |
| `opinion_span` | String | Yes | Exact opinion-bearing expression in the clause. Empty/null if `No Aspect Opinion`. |
| `opinion_start_char` | Integer | Yes | 0-based start offset of `opinion_span` in `review_text`. |
| `opinion_end_char` | Integer | Yes | 0-based end offset (exclusive) of `opinion_span` in `review_text`. |
| `sentiment` | Enum | Yes | `Positive`, `Negative`, `Neutral`, or `Unclear`. Required when `Annotated`. Empty/null if `No Aspect Opinion`. |
| `theme` | String | Yes | Controlled theme describing the specific experience/issue (e.g., `Food quality`, `Staff courtesy`). |
| `llm_rationale` | String | No | Explicit reasoning explaining why the aspect, span, and sentiment were chosen. |
| `annotator_version` | String | No | Identifier for model, prompt, and pipeline version. |

---

## 3. Allowed Enumerations

### A. `annotation_status`
- **`Annotated`**: The clause contains a clear aspect–opinion assertion with identifiable polarity.
- **`Needs Review`**: The assertion is ambiguous, sarcastic, multilingual, context-dependent, or contains an unresolved conflict.
- **`No Aspect Opinion`**: The clause is purely descriptive, factual, procedural, or conversational without an evaluative judgment about an aspect.
  > **Note**: `No Aspect Opinion` is an **annotation status**, NOT an aspect category. When selected, `aspect`, `sentiment`, `aspect_target_span`, and `opinion_span` must be blank.

### B. `aspect` Categories
1. `Food / Dining` — Food taste, ingredients, temperature, portion sizes, dishes, drinks, menu.
2. `Staff / Service` — Waiters, managers, courtesy, speed of service, attentiveness, order errors.
3. `Facilities / Amenities` — Ambience, seating, decor, music, noise, AC, lighting, restrooms, parking.
4. `Price / Value` — Cost, affordability, value for money, bill charges, discounts, pricing fairness.
5. `Cleanliness` — Hygiene, table cleanliness, cutlery cleanliness, washroom cleanliness, hair/insects.
6. `Location` — Accessibility, parking area, neighborhood, view, travel convenience.
7. `Booking / Check-in / Check-out` — Reservations, waitlists, table allocation, billing queue.
8. `Safety / Security` — Foodborne illness/safety, physical safety, security guards.
9. `Room` — Used only if an establishment has accommodation (reserved for lodging/hotel context).
10. `Other` — Recognizable establishment aspect that strictly falls outside above categories.

### C. `sentiment` Polarity
- `Positive`: Explicit favorable opinion or praise.
- `Negative`: Explicit unfavorable evaluation, dissatisfaction, or complaint.
- `Neutral`: Non-polar factual assessment or balanced evaluation devoid of positive or negative emotion.
- `Unclear`: Sentiment polarity cannot be reliably deduced from the available text.

---

## 4. Verification Template Columns
For human verification (`verified_annotations_template.csv`), the following columns are added to each assertion row:
- `verification_status` (`Pending`, `Verified`, `Corrected`, `Rejected`)
- `verified_aspect` (Corrected aspect, or left blank if verified as-is)
- `verified_target_span` (Corrected target span)
- `verified_opinion_span` (Corrected opinion span)
- `verified_sentiment` (Corrected sentiment polarity)
- `verified_theme` (Corrected theme)
- `correction_notes` (Explanation for human edits or rejections)
