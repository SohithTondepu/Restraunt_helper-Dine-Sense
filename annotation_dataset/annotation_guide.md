# DineSense AI Annotation Guide & Edge Case Rules

**Version:** 1.0.0  
**Domain:** Establishment (Restaurant & Dining) Review ABSA  

---

## 1. Core Principles

1. **Source Text Integrity is Absolute**:
   - Never correct spelling mistakes, punctuation, or grammar in the source review text.
   - Character offsets are measured directly against the raw string in Python (`len()` and string slicing) including CRLF line endings (`\r\n`).
   - Every offset slice `review_text[start:end]` must match the substring verbatim.

2. **No Invented Evidence or Targets**:
   - `aspect_target_span` must be an exact substring from the review text representing the item or feature evaluated (e.g., `"Saturday lunch"`, `"biryani"`, `"waiter"`, `"ambience"`).
   - If the target is implicit (e.g., in `"A bit expensive"`, where no noun is mentioned), leave `aspect_target_span` as empty string `""` and offsets as null. **Do not fabricate words like "food" or "price".**

3. **Strict Sentiment Polarity Criteria**:
   - **Do not assume sentiment from descriptive statements.** A statement of capability or activity like *"One can also chill with friends and or parents"* is a descriptive observation, not an evaluative opinion. It must be classified with `annotation_status = 'No Aspect Opinion'` rather than forced into Positive.
   - **Neutral vs Unclear**:
     - `Neutral`: Non-polar evaluative statements (e.g., *"The soup had medium spice"*, *"Portion was normal"*).
     - `Unclear`: Ambiguous polarity where the annotator cannot tell if the reviewer was pleased or displeased without additional context.
   - **Do not let Star Ratings override clause sentiment**: Even if a review has a 5-star rating, a clause stating *"parking was difficult"* is strictly `Negative`.

4. **Clause Segmentation & Hierarchy**:
   - Reviews are split at clause boundaries, coordinate conjunctions (`and`, `but`, `however`), semicolons, and sentence endings.
   - If a clause contains multiple aspects, create separate assertion rows under the same `clause_id`.
   - If a clause contains no opinion on any aspect, record one row with `annotation_status = 'No Aspect Opinion'` and leave `aspect`, `aspect_target_span`, `opinion_span`, and `sentiment` empty.

---

## 2. Walkthrough Examples

### Example 1: Explicit Target vs Implicit Target in Price/Value
**Review snippet:** `had Saturday lunch , which was cost effective .`
- **Clause:** `had Saturday lunch , which was cost effective`
- **Assertion:**
  - `aspect`: `Price / Value`
  - `aspect_target_span`: `Saturday lunch` (exact target evaluated in text)
  - `opinion_span`: `cost effective`
  - `sentiment`: `Positive`
  - `theme`: `Value for money`
  - `annotation_status`: `Annotated`
  - *Note:* Do not invent `"Price"` or `"Cost"` as the target span. The noun phrase evaluated in the sentence is `"Saturday lunch"`.

### Example 2: Non-Evaluative Activity / Observation
**Review snippet:** `One can also chill with friends and or parents.`
- **Clause:** `One can also chill with friends and or parents.`
- **Assertion:**
  - `aspect`: `null`
  - `aspect_target_span`: `""`
  - `opinion_span`: `""`
  - `sentiment`: `null`
  - `theme`: `null`
  - `annotation_status`: `No Aspect Opinion`
  - `llm_rationale`: `Descriptive statement of possible activity; expresses no evaluative sentiment or aspect rating.`

### Example 3: Multi-Aspect Single Clause
**Review snippet:** `Great food and quick service.`
- **Clause:** `Great food and quick service.`
  - **Assertion 1 (`..._A01`):**
    - `aspect`: `Food / Dining`
    - `aspect_target_span`: `food`
    - `opinion_span`: `Great`
    - `sentiment`: `Positive`
    - `theme`: `Food quality`
    - `annotation_status`: `Annotated`
  - **Assertion 2 (`..._A02`):**
    - `aspect`: `Staff / Service`
    - `aspect_target_span`: `service`
    - `opinion_span`: `quick`
    - `sentiment`: `Positive`
    - `theme`: `Service speed / Wait time`
    - `annotation_status`: `Annotated`

### Example 4: Contrastive Sentiment
**Review snippet:** `The food was awesome but staff was rude.`
- **Clause 1:** `The food was awesome`
  - `aspect`: `Food / Dining`, target: `food`, opinion: `awesome`, sentiment: `Positive`
- **Clause 2:** `but staff was rude`
  - `aspect`: `Staff / Service`, target: `staff`, opinion: `rude`, sentiment: `Negative`

---

## 3. Ambiguity & `Needs Review` Protocol
Mark `annotation_status = 'Needs Review'` when:
1. Sarcasm is suspected (e.g., *"Thanks for ruining our anniversary dinner"*).
2. Code-switched or mixed languages where nuance is non-obvious.
3. Slang or regional vernacular with ambiguous polarity.
4. The clause contains contradictory sentiment qualifiers that cannot be separated.
