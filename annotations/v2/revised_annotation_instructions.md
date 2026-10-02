# DineSense AI Revised Annotation Guidelines (Version 2.0)

**Effective Date:** 2026-09-30  
**Domain:** Establishment (Restaurant & Dining) Review ABSA  
**Schema Level:** Review → Clause → Assertion Hierarchy (with Clause Elaboration Linking)

---

## 1. Core Principles & Structural Constraints

1. **Source Text & Offset Invariants**:
   - The original review text must be preserved **verbatim**, including case, punctuation, typos, and CRLF (`\r\n`) line endings.
   - All character offsets (`clause_start_char`, `clause_end_char`, `target_start_char`, `target_end_char`, `opinion_start_char`, `opinion_end_char`) are **zero-based and end-exclusive** (`[start, end)`), computed strictly against the unmodified `review_text`.
   - Invariant: `review_text[start:end] == span_text`.

2. **Hierarchical Unit of Annotation**:
   - **Review**: Complete user document from a specific establishment.
   - **Clause**: Syntactic or rhetorical segment.
   - **Assertion**: Single aspect–opinion evaluation. A clause may generate zero, one, or multiple assertions.

3. **Elaboration & Descriptive Clause Linking**:
   - A clause that merely elaborates on, names items for, or provides context to a prior assertion must **not** duplicate sentiment or receive an independent artificial rating.
   - Explicit fields added:
     - `clause_relation`: `primary_assertion`, `elaborates_on`, `descriptive_context`, or `independent`.
     - `elaboration_of`: `<parent_assertion_id>` if this clause provides subordinate detail for an earlier assertion; `null` otherwise.
   - If an elaborating clause has no independent evaluative opinion (e.g., *"Crispy veg and corn 65 were other vegetarian starters which we had"* following an appetizer discussion), classify as `annotation_status = 'Elaborating Clause'` or `'No Aspect Opinion'` and link `elaboration_of`.

---

## 2. Specific Semantic Corrections

### A. Negation and Contrast Handling
- **Contextual Polarity**: Never assign sentiment by looking at isolated polar tokens. The presence of the word `"good"` does not make `"not good"` positive.
- **Span Completeness**: When a sentiment word is negated or modified, include the negation particle in the opinion span:
  - `"The food is not good, very bad"` →
    - Opinion Span: `"not good, very bad"` (or two linked opinions: `"not good"`, `"very bad"`)
    - Sentiment: `Negative`
  - `"not that great"` → Opinion Span: `"not that great"`, Sentiment: `Negative`
  - `"has nothing close to sizzler"` → Opinion Span: `"nothing close to"`, Sentiment: `Negative`
  - `"never disappointed"` → Opinion Span: `"never disappointed"`, Sentiment: `Positive`

### B. Opinion Span vs. Descriptive Attribute Selection
- **Evaluative Expressions Only**: An opinion span must capture the subjective appraisal or reaction (e.g., *delicious, awful, courteous, slow, overpriced*).
- **Descriptive Attributes Are Not Opinions**: Dish ingredients, preparation formats, temperatures, and menu titles are factual descriptions unless accompanied by subjective appraisal:
  - *Example:* `"...served with crispy fried noodles, it tasted delicious"`
    - Target: `"noodles"` (or `"it"`)
    - Opinion Span: `"delicious"` (evaluative expression)
    - Non-Opinion: `"crispy fried"` describes the item/ingredient format; it must **not** be tagged as an opinion span.
  - *Example:* `"Crispy veg and corn 65 were other vegetarian starters which we had"`
    - `"Crispy veg"` is the proper name of the starter dish on the menu.
    - Opinion Span: `""` (none).
    - Status: `No Aspect Opinion` / `Descriptive Context`.

### C. Context-Sensitive Words
- Words like `"fine"`, `"hot"`, `"cold"`, and `"place"` must be disambiguated by syntax and meaning:
  - `"fine-dine restaurant"`: Describes the dining category/format. Do **not** annotate `"fine"` as a neutral opinion.
  - `"one fine night"`: Temporal idiom. Non-evaluative.
  - `"the food was fine"`: Evaluative neutral statement (`Sentiment = Neutral`).
  - `"hot chicken wings"`: Dish title / preparation attribute.
  - `"food was cold when served"`: Evaluative negative opinion on food temperature (`Sentiment = Negative`).

### D. Grounded Aspect & Theme Evidence
- Never assign an aspect or theme without explicit textual support.
- **Handling Global / Whole-Establishment Praise**:
  - Reviews often evaluate the entire dining establishment as a whole:
    - *Examples:* `"MUST TRY PLACE"`, `"The place is amazing"`, `"Best place for foodies"`, `"A must try in that area"`.
  - In v1, the presence of the word `"place"` erroneously triggered `Facilities / Amenities` → `Air conditioning / Ventilation`.
  - In v2, evaluations targeting the whole venue/restaurant are categorized under:
    - **Aspect:** `General / Whole Establishment`
    - **Theme:** `Overall experience` or `Establishment recommendation`
    - **Target:** `"PLACE"`, `"place"`, or `""` (implicit)
    - **Strict Prohibition:** Never invent `Air conditioning / Ventilation` or `Facilities / Amenities` for holistic praise.

### E. Reserved Role of `Other` and Explicit Implicit-Target Handling
- **`Other` is NOT a default for uncertainty**:
  - Use `Other` strictly when the text clearly discusses an identifiable aspect of the establishment that falls outside the defined taxonomy (e.g., *Valet parking*, *Live sports screening*, *Merchandise*).
- **Implicit Targets**:
  - When an opinion is clearly expressed about an aspect (e.g., `"A bit expensive"`, `"too tasty it is"`), assign the aspect (`Price / Value`, `Food / Dining`) and leave `aspect_target_span` as `""` with offsets `null`.
  - If the aspect itself cannot be identified from the text or immediate context, set `aspect = "Unclear"` and `theme = "Unspecified"` and mark `annotation_status = 'Needs Review'`. Do not invent a target or dump it into `Other`.

### F. Sentiment Polarity Classes
The five allowed sentiment polarities are strictly defined:
1. `Positive`: Clear favorable opinion, commendation, or satisfaction.
2. `Negative`: Clear unfavorable opinion, complaint, criticism, or dissatisfaction.
3. `Neutral`: Explicitly non-evaluative, factual, or balanced evaluation devoid of positive or negative emotion (e.g., *"spice level was medium"*, *"portion was standard"*).
4. `Mixed`: Both positive and negative sentiment elements co-occur in the same atomic assertion and cannot be cleanly separated (e.g., *"tasty but way too oily"*).
5. `Unclear`: Text is ambiguous or insufficient to determine whether the reviewer was pleased or displeased.
   *(Note: Decoupled from aspect uncertainty. A clear aspect can have `Unclear` sentiment, and vice versa).*

---

## 3. Revised Taxonomy

### Aspects:
- `Food / Dining`
- `Staff / Service`
- `Facilities / Amenities`
- `Price / Value`
- `Cleanliness`
- `Location`
- `Booking / Check-in / Check-out`
- `General / Whole Establishment` *(NEW: for whole-venue appraisals like "MUST TRY PLACE", "Best place for foodies")*
- `Safety / Security`
- `Other` *(Reserved: only for identifiable non-standard aspects)*
- `Unclear` *(Reserved: when aspect cannot be deduced from evidence)*

### Sentiments:
- `Positive`
- `Negative`
- `Neutral`
- `Mixed`
- `Unclear`

### Annotation Statuses:
- `Annotated` (Complete, confident aspect-opinion assertion)
- `Elaborating Clause` (Subordinate clause elaborating on an earlier assertion)
- `No Aspect Opinion` (Factual, procedural, conversational statement)
- `Needs Review` (Ambiguous, sarcastic, or unresolved edge cases)
