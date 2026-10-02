# DineSense AI: Unresolved Cases Requiring Human Judgment (25-Review Test Batch)

**Document Version:** 2.0  
**Date:** 2026-09-30  

The following 5 cases from the 25-review test batch represent genuine edge cases where textual evidence is ambiguous or where multiple valid interpretations exist. These require explicit human verification.

---

### Case 1: Rating Breakdowns Formatted as Text (`REV_06534`)
**Review Text:** `"Food -4/5 \r\nAmbience - 5/5 \r\nValue for money - 3.5/5 \r\nService -4/5"`  
- **Dilemma:** Reviewers explicitly typing out numerical aspect scores in the review body.
- **Interpretation A (Annotated):** Treat typed ratings (e.g. `4/5`) as evaluative opinions with corresponding sentiment (`Positive` for 4/5, `Neutral` for 3.5/5).
- **Interpretation B (No Aspect Opinion / Metadata):** Treat numerical breakdowns as structured metadata that duplicate star ratings rather than natural language opinion spans.
- **Current V2 Handling:** Marked as `No Aspect Opinion` / `Needs Review` to avoid forcing non-linguistic spans into opinion strings.

### Case 2: Normative / Counterfactual Desiderata (`REV_04412`)
**Review Text:** `"Rolls should be crispy with stuff not coming out."`  
- **Dilemma:** Does a counterfactual statement (*"should be X"*) constitute an explicit Negative opinion on the current dish?
- **Interpretation A:** Implicit Negative opinion targeting `Rolls` with opinion `should be crispy` (implying they were not crispy).
- **Interpretation B:** Prescriptive opinion / suggestion rather than direct evaluative assertion of actual consumption.
- **Current V2 Handling:** Marked as `Annotated` (`Negative`, `Food quality`), but flagged for human validation.

### Case 3: Food Preparation Attribute vs Displeasure (`REV_06090`)
**Review Text:** `"The food here is somewhat oily but it tastes delicious."`  
- **Dilemma:** Is `somewhat oily` always a complaint/Negative assertion when followed immediately by `tastes delicious`?
- **Interpretation A:** Split into Negative (`oily`) and Positive (`delicious`).
- **Interpretation B:** Single assertion with `Mixed` sentiment on `Food quality`.
- **Current V2 Handling:** Split into two separate assertions with coordinate linking, allowing contrastive ABSA training.

### Case 4: Rhetorical / Hyperbolic Complaints (`REV_08253`)
**Review Text:** `"if there was minus rating, I would have given that."`  
- **Dilemma:** Hyperbolic rhetorical expression without explicit aspect nouns.
- **Interpretation A:** `General / Whole Establishment` with `Negative` sentiment, opinion span `if there was minus rating`.
- **Interpretation B:** Conversational hyperbole classified as `No Aspect Opinion`.
- **Current V2 Handling:** Assigned to `General / Whole Establishment`, `Negative`, theme `Overall experience`.

### Case 5: Instagram Handle Sign-Offs (`REV_05070`)
**Review Text:** `"Do follow us on Instagram:forkandspoonstoryhyd"`  
- **Dilemma:** Promotional food blogger boilerplate appearing at start and end of reviews.
- **Current V2 Handling:** Strictly classified as `No Aspect Opinion` with `clause_relation = 'independent'`.
