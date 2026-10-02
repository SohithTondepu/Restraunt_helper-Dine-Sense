# DineSense AI — Restaurant Intelligence Pipeline
## Implementation Action Plan for Antigravity

**Document type:** Engineering execution specification  
**Audience:** Antigravity coding agent and project owner  
**Goal:** Evolve the existing DineSense AI ABSA application into an evidence-grounded restaurant intelligence workflow.  
**Execution principle:** Inspect the repository first; preserve working functionality; implement in small, testable increments; never fabricate data, causal explanations, or evaluation results.

---

# 0. Mission and Definition of Done

Build an end-to-end pipeline that turns restaurant reviews into traceable aspect–opinion evidence, aggregates recurring themes, surfaces descriptive diagnostics, and presents carefully qualified operational follow-up suggestions.

The system must answer:
1. **What are customers saying?** — aspect/opinion records with exact review evidence.
2. **How common or recent is it?** — transparent aggregation with denominators and time windows.
3. **What patterns are associated with it?** — descriptive diagnostics, explicitly not causal claims.
4. **What could the operator investigate or try?** — evidence-linked suggestions labelled as hypotheses or actions to validate.

The central design object is an **evidence-linked aspect-opinion record**. Metrics, diagnostics, and recommendations must derive from these records rather than being generated independently from raw text.

## Definition of Done

- Reviews can be ingested, validated, segmented, and processed through existing ABSA components.
- The pipeline emits zero or more structured aspect-opinion records per review.
- Every record retains stable review identity and exact evidence text.
- Pre-annotation can be exported, corrected, imported, and evaluated without using the protected gold benchmark.
- Canonical themes group related expressions without erasing original wording.
- Restaurant prevalence, sentiment, and trends show explicit denominators, periods, and low-sample safeguards.
- Diagnostics describe associations; they do not assert causes from reviews alone.
- Recommendations link to evidence and distinguish possible drivers from verified causes.
- Existing Streamlit workflows remain usable or are migrated with regression tests.
- Unit, integration, data-quality, and regression tests pass.
- Documentation covers setup, contracts, commands, limitations, and operation.

---

# 1. Non-Negotiable Engineering Rules

1. **Repository truth first.** Do not assume paths, APIs, or current behaviour. Inspect before editing.
2. **Preserve working behaviour.** Prefer adapters and incremental changes over broad rewrites.
3. **No invented evidence.** Quotes must be exact source text.
4. **No causal overclaim.** Reviews support observed complaints and associations, not operational causes. Use “possible driver”, “hypothesis”, or “investigate” unless independent operational data verifies a cause.
5. **No silent label changes.** Keep raw predictions, human corrections, model/version metadata, and annotation status separate.
6. **Gold benchmark protection.** The existing 500-clause gold benchmark is evaluation-only. Do not train on it, tune thresholds against it, or include it in pre-annotation exports. Locate and enforce this exclusion.
7. **Granularity discipline.** Review-level DistilBERT metrics do not establish clause-level sentiment quality.
8. **Uncertainty visible.** Show counts, denominators, time windows, and low-volume warnings.
9. **Reproducibility.** Record model/config versions, run timestamp, and input fingerprint.
10. **Fail clearly.** Invalid rows and inference failures need actionable statuses, not silent drops.
11. **Privacy.** Do not expose reviewer identity or unnecessary personal data.
12. **No hidden LLM dependency.** Core pipeline must work with existing models/rules. Any future LLM is optional and evidence-constrained.

---

# 2. Existing System — Preserve, Wrap, Validate

The project overview documents these capabilities. Verify each in the repository before relying on it.

| Capability | Documented approach | Required treatment |
|---|---|---|
| Dataset | 9,600+ reviews, 100 restaurants | Inspect schema, IDs, duplicates, timestamps, missingness |
| Review sentiment | Fine-tuned DistilBERT | Keep for review-level task; do not assume clause-level performance |
| Segmentation | spaCy sentence splitting + regex discourse markers; RST-inspired | Preserve baseline; test fragments, negation, contrast, mixed polarity |
| Aspect matching | Lexicons + MiniLM centroids; Food, Service, Price, Ambience | Wrap; log matches, scores, threshold, ambiguity |
| Opinion extraction | spaCy dependency patterns | Treat as syntactic extraction, not causal discovery |
| Complaint clusters | Near-synonym grouping with quotes | Refactor as canonical themes with reversible provenance |
| Health score | Time-decayed Empirical Bayes | Keep secondary; expose inputs and limitations |
| SOP mapping | Complaint-to-intervention repository | Convert to evidence-linked suggestions; applicability unverified |
| Fact checks | Regex numeric checks | Retain, then add structured metric references |
| Streamlit | Scorecards, root-cause view, CSV, sandbox, benchmarking | Integrate incrementally; preserve parity |

### Reported metrics — verify before publishing

The overview reports review-level DistilBERT accuracy **87.5%**, Macro-F1 **0.747**, CPU latency **15–25 ms/review**, DeBERTa Neutral F1 **0.510**, and Cohen’s Kappa **0.812** on a 500-clause benchmark. A separate project discussion mentioned Kappa **0.852**; resolve this discrepancy from evaluation artifacts before using either value.

A later clause-level evaluation reported aspect-matching Macro-F1 **0.736** and clause-sentiment Macro-F1 **0.517**. Verify artifacts and label definitions. Never conflate these with review-level metrics.

---

# 3. Phase 0 — Repository Reconnaissance and Baseline

## Tasks

1. List repository structure, excluding environments, caches, model weights, and large data from the summary.
2. Locate application entry points; segmentation, aspect, sentiment, opinion, clustering, health-score, SOP, and Streamlit modules; model/config paths; datasets; annotation files; tests; scripts; notebooks; dependency files.
3. Run current app and tests using repository instructions.
4. Trace one review end-to-end; capture actual intermediate objects and output schemas.
5. Find places where labels are overwritten, review sentiment is used for clauses, evidence is regenerated, “root cause” is claimed, recommendations appear verified, or metrics omit denominators.
6. Locate the 500-clause gold benchmark and metric artifacts. Determine how to enforce exclusion.
7. Create `docs/implementation_baseline.md` with repository map, commands, current flow, test status, risks, and concrete files proposed for change.

## Acceptance criteria

- No broad code changes during reconnaissance.
- App launches or the exact blocker is documented.
- Current input/output contracts are captured with examples.
- Unknowns are listed; do not invent paths or behaviour.

---

# 4. Target Architecture

```text
Raw reviews
   |
[1. Ingestion + validation]
   |
[2. Normalization + segmentation]
   |
[3. Aspect/opinion extraction + sentiment]
   |
[4. Canonical theme assignment]
   |
   +--------------------+
   |                    |
[5. Aggregation]   [Evidence records]
   |                    |
[6. Diagnostics] <------+
   |
[7. Evidence-linked suggestions]
   |
[8. Streamlit views]
```

Keep inference, human annotation, normalization, analytics, decision support, and presentation separate. UI code must not independently recalculate business metrics.

---

# 5. Data Contracts

Use typed models or validated tabular schemas consistent with the repository. Pydantic/dataclasses are acceptable if already available. Add explicit schema versions.

## 5.1 Review

| Field | Type / rule |
|---|---|
| `review_id` | Stable, non-empty unique key |
| `restaurant_id` | Stable restaurant key |
| `review_text` | Original text, never overwritten |
| `review_date` | Date/datetime or null; parse safely |
| `rating` | Numeric or null; validate source range |
| `source` | Optional source label |
| `reviewer_id` | Optional; use for leakage control, not UI |
| `schema_version` | Explicit version |

Keep normalized text separately if needed.

## 5.2 Clause

`clause_id`, `review_id`, `clause_text`, `char_start`, `char_end`, `segment_index`, `segmentation_method`, `segmentation_version`, `is_fragment`.

Offsets must refer to original review text; define end-exclusive convention. If current segmenter cannot track offsets, implement this before claiming span traceability. Record meaningful short fragments rather than silently dropping them.

## 5.3 Aspect-opinion assertion — central object

One row represents one aspect/opinion/sentiment assertion. A clause may yield multiple rows.

| Field | Requirement |
|---|---|
| `assertion_id` | Unique |
| `review_id`, `clause_id` | Foreign keys |
| `aspect` | Food, Service, Price, Ambience, Other, Unassigned |
| `aspect_target` | Exact target span when available |
| `opinion_span` | Exact opinion phrase, not a generated paraphrase |
| `sentiment_label` | Negative, Neutral, Positive, Uncertain |
| `sentiment_scores` | Optional class probabilities |
| `polarity` | Optional `P(pos)-P(neg)`; document as model-derived |
| `aspect_method`, `aspect_score` | Method and score with defined meaning |
| `sentiment_method`, `model_version` | Method/checkpoint |
| `theme_id` | Nullable canonical theme |
| `evidence_text` | Exact clause/source substring |
| `annotation_status` | unreviewed, accepted, corrected, rejected, adjudicated |
| `annotator_id` | Optional pseudonymous ID |
| `created_at`, `schema_version` | Timestamp and schema version |

Support zero assertions, `Unassigned`, and `Uncertain`. “No opinion detected” is not Neutral.

## 5.4 Theme

`theme_id`, `aspect`, `canonical_name`, `definition`, `aliases`, `status` (active/deprecated/needs_review), `created_by`, `version`.

Every assignment must be reversible; retain original phrase and mapping version.

## 5.5 Diagnostic

Store diagnostic ID/type, restaurant/aspect/theme, period and comparison period, numerator, denominator, rate, sample size, minimum-sample status, supporting assertion/review IDs, descriptive interpretation, caveat, and generation version.

## 5.6 Recommendation

Store ID, linked themes/diagnostics, exact evidence references, observed issue, possible driver (hypothesis), suggested investigation/action, rationale, assumptions, validation data needed, and status. Do not fabricate expected impact.

---

# 6. Phase 1 — Ingestion, Validation, Provenance

## Tasks

1. Adapt or create one ingestion entry point for the current source format.
2. Validate required columns, types, nulls, rating range, dates, and ID uniqueness.
3. Detect exact duplicates and near duplicates separately; do not auto-delete near duplicates.
4. Preserve raw text and source columns.
5. Produce validation report: received/valid/rejected rows, reasons, duplicates, missingness, restaurant/date coverage.
6. Reject or quarantine invalid rows with reasons; never silently drop.
7. Save run manifest: input fingerprint, counts, code/config version, timestamp.

## Tests

Missing column, blank text, duplicate ID, malformed date, out-of-range rating, Unicode, empty input, deterministic fingerprint.

## Acceptance

Users can inspect which rows were accepted/rejected and why. Original text remains unchanged.

---

# 7. Phase 2 — Normalization and Discourse-Aware Segmentation

## Tasks

1. Wrap current spaCy sentence splitting and regex rules behind a documented interface.
2. Call it **discourse-aware rule-based segmentation** or **RST-inspired segmentation**, not a full RST parser.
3. Preserve offsets into original review.
4. Test contrast (`but`, `however`, `yet`, `whereas`), concession (`although`, `despite`), punctuation, negation, short fragments, conjunction ambiguity, quotations, and abbreviations.
5. Do not drop short fragments solely due to word count; mark them for downstream handling.
6. Preserve negation and intensifiers.
7. Record method/version and segment index.

Example: “The biryani was excellent, but the waiter was rude and the prices were high.” must not collapse into one review-wide sentiment. Distinct propositions/aspects should remain recoverable.

Create at least 30 segmentation fixtures. Assert offsets and reconstruction where applicable.

## Acceptance

Deterministic output for fixed version/config; evidence text preserved; no claim of full discourse parsing.

---

# 8. Phase 3 — Aspect and Opinion Extraction

## Tasks

1. Define stable interface: clause + context → zero or more candidate assertions.
2. Wrap current hybrid matcher: lexicons first, MiniLM centroid fallback; current threshold 0.22 is configurable, not assumed correct.
3. Log candidate scores, thresholds, and decision path. Support multiple aspects and ambiguity.
4. Test the waiting/delay priority rule against “the waiting area was comfortable”.
5. Wrap spaCy dependency opinion extraction; return exact spans/offsets where possible.
6. Handle negation, intensifiers, coordination (“cold and bland”), implicit aspect (“cost an arm and a leg”), implicit target (“too loud”), multiple aspects, no opinion, and sarcasm/uncertainty.
7. Preserve unmatched candidates for review.
8. Keep aspect/opinion extraction separate from sentiment classification.

Test at least: “biryani excellent”, “waiter rude”, “prices high”, “music too loud”, “not fresh”, mixed polarity, implicit price, ambiguous waiting, factual/no opinion, unclear sarcasm.

## Acceptance

Each assertion has evidence and method metadata. Threshold/rules are configurable. Uncertainty is represented, not hidden.

---

# 9. Phase 4 — Sentiment Inference and Evaluation

## Tasks

1. Keep review-level DistilBERT separate from clause-level inference.
2. Verify checkpoint, tokenizer, label mapping, preprocessing, and max length.
3. Classify the clause/opinion-bearing context, not an isolated opinion word.
4. Return probabilities, label, model version, and inference status.
5. Add abstention only if threshold is validated on development data. Otherwise show scores and flag low confidence for review.
6. Preserve negation; handle empty/truncated inputs.
7. Do not describe the model as clause-validated unless evidence supports it.
8. Support batch inference and deterministic ordering.

## Evaluation

- Train/tune only on newly annotated train/dev data.
- Keep the existing 500-clause benchmark locked for final evaluation.
- Report clause-level macro-F1, per-class precision/recall/F1, confusion matrix, support, class distribution, and errors.
- Report review-level metrics separately.
- Tune thresholds on dev only.
- No improvement claim without controlled comparison on the same untouched test set.

## Acceptance

Code/UI distinguish task granularity and metrics. Every prediction is traceable to model/version/input.

---

# 10. Phase 5 — Pre-Annotation and Human Review

## Goal

Current models/rules propose labels for a **new annotation pool**. Humans accept, correct, add, or reject. Suggestions are not ground truth.

## Tasks

1. Locate/protect the 500-clause gold benchmark; block it from standard training/pre-annotation exports.
2. Sample a new pool with random examples, all aspects/classes, mixed polarity, multiple aspects, low-confidence/unmatched cases, negation, short fragments, and varied restaurants/dates.
3. Do not select only high-confidence predictions.
4. Run pipeline and export one row per proposed assertion. Include no-assertion clauses where useful for coverage analysis.
5. Keep suggested and final labels in separate fields. Consider hiding confidence in the annotator UI to reduce anchoring.
6. Annotator actions: accept, correct aspect, correct span, correct sentiment, add assertion, mark no opinion, unclear, reject.
7. Provide guidelines for negation, neutral vs no opinion, implicit aspects, mixed aspects, sarcasm, and span boundaries.
8. Double-annotate a subset; calculate agreement separately for aspect, span, and sentiment; adjudicate and preserve history.
9. Pilot 300–500 newly annotated assertion examples, review errors/guidelines, then scale.
10. Validate imports and report acceptance/correction/rejection rates.

Minimum fields: `review_id`, `clause_id`, `review_text`, `clause_text`, suggested aspect/target/opinion/sentiment/scores, final aspect/target/opinion/sentiment, status, notes, model version, segmentation version, schema version.

## Acceptance

Gold benchmark is excluded; imports validate IDs/labels/spans; suggestions are never overwritten; reports show status and label counts.

---

# 11. Phase 6 — Canonical Theme Normalization

## Tasks

1. Create a registry with stable IDs, aspect, name, definition, aliases, status, and version.
2. Start with curated/rule mappings. Embeddings may propose candidates but must not silently create irreversible mappings.
3. Preserve original target/opinion and mapping version.
4. Add review queue for new phrases, low-similarity matches, conflicting aspects, high-volume unmapped phrases, and duplicate themes.
5. Support merge/deprecation while retaining historical IDs and migration mapping.
6. Do not merge semantically similar but operationally distinct phrases (e.g. “slow service” vs “slow cooking”).
7. Preserve polarity; praise and complaint may map to the same theme.
8. Document inclusion/exclusion examples.

## Acceptance

Mappings are explainable, reversible, and human-reviewable.

---

# 12. Phase 7 — Restaurant-Level Aggregation

## Required metrics

For restaurant/aspect/theme/time window:
- distinct reviews mentioning theme;
- total eligible reviews in denominator;
- prevalence = distinct reviews mentioning theme / eligible reviews;
- assertion count (secondary, not interchangeable with prevalence);
- sentiment counts/proportions;
- distinct reviews by sentiment;
- first/last date where available;
- comparable-period trend;
- missing-date coverage and minimum-sample status.

**Prevalence uses distinct review IDs.** Repeating a phrase five times in one review does not count as five reviews.

## Tasks

1. Implement shared analytics functions; no ad hoc page-level calculations.
2. Define denominator and exclusion policies.
3. Make time windows explicit and comparable.
4. Show raw counts with percentages.
5. Configure minimum sample sizes; below threshold show “low sample”/“insufficient data”.
6. Handle zero denominators, missing dates, duplicate reviews, and multiple assertions.
7. Keep Bayesian health score secondary; expose prior/global mean, decay parameter, effective sample size, and period.
8. Add uncertainty intervals where appropriate and document method.

## Acceptance

Every displayed percentage can be reproduced from numerator/denominator. Hand-calculated fixtures pass.

---

# 13. Phase 8 — Descriptive Diagnostics, Not Causal Discovery

## Allowed diagnostics

Theme prevalence, period-over-period change, co-occurrence within reviews, sentiment distribution, sufficiently supported segment comparisons, and representative review evidence.

## Tasks

1. Derive diagnostics from validated aggregate records.
2. Store period, numerator, denominator, sample size, baseline, and supporting IDs.
3. Use language such as “appeared in X of Y reviews”, “changed from A% to B%”, or “co-occurred with theme Z”.
4. Do not say “caused”, “due to”, or “root cause” based only on reviews.
5. Guard against small samples, sparse time series, and changing review volume.
6. Caveat comparisons across different collection mechanisms.
7. Drill down to exact evidence.
8. Explain that reviews cannot verify staffing, kitchen workflow, inventory, order timing, or management causes.

Future extension points may include POS timestamps, staffing, kitchen load, refunds, and resolution logs. Do not fabricate or simulate these data in the current release.

## Acceptance

Diagnostics are descriptive, reproducible, and evidence-linked.

---

# 14. Phase 9 — Evidence-Linked Recommendations

Each suggestion must show:
1. **Observed signal:** measured issue and period.
2. **Evidence:** exact quotes, counts, review IDs.
3. **Possible driver:** hypothesis, not finding.
4. **Suggested investigation/action:** proportionate and concrete.
5. **Validation required:** operational data or manager check.
6. **Status:** proposed, investigating, accepted, rejected, completed, outcome unknown.

## Tasks

- Inspect current SOP repository and map entries to canonical themes.
- Add applicability conditions and evidence requirements.
- Treat existing complaint-to-intervention mappings as candidate suggestions, not validated prescriptions.
- Link every suggestion to theme/diagnostic and review IDs.
- Never invent cost, savings, revenue impact, or expected improvement.
- Regex fact-checking does not validate action suitability. Prefer structured metric references rendered from computed values.
- Require human approval before representing a suggestion as an operational decision.
- Label unverified drivers as “hypothesis / requires validation”.

## Acceptance

Every recommendation is traceable and distinguishes observation, hypothesis, and action. No guaranteed outcome.

---

# 15. Phase 10 — Streamlit Integration

Reuse existing pages where possible.

### Overview
Restaurant selector, date window, eligible review count/date coverage, aspect summary, low-sample warnings, health score as secondary with method disclosure.

### What customers mention
Theme prevalence, numerator/denominator, sentiment distribution, trends, filters, evidence drill-down.

### Evidence explorer
Exact review/clause text, highlighted spans, model vs human label, theme mapping/version, date/rating, source ID, processing metadata.

### Diagnostics
Descriptive patterns, explicit periods, sample warnings, “association, not causation” note, linked evidence.

### Action workspace
Observed signal, possible driver, suggested investigation, validation needed, evidence, status.

### Annotation review
Validated pre-annotation export/import, protected gold set, annotation progress and disagreement summaries.

### Existing tools
Retain batch CSV, live sandbox, benchmarking. Clearly separate review-level and clause-level metrics.

## UI rules

- Never show a percentage without numerator, denominator, and period (or accessible detail).
- Never show an untraceable quote.
- Do not rank low-volume restaurants as best/worst.
- Use “not enough data” rather than zero when denominator is absent.
- Ensure filters apply consistently.
- Summary metric → evidence drill-down in no more than two interactions.

---

# 16. Phase 11 — Testing and Quality Gates

## Unit tests
Ingestion, duplicates, segmentation offsets, aspect thresholds, ambiguous cases, opinion spans, negation, label mapping, theme versioning, distinct-review prevalence, date comparisons, minimum samples, recommendation references, and metric rendering.

## Integration tests
1. Review → validation → clauses → assertions → themes → aggregates.
2. Mixed-polarity multi-aspect review yields separate assertions.
3. Evidence matches source exactly.
4. Missing dates are not fabricated.
5. Low-volume restaurant warns.
6. Human corrections do not mutate suggestions.
7. Gold exclusion is enforced.
8. Streamlit uses shared analytics.

## Regression
Compare old/new outputs on fixed examples; document intentional differences. Do not preserve demonstrably incorrect behaviour merely to match old output.

## Quality gates
- No silent row drops.
- No causal language unsupported by data.
- No “verified” recommendation without verification.
- No metric without denominator and period.
- No gold benchmark leakage.
- Tests do not require network/model downloads unless explicitly optional integration tests.

---

# 17. Phase 12 — Documentation and Readiness

Document architecture, setup/run commands, model configuration, contracts/schema versions, annotation workflow, benchmark protection, metric definitions, denominator policy, limitations, test commands, reproducibility, theme/SOP editing, and interpretation.

Include limitations:
- Reviews are self-selected and may not represent all customers.
- Review ratings are not clause-level labels.
- Segmentation/extraction can fail on sarcasm, implicit targets, and complex syntax.
- Clause-level sentiment is weaker than review-level performance and needs separate validation.
- Complaint extraction identifies linguistic patterns, not operational causes.
- SOP mappings remain suggestions until tested.
- Health score is an experimental summary, not an industry-standard validated measure.

---

# 18. Execution Milestones

| Milestone | Scope | Exit condition |
|---|---|---|
| M0 | Reconnaissance | Map, baseline, actual file plan |
| M1 | Contracts + ingestion | Validated records and tests |
| M2 | Segmentation + assertion interface | Traceable clauses and multi-assertion output |
| M3 | Existing model/rule adapters | Stable outputs, provenance, regression tests |
| M4 | Pre-annotation | Safe export/import; gold protected |
| M5 | Theme registry | Reversible mappings |
| M6 | Aggregation | Reproducible rates and safeguards |
| M7 | Diagnostics | Descriptive, evidence-linked patterns |
| M8 | Recommendations | Hypothesis-labelled suggestions |
| M9 | Streamlit | End-to-end evidence drill-down |
| M10 | Hardening/docs | Tests, limitations, reproducible run |

At every milestone: run tests, summarize changed files, report pass/fail, list assumptions/known issues, and do not bypass failed acceptance criteria silently.

---

# 19. Antigravity Instructions

1. Start with repository inspection; do not immediately code.
2. Return repository map and baseline findings first.
3. Propose a file-by-file plan using actual paths.
4. Implement in small, reviewable milestones.
5. Read callers/tests before changing existing modules.
6. Follow current language, dependency, formatting, and test conventions.
7. Do not add a database, vector store, API server, LLM provider, or annotation platform unless architecture requires it. CSV pilot is acceptable.
8. Reuse configured local checkpoints; do not download large models/data without need.
9. Synthetic fixtures must be clearly labelled.
10. Never modify protected gold benchmark.
11. If a required data source is absent, implement a safe unavailable state and document the dependency.
12. Resolve business-definition ambiguity conservatively and document assumptions.
13. Do not claim completion from code generation alone. Run tests and demonstrate a small real, non-gold sample.
14. Final handoff must include architecture, files changed, commands, test outcomes, sample output, limitations, and owner decisions.

## Required demonstration

Run one non-gold review with at least two aspects and mixed polarity. Show original review, clauses/offsets, assertion records, predictions, theme mapping, aggregate contribution, any valid diagnostic, and recommendation with hypothesis label/evidence links. If sample size is insufficient, output “insufficient data”, not a fabricated conclusion.

---

# 20. Out of Scope

Unless separately approved:
- causal root-cause discovery from review text;
- automatic execution of operational changes;
- guaranteed ROI/impact estimates;
- replacing models without evaluation;
- training on protected gold benchmark;
- full RST parser;
- LLM diagnosis as source of truth;
- POS/staffing integration without real data;
- cloud migration or multi-tenant redesign;
- presenting health score as validated industry standard.

---

# 21. Owner Acceptance Checklist

- [ ] Actual repository architecture inspected and documented.
- [ ] Existing features run or migration documented.
- [ ] Review-level and clause-level metrics separated.
- [ ] Gold benchmark protected.
- [ ] Human labels separate from suggestions.
- [ ] Quotes exact and traceable.
- [ ] Theme mappings reversible.
- [ ] Prevalence uses distinct reviews and denominators.
- [ ] Low sample/missing dates visible.
- [ ] Diagnostics descriptive, not causal.
- [ ] Recommendations distinguish observation, hypothesis, and action.
- [ ] Every recommendation links to evidence.
- [ ] Edge-case and end-to-end tests pass.
- [ ] Limitations documented honestly.
- [ ] Non-gold end-to-end demonstration completed.

**Final product principle:** DineSense should help an operator move from customer evidence to a sensible investigation—not present model-generated interpretation as established operational truth.
