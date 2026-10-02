# DineSense AI Annotation Guidelines (Version 2.1)

**Effective Date:** 2026-10-01  
**Domain:** Restaurant Review Aspect-Based Sentiment Analysis (ABSA)  
**Schema Level:** Review → Clause → Assertion Hierarchy (with Clause Elaboration Linking)

---

## 1. Updated 5-Aspect Taxonomy

Every evaluative assertion must be assigned to exactly one of the following five core aspect categories, supported by explicit textual evidence:

| Aspect Category | Definition & Scope | Canonical Targets & Topics |
| :--- | :--- | :--- |
| **1. Food** | Taste, flavor, temperature, freshness, portion size, ingredients, dishes, beverages, and culinary preparation. | Biryani, chicken, noodles, pasta, desserts, drinks, taste, spicy, cold, oily, fresh, portion, menu. |
| **2. Service** | Staff hospitality, speed of order turnaround, attentiveness, courtesy, order accuracy, delivery, and management. | Waiters, staff, server, captain, delivery, management, polite, rude, slow, quick, wrong order. |
| **3. Price / Value** | Cost, affordability, fairness of pricing, value for money, billing transparency, discounts, and offers. | Cost, price, bill, expensive, pocket-friendly, worth it, cost-effective, value for money, overpriced. |
| **4. Ambience** | Physical dining environment, interior decor, seating comfort, background music, lighting, noise, and cleanliness. | Ambience, interior, decor, seating, tables, music, noise, AC, view, lighting, hygiene, cleanliness. |
| **5. General Experience** | Holistic impression, overall visit experience, repeat intention, or broad recommendation regarding the **restaurant as a whole**, without evaluating an isolated departmental attribute. | Restaurant, place, outlet, visit, experience, "must visit", "loved this place", "will come again", "disappointing experience". |

> **Note on `No Aspect Opinion`:** `No Aspect Opinion` is **not an aspect category**. It is an `annotation_status` assigned to clauses that are purely factual, procedural, descriptive, or conversational, containing no evaluative sentiment.

---

## 2. General Experience: Definition, Invariants & Boundary Rules

### Definition
Use **General Experience** when the reviewer expresses a subjective evaluation, reaction, or recommendation about the **establishment as a whole**, rather than evaluating food, service, price, or physical ambience.

### Core Annotation Rules
1. **Whole-Venue Evaluation Only**: The sentiment must be directed at the restaurant overall (e.g., *"Loved this place!"*, *"A must-visit restaurant in town"*).
2. **Never a Fallback for Ambiguity**: Do **not** use `General Experience` as a dumping ground for unclear sentences, missing keywords, or difficult phrases. If the aspect cannot be identified with textual evidence, flag the record as `annotation_status = 'Needs Review'` with `aspect = 'Unclear'`.
3. **Strict Separation from `No Aspect Opinion`**:
   - Descriptive statements of visit facts without evaluation (e.g., *"Visited yesterday with family"*, *"Reached at 9 PM"*) express no sentiment and must be labeled `No Aspect Opinion`.
   - General evaluations (e.g., *"Had a wonderful time here"*) express sentiment and belong in `General Experience` (`Positive`).
4. **Co-Occurrence with Specific Aspects**:
   - If a review contains both a specific departmental evaluation and an independent overall recommendation, create **separate assertion records**:
     - *"The food was amazing. Must visit!"* →
       - Assertion 1: `Food` — `Positive` (Target: `"food"`, Opinion: `"amazing"`)
       - Assertion 2: `General Experience` — `Positive` (Target: `"place"` / implicit, Opinion: `"Must visit"`)
5. **No Automatic Duplication**:
   - If a sentiment is already clearly tied to a specific aspect, do not duplicate it as `General Experience`:
     - *"The service was terrible"* → `Service` — `Negative` **only**. (Do not add a second assertion for General Experience).
6. **Typo and Informal Spelling Invariance**:
   - Common spelling mistakes, phonetic transcriptions, and colloquialisms must be accurately normalized during aspect recognition while preserving the exact verbatim string and offsets:
     - *"Worst taste"* / *"avarage taste"* → `Food`
     - *"very quick delivery.."* → `Service`
     - *"restro is nice"* → `General Experience` (or `Ambience` if referring strictly to decor)

---

## 3. Disambiguation Table: General Experience vs. Other Categories vs. No Aspect Opinion

| Example Text | Correct Category & Polarity | Target Span | Opinion Span | Why This Classification? |
| :--- | :--- | :--- | :--- | :--- |
| *"Must-visit place!"* | **General Experience** — `Positive` | `"place"` | `"Must-visit"` | Recommends the entire venue, not just the food or decor. |
| *"I’ll definitely visit again."* | **General Experience** — `Positive` | `"place"` (implicit) | `"definitely visit again"` | Evaluates repeat intention for the overall establishment. |
| *"Overall, a disappointing experience."* | **General Experience** — `Negative` | `"experience"` | `"disappointing"` | Explicit holistic summary of the visit. |
| *"Loved this place!"* | **General Experience** — `Positive` | `"place"` | `"Loved"` | Broad positive appraisal of the restaurant. |
| *"if there was minus rating, I would have given that."* | **General Experience** — `Negative` | `"place"` (implicit) | `"if there was minus rating"` | Hyperbolic holistic dissatisfaction with the establishment. |
| *"The food was delicious, but the place is a must visit."* | **Assertion 1:** `Food` — `Positive`<br>**Assertion 2:** `General Experience` — `Positive` | 1: `"food"`<br>2: `"place"` | 1: `"delicious"`<br>2: `"must visit"` | Clause 1 evaluates culinary taste; Clause 2 provides an independent overall venue recommendation. |
| *"The place has beautiful lighting and cozy sofas."* | **Ambience** — `Positive` | `"lighting"`, `"sofas"` | `"beautiful"`, `"cozy"` | Despite using the word *"place"*, the evaluation is strictly about physical decor and furnishings. |
| *"The place was noisy and crowded."* | **Ambience** — `Negative` | `"place"` | `"noisy and crowded"` | Evaluates the acoustic and spatial environment of the venue. |
| *"The service was terrible."* | **Service** — `Negative` | `"service"` | `"terrible"` | Strictly departmental complaint. Do **not** fabricate a General Experience assertion. |
| *"Visited yesterday at 8 PM with my colleagues."* | **No Aspect Opinion** | `""` | `""` | Purely factual visit context; contains no evaluative opinion. |
| *"Do follow us on Instagram: forkandspoonstoryhyd"* | **No Aspect Opinion** | `""` | `""` | Promotional boilerplate; contains zero sentiment. |
| *"We ordered two plates of chicken biryani."* | **No Aspect Opinion** | `""` | `""` | Procedural order statement without taste or quality appraisal. |

---

## 4. Typo, Slang & Colloquialism Handling

Reviewers frequently use phonetic spellings, regional terms, and typos. Annotations must accurately identify the intended aspect and sentiment while strictly extracting the verbatim substring:

1. **Food Typos**:
   - `"Worst taste"` / `"avarage taste"` / `"below average taste"` → `Food` (`Taste / Flavor`), extracting exact spans like `"avarage"`.
   - `"food was bad..and very less in quantity"` → `Food` (`Portion size`), Opinion: `"very less in quantity"`.
   - `"not cooked properly, no paneer in paratha"` → `Food` (`Food quality`), Opinion: `"not cooked properly"`.
2. **Service Typos & Colloquialisms**:
   - `"very quick delivery.."` → `Service` (`Service speed / Wait time`), Target: `"delivery"`, Opinion: `"very quick"`.
   - `"services is good bikash shoo"` → `Service` (`Staff courtesy`), Target: `"services"`, Opinion: `"good"`.
3. **Ambience & General Typos**:
   - `"restro is designed very well"` → `Ambience` (`Interior / Decor`), Target: `"restro"`, Opinion: `"designed very well"`.
   - `"toooo good so as beer and vodka"` → `Food` (`Drink quality`), Opinion: `"toooo good"`.
