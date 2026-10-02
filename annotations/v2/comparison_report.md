# DineSense AI: V1 vs V2 Annotation Comparison Report (25-Review Test Batch)

**Evaluation Date:** 2026-09-30  
**Scope:** 25-Review Test Batch (Focused on Negation, Contrast, Descriptive Clauses, and Grounded Aspects)  
**V1 Model:** `gemini-3.8-flash-preannotator-v1.0` | **V2 Model:** `gemini-3.8-flash-preannotator-v2.0`  

---

## 1. Aggregate Metric Comparison

| Metric | V1 Pilot Annotations | V2 Revised Annotations | Variance / Impact |
| :--- | :---: | :---: | :--- |
| **Total Assertions** | 228 | 212 | Refined clause segmentation & multi-assertion logic |
| **`Annotated` Assertions** | 117 (51.3%) | 87 (41.0%) | Pruned spurious opinions from descriptive text |
| **`No Aspect Opinion`** | 110 (48.2%) | 125 (59.0%) | Correctly identifies factual context & dish lists |
| **Aspect: `Other`** | 24 | 0 | Eliminated `Other` as default dumping ground |
| **Aspect: `General / Whole Establishment`** | 0 (Not in v1) | 22 | Captures holistic recommendations (`MUST TRY PLACE`, etc.) |
| **Hallucinated AC Theme** | 7 | 0 | **Completely eliminated false AC assignments** |

---

## 2. Key Error Corrections Walkthrough

### Fix A: Negation & Compound Opinion Spans
In V1, isolated positive tokens like `good` or `great` were extracted without their negation particle, flipping sentiment to Positive.

| Review ID & Establishment | Clause Text | V1 Annotation | V2 Corrected Annotation |
| :--- | :--- | :--- | :--- |
| `REV_01800` (10 Downing Street) | `"...what dressing giving i Don't no but this is not good very bad"` | Target: `dressing`, Op: `not good`, Sent: `Positive` *(Token match bug)* | Target: `dressing`, Op: `not good very bad`, Sent: `Negative` |
| `REV_05070` (Biryanis And More) | `"not that great"` | Op: `great`, Sent: `Positive`, Aspect: `Other` | Op: `not that great`, Sent: `Negative`, Aspect: `Food / Dining` |
| `REV_01284` (Absolute Sizzlers) | `"It's named sizzler but has nothing close to sizzler"` | Op: `close`, Sent: `Positive` | Target: `sizzler`, Op: `nothing close to`, Sent: `Negative` |
| `REV_04412` (Owm Nom Nom) | `"Chicken kadai curry was not tasty at all"` | Op: `tasty`, Sent: `Positive` | Target: `Chicken kadai curry`, Op: `not tasty at all`, Sent: `Negative` |

### Fix B: Opinion Span Selection vs Descriptive Food Attributes
In V1, culinary attributes like `crispy` or `fine-dine` were tagged as opinion expressions even when naming dishes or styles.

| Review ID & Establishment | Clause Text | V1 Annotation | V2 Corrected Annotation |
| :--- | :--- | :--- | :--- |
| `REV_05070` (Biryanis And More) | `"...served with crispy fried noodles,it tasted delicious"` | Op: `crispy`, Sent: `Positive` *(Ignored explicit opinion)* | Target: `noodles`, Op: `delicious`, Sent: `Positive`. *(Identifies `crispy fried` as preparation format)* |
| `REV_05070` (Biryanis And More) | `"Crispy veg and corn 65 were other vegetarian starters which we had"` | Op: `Crispy`, Target: `veg`, Sent: `Positive` | Status: `No Aspect Opinion` (`descriptive_context`). `Crispy veg` is dish title. |

### Fix C: Context-Sensitive Word Disambiguation (`fine`)
| Review ID & Establishment | Clause Text | V1 Annotation | V2 Corrected Annotation |
| :--- | :--- | :--- | :--- |
| `REV_05070` (Biryanis And More) | `"...is fine-dine restaurant"` | Op: `fine`, Sent: `Neutral` | Status: `No Aspect Opinion`. `fine-dine` is business format description. |
| `REV_02912` (Hunger Maggi Point) | `"So one fine night, ordered for a Corn Masala Cheese Butter Maggi through Zomato"` | Op: `fine`, Sent: `Neutral`, Aspect: `Other` | Status: `No Aspect Opinion`. Idiomatic time expression. |

### Fix D: Grounded Aspect & Theme Evidence (Elimination of False AC/Facilities)
| Review ID & Establishment | Clause Text | V1 Annotation | V2 Corrected Annotation |
| :--- | :--- | :--- | :--- |
| `REV_06534` (Yum Yum Tree) | `"MUST TRY PLACE"` | Aspect: `Facilities / Amenities`, Theme: `Air conditioning / Ventilation` *(Hallucinated)* | Aspect: `General / Whole Establishment`, Theme: `Establishment recommendation`, Target: `PLACE` |
| `REV_08836` (Cascade - Radisson) | `"The place is amazing"` | Aspect: `Facilities / Amenities`, Theme: `Air conditioning / Ventilation` *(Hallucinated)* | Aspect: `General / Whole Establishment`, Theme: `Overall experience`, Target: `place` |
| `REV_08836` (Cascade - Radisson) | `"A must try in that area"` | Aspect: `Location`, Theme: `Accessibility / Ease of finding` | Aspect: `General / Whole Establishment`, Theme: `Establishment recommendation` |

### Fix E: Elaboration & Descriptive Clause Linking
V2 introduces `clause_relation` and `elaboration_of` to link subordinate clauses that provide context or lists for prior assertions without duplicating sentiment.
- In `REV_05070`, menu items listed under starters (`Chicken manchow soup`, `Veg lemon coriander soup`, `Qubani ka meetha`) are marked as `descriptive_context` and linked to the parent assertion rather than generating spurious opinions.

---

## 3. Sample Composition of the 25-Review Test Batch

- **Total Unique Establishments:** 25 (1 review per establishment)
- **Rating Distribution:** 12 High (4.0–5.0), 5 Mid (3.0–3.5), 8 Low (1.0–2.5)
- **Length Distribution:** 7 Short (<150), 10 Medium (150–350), 8 Long (>350)
- **Phenomena Tested:** Negation phrases, contrastive splits, dish list elaborations, temporal idioms, whole-venue recommendations, implicit targets, delivery service.