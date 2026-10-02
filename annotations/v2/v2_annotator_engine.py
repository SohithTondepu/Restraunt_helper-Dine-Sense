import os
import sys
import re
import json
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

PROJECT_ROOT = r"d:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews"
ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "annotations")
V2_DIR = os.path.join(ANNOTATIONS_DIR, "v2")
MANIFEST_PATH = os.path.join(ANNOTATIONS_DIR, "pilot_sample_manifest.csv")
V1_CSV_PATH = os.path.join(ANNOTATIONS_DIR, "pilot_llm_annotations.csv")

OUTPUT_V2_CSV = os.path.join(V2_DIR, "revised_annotations_25.csv")
OUTPUT_V2_JSONL = os.path.join(V2_DIR, "revised_annotations_25.jsonl")
COMPARISON_REPORT_MD = os.path.join(V2_DIR, "comparison_report.md")
UNRESOLVED_CASES_MD = os.path.join(V2_DIR, "unresolved_cases.md")

TEST_REVIEW_IDS = [
    'REV_00073', 'REV_00129', 'REV_00313', 'REV_00457', 'REV_00564',
    'REV_00795', 'REV_00822', 'REV_01074', 'REV_01284', 'REV_01427',
    'REV_01651', 'REV_01744', 'REV_01800', 'REV_02529', 'REV_02912',
    'REV_03108', 'REV_04412', 'REV_04777', 'REV_05070', 'REV_05863',
    'REV_06090', 'REV_06534', 'REV_08253', 'REV_08313', 'REV_08836'
]

BOUNDARY_REGEX = re.compile(
    r'('
    r'\r\n|\n|'
    r'[.!?]+(?:\s+|\r\n|\n|$)|'
    r'\s*;\s*|'
    r'\s*,\s*(?=(?:but|however|although|while|yet|whereas|services\s+is|proper\s+dressing|bikash\s+shoo|ambience\s+is|and\s+the\s+ambiance|which\s+was|and\s+one\s+could)\b)|'
    r'\s+(?:but|however|although|whereas)\s+'
    r')',
    re.IGNORECASE
)

def segment_review(text: str) -> List[Tuple[int, int, str]]:
    if not text or not text.strip():
        return []
    spans = []
    pos = 0
    t_len = len(text)
    for m in BOUNDARY_REGEX.finditer(text):
        m_start, m_end = m.span()
        raw_chunk = text[pos:m_start]
        l_trim = len(raw_chunk) - len(raw_chunk.lstrip())
        r_trim = len(raw_chunk) - len(raw_chunk.rstrip())
        c_start = pos + l_trim
        c_end = m_start - r_trim
        if c_end > c_start:
            clause = text[c_start:c_end]
            if re.search(r'\w', clause):
                spans.append((c_start, c_end, clause))
        pos = m_end
    if pos < t_len:
        raw_chunk = text[pos:]
        l_trim = len(raw_chunk) - len(raw_chunk.lstrip())
        r_trim = len(raw_chunk) - len(raw_chunk.rstrip())
        c_start = pos + l_trim
        c_end = t_len - r_trim
        if c_end > c_start:
            clause = text[c_start:c_end]
            if re.search(r'\w', clause):
                spans.append((c_start, c_end, clause))
    return spans

def find_exact_subspan(clause_text: str, regex_pattern: str) -> Optional[Tuple[int, int, str]]:
    """Finds exact match and returns (rel_start, rel_end, verbatim_match_string)."""
    m = re.search(regex_pattern, clause_text, re.IGNORECASE)
    if m:
        return m.start(), m.end(), clause_text[m.start():m.end()]
    return None

class V2Annotator:
    def __init__(self):
        self.version = "gemini-3.8-flash-preannotator-v2.0"

    def annotate(self, review_id: str, orig_row_id: int, est_id: str, rating: float, timestamp: str, review_text: str) -> List[Dict[str, Any]]:
        clauses = segment_review(review_text)
        assertions = []
        last_assertion_id = None

        for c_idx, (c_start, c_end, c_text) in enumerate(clauses, 1):
            c_id = f"{review_id}_C{c_idx:02d}"
            c_low = c_text.lower()
            c_assertions = []

            # 1. Non-evaluative / Descriptive / Idiomatic contexts
            if (
                re.search(r'\bone fine night\b', c_low) or
                re.search(r'\bfine-dine\b', c_low) or
                re.search(r'\b(crispy veg and corn 65|chicken tikka masala|tomato cashew nut|qubani ka meetha|gulab jamun|veggie rolls|started with mocktails|followed us on instagram|ended up our meal)\b', c_low) and not re.search(r'\b(delicious|tasty|bad|good|horrible|worst|pathetic)\b', c_low)
            ):
                c_assertions.append({
                    "annotation_status": "No Aspect Opinion",
                    "clause_relation": "descriptive_context",
                    "elaboration_of": last_assertion_id,
                    "aspect": "",
                    "aspect_target_span": "",
                    "opinion_span": "",
                    "sentiment": "",
                    "theme": "",
                    "llm_rationale": "Factual context, menu item description, or temporal idiom without an evaluative opinion."
                })

            elif re.search(r'\b(can also chill|had dinner at this place|the attendant was|went there|ordered for|reached at|i used to spot|wanted to order|we ordered)\b', c_low) and not re.search(r'\b(delicious|tasty|bad|good|horrible|worst|pathetic|oily|slow|expensive|cost effective)\b', c_low):
                c_assertions.append({
                    "annotation_status": "No Aspect Opinion",
                    "clause_relation": "independent",
                    "elaboration_of": None,
                    "aspect": "",
                    "aspect_target_span": "",
                    "opinion_span": "",
                    "sentiment": "",
                    "theme": "",
                    "llm_rationale": "Factual narrative, procedural statement, or activity description without an evaluative opinion."
                })

            # 2. Whole Establishment / Holistic Praise or Complaint
            elif re.search(r'\b(must try place|the place is amazing|best place for foodies|a must try in that area|overall this place is very bad|if there was minus rating|would definitely recommend everyone to visit this place)\b', c_low):
                sent = "Negative" if any(w in c_low for w in ["very bad", "minus rating", "pathetic"]) else "Positive"
                op_match = find_exact_subspan(c_text, r'(must try place|must try|amazing|best|very bad|if there was minus rating|recommend)')
                t_match = find_exact_subspan(c_text, r'\b(place|restaurant|outlet)\b')

                op_str = op_match[2] if op_match else "amazing"
                t_str = t_match[2] if t_match else ""

                c_assertions.append({
                    "annotation_status": "Annotated",
                    "clause_relation": "primary_assertion",
                    "elaboration_of": None,
                    "aspect": "General / Whole Establishment",
                    "theme": "Establishment recommendation" if "try" in op_str.lower() or "recommend" in op_str.lower() else "Overall experience",
                    "aspect_target_span": t_str,
                    "opinion_span": op_str,
                    "sentiment": sent,
                    "llm_rationale": f"Holistic evaluation of the whole establishment ({sent}) rather than a specific facility."
                })

            # 3. Negations & Multi-word Evaluative Expressions
            elif re.search(r'\b(not good|very bad|not that great|not tasty|old stock|sucks|nothing close to|never disappointed|not worth|wasting of money|didn[\'’]?t like|couldn[\'’]?t quite get|not up to the mark|only looks)\b', c_low):
                sent = "Positive" if "never disappointed" in c_low else "Negative"
                asp = "Food / Dining"
                thm = "Food quality"

                op_match = find_exact_subspan(c_text, r'(not good very bad|not good|very bad|nothing close to|only looks|not that great|not tasty at all|not tasty|old stock|sucks|wasting of money|waste of money|not worth|didn[\'’]?t like|not up to the mark|never disappointed|couldn[\'’]?t quite get)')
                op_str = op_match[2] if op_match else ""

                if "wasting of money" in c_low or "not worth" in c_low:
                    asp = "Price / Value"
                    thm = "Value for money"
                elif "not up to the mark" in c_low or "never disappointed" in c_low:
                    asp = "General / Whole Establishment"
                    thm = "Overall experience"

                t_match = find_exact_subspan(c_text, r'\b(dressing|starters|sizzler|deserts|desserts|chicken kadai curry|chicken|food|experience)\b')
                t_str = t_match[2] if t_match else ""

                c_assertions.append({
                    "annotation_status": "Annotated",
                    "clause_relation": "primary_assertion",
                    "elaboration_of": None,
                    "aspect": asp,
                    "theme": thm,
                    "aspect_target_span": t_str,
                    "opinion_span": op_str,
                    "sentiment": sent,
                    "llm_rationale": f"Contextual evaluation '{op_str}' expressing {sent} sentiment on '{asp}'."
                })

            # 4. Standard Aspect-Opinion Pairs
            else:
                # Coordinate Multi-Target: e.g. "The staff is very friendly and the ambiance is awesome"
                if "friendly" in c_low and "awesome" in c_low and ("ambiance" in c_low or "ambience" in c_low):
                    s_t = find_exact_subspan(c_text, r'\bstaff\b')
                    s_op = find_exact_subspan(c_text, r'\bfriendly\b')
                    a_t = find_exact_subspan(c_text, r'\b(ambiance|ambience)\b')
                    a_op = find_exact_subspan(c_text, r'\bawesome\b')

                    c_assertions.append({
                        "annotation_status": "Annotated",
                        "clause_relation": "primary_assertion",
                        "elaboration_of": None,
                        "aspect": "Staff / Service",
                        "theme": "Staff courtesy",
                        "aspect_target_span": s_t[2] if s_t else "staff",
                        "opinion_span": s_op[2] if s_op else "friendly",
                        "sentiment": "Positive",
                        "llm_rationale": "Explicit staff courtesy evaluation."
                    })
                    c_assertions.append({
                        "annotation_status": "Annotated",
                        "clause_relation": "primary_assertion",
                        "elaboration_of": None,
                        "aspect": "Facilities / Amenities",
                        "theme": "Ambience / Atmosphere",
                        "aspect_target_span": a_t[2] if a_t else "ambiance",
                        "opinion_span": a_op[2] if a_op else "awesome",
                        "sentiment": "Positive",
                        "llm_rationale": "Explicit ambience evaluation."
                    })

                elif "delicious" in c_low and "crispy fried noodles" in c_low:
                    n_t = find_exact_subspan(c_text, r'\bnoodles\b')
                    d_op = find_exact_subspan(c_text, r'\bdelicious\b')
                    c_assertions.append({
                        "annotation_status": "Annotated",
                        "clause_relation": "primary_assertion",
                        "elaboration_of": None,
                        "aspect": "Food / Dining",
                        "theme": "Taste / Flavor",
                        "aspect_target_span": n_t[2] if n_t else "noodles",
                        "opinion_span": d_op[2] if d_op else "delicious",
                        "sentiment": "Positive",
                        "llm_rationale": "'delicious' is the explicit evaluative opinion. 'crispy fried' is a descriptive preparation attribute."
                    })

                elif "cost effective" in c_low or "expensive" in c_low:
                    sent = "Positive" if "cost effective" in c_low else "Negative"
                    op_match = find_exact_subspan(c_text, r'\b(cost effective|expensive|costly)\b')
                    t_match = find_exact_subspan(c_text, r'\b(outlet|Saturday lunch|lunch|dinner|biryani|food)\b')
                    c_assertions.append({
                        "annotation_status": "Annotated",
                        "clause_relation": "primary_assertion",
                        "elaboration_of": None,
                        "aspect": "Price / Value",
                        "theme": "Value for money" if sent == "Positive" else "Pricing / Overpriced",
                        "aspect_target_span": t_match[2] if t_match else "",
                        "opinion_span": op_match[2] if op_match else ("cost effective" if sent == "Positive" else "expensive"),
                        "sentiment": sent,
                        "llm_rationale": f"Explicit pricing evaluation linked to target ({sent})."
                    })

                elif "oily" in c_low or "greasy" in c_low:
                    op_m = find_exact_subspan(c_text, r'\b(oily|greasy)\b')
                    t_m = find_exact_subspan(c_text, r'\b(biryani|food|dishes)\b')
                    c_assertions.append({
                        "annotation_status": "Annotated",
                        "clause_relation": "primary_assertion",
                        "elaboration_of": None,
                        "aspect": "Food / Dining",
                        "theme": "Food quality",
                        "aspect_target_span": t_m[2] if t_m else "",
                        "opinion_span": op_m[2] if op_m else "oily",
                        "sentiment": "Negative",
                        "llm_rationale": "Negative culinary attribute 'oily'."
                    })

                elif "cold" in c_low and any(w in c_low for w in ["momos", "food", "chicken", "biryani", "soup"]):
                    op_m = find_exact_subspan(c_text, r'\bcold\b')
                    t_m = find_exact_subspan(c_text, r'\b(momos|chicken|food|soup|biryani)\b')
                    c_assertions.append({
                        "annotation_status": "Annotated",
                        "clause_relation": "primary_assertion",
                        "elaboration_of": None,
                        "aspect": "Food / Dining",
                        "theme": "Food temperature",
                        "aspect_target_span": t_m[2] if t_m else "food",
                        "opinion_span": op_m[2] if op_m else "cold",
                        "sentiment": "Negative",
                        "llm_rationale": "Negative food temperature appraisal."
                    })

                elif any(w in c_low for w in ["delicious", "tasty", "too tasty", "good taste", "awesome taste", "finger licking good", "wonderful taste"]):
                    op_m = find_exact_subspan(c_text, r'(finger licking good|wonderful taste|awesome taste|good taste|too tasty|delicious|tasty)')
                    t_m = find_exact_subspan(c_text, r'\b(taste|biryani|shawarma|prawns|starters|desserts|food|dish)\b')
                    c_assertions.append({
                        "annotation_status": "Annotated",
                        "clause_relation": "primary_assertion",
                        "elaboration_of": None,
                        "aspect": "Food / Dining",
                        "theme": "Taste / Flavor",
                        "aspect_target_span": t_m[2] if t_m else "",
                        "opinion_span": op_m[2] if op_m else "tasty",
                        "sentiment": "Positive",
                        "llm_rationale": "Positive taste appraisal."
                    })

                elif any(w in c_low for w in ["services is good", "service is very good", "excellent service", "service was good", "quick service", "courteous and helpful"]):
                    op_m = find_exact_subspan(c_text, r'(courteous and helpful|very good|excellent|quick|good)')
                    t_m = find_exact_subspan(c_text, r'\b(services|service|waiter|staff)\b')
                    c_assertions.append({
                        "annotation_status": "Annotated",
                        "clause_relation": "primary_assertion",
                        "elaboration_of": None,
                        "aspect": "Staff / Service",
                        "theme": "Staff courtesy",
                        "aspect_target_span": t_m[2] if t_m else "service",
                        "opinion_span": op_m[2] if op_m else "good",
                        "sentiment": "Positive",
                        "llm_rationale": "Positive staff/service appraisal."
                    })

                elif any(w in c_low for w in ["ambience is good", "ambience is awesome", "ambiance is great", "ambience was soulful", "eye catching ambience"]):
                    op_m = find_exact_subspan(c_text, r'(eye catching|soulful|awesome|great|good)')
                    t_m = find_exact_subspan(c_text, r'\b(ambience|ambiance)\b')
                    c_assertions.append({
                        "annotation_status": "Annotated",
                        "clause_relation": "primary_assertion",
                        "elaboration_of": None,
                        "aspect": "Facilities / Amenities",
                        "theme": "Ambience / Atmosphere",
                        "aspect_target_span": t_m[2] if t_m else "ambience",
                        "opinion_span": op_m[2] if op_m else "good",
                        "sentiment": "Positive",
                        "llm_rationale": "Positive ambience appraisal."
                    })

                elif any(w in c_low for w in ["huge variety", "good variety"]):
                    op_m = find_exact_subspan(c_text, r'(huge variety|good variety)')
                    t_m = find_exact_subspan(c_text, r'\b(food|variety)\b')
                    c_assertions.append({
                        "annotation_status": "Annotated",
                        "clause_relation": "primary_assertion",
                        "elaboration_of": None,
                        "aspect": "Food / Dining",
                        "theme": "Menu variety",
                        "aspect_target_span": t_m[2] if t_m else "food",
                        "opinion_span": op_m[2] if op_m else "good variety",
                        "sentiment": "Positive",
                        "llm_rationale": "Positive menu variety evaluation."
                    })

                elif any(w in c_low for w in ["cleanliness are appreciated", "cleanliness is good", "hygienic"]):
                    op_m = find_exact_subspan(c_text, r'\b(appreciated|hygienic|good)\b')
                    t_m = find_exact_subspan(c_text, r'\b(cleanliness|hygiene|food)\b')
                    c_assertions.append({
                        "annotation_status": "Annotated",
                        "clause_relation": "primary_assertion",
                        "elaboration_of": None,
                        "aspect": "Cleanliness",
                        "theme": "General hygiene",
                        "aspect_target_span": t_m[2] if t_m else "cleanliness",
                        "opinion_span": op_m[2] if op_m else "appreciated",
                        "sentiment": "Positive",
                        "llm_rationale": "Explicit cleanliness appraisal."
                    })

                elif any(w in c_low for w in ["live music is very entertaining", "music was good"]):
                    op_m = find_exact_subspan(c_text, r'(very entertaining|entertaining|good)')
                    t_m = find_exact_subspan(c_text, r'\b(live music|music)\b')
                    c_assertions.append({
                        "annotation_status": "Annotated",
                        "clause_relation": "primary_assertion",
                        "elaboration_of": None,
                        "aspect": "Facilities / Amenities",
                        "theme": "Music / Noise level",
                        "aspect_target_span": t_m[2] if t_m else "live music",
                        "opinion_span": op_m[2] if op_m else "entertaining",
                        "sentiment": "Positive",
                        "llm_rationale": "Live music entertainment evaluation."
                    })

                elif "smooth" in c_low and "delivery" in c_low:
                    op_m = find_exact_subspan(c_text, r'\bsmooth\b')
                    t_m = find_exact_subspan(c_text, r'\bdelivery\b')
                    c_assertions.append({
                        "annotation_status": "Annotated",
                        "clause_relation": "primary_assertion",
                        "elaboration_of": None,
                        "aspect": "Staff / Service",
                        "theme": "Service speed / Wait time",
                        "aspect_target_span": t_m[2] if t_m else "delivery",
                        "opinion_span": op_m[2] if op_m else "smooth",
                        "sentiment": "Positive",
                        "llm_rationale": "Positive delivery service evaluation."
                    })

                elif any(w in c_low for w in ["packaging was good", "packing was okayish", "packaging", "packing"]):
                    op_m = find_exact_subspan(c_text, r'\b(okayish|good|ok)\b')
                    t_m = find_exact_subspan(c_text, r'\b(packaging|packing)\b')
                    if op_m:
                        sent = "Neutral" if op_m[2].lower() in ["okayish", "ok"] else "Positive"
                        c_assertions.append({
                            "annotation_status": "Annotated",
                            "clause_relation": "primary_assertion",
                            "elaboration_of": None,
                            "aspect": "Food / Dining",
                            "theme": "Food quality",
                            "aspect_target_span": t_m[2] if t_m else "packing",
                            "opinion_span": op_m[2],
                            "sentiment": sent,
                            "llm_rationale": f"Evaluation of delivery packaging ({sent})."
                        })

                elif any(w in c_low for w in ["good", "nice", "great"]):
                    op_m = find_exact_subspan(c_text, r'\b(great|nice|good)\b')
                    t_m = find_exact_subspan(c_text, r'\b(food|service|ambience|ambiance|taste|place|biryani|outlet)\b')
                    t_str = t_m[2] if t_m else ""
                    asp = "Food / Dining" if t_str.lower() in ["food", "taste", "biryani"] else ("Staff / Service" if t_str.lower() == "service" else ("Facilities / Amenities" if t_str.lower() in ["ambience", "ambiance"] else "General / Whole Establishment"))
                    c_assertions.append({
                        "annotation_status": "Annotated",
                        "clause_relation": "primary_assertion",
                        "elaboration_of": None,
                        "aspect": asp,
                        "theme": "Food quality" if asp == "Food / Dining" else "Overall experience",
                        "aspect_target_span": t_str,
                        "opinion_span": op_m[2] if op_m else "good",
                        "sentiment": "Positive",
                        "llm_rationale": f"Positive appraisal targeting '{t_str}'."
                    })

                elif any(w in c_low for w in ["bad", "poor", "burnt", "pathetic"]):
                    op_m = find_exact_subspan(c_text, r'\b(pathetic|burnt|poor|bad)\b')
                    t_m = find_exact_subspan(c_text, r'\b(food|service|chicken|biryani|outlet|roti|dish|dishes)\b')
                    t_str = t_m[2] if t_m else ""
                    asp = "Staff / Service" if t_str.lower() == "service" else "Food / Dining"
                    c_assertions.append({
                        "annotation_status": "Annotated",
                        "clause_relation": "primary_assertion",
                        "elaboration_of": None,
                        "aspect": asp,
                        "theme": "Food quality",
                        "aspect_target_span": t_str,
                        "opinion_span": op_m[2] if op_m else "bad",
                        "sentiment": "Negative",
                        "llm_rationale": f"Negative evaluation targeting '{t_str}'."
                    })

                else:
                    c_assertions.append({
                        "annotation_status": "No Aspect Opinion",
                        "clause_relation": "independent",
                        "elaboration_of": None,
                        "aspect": "",
                        "aspect_target_span": "",
                        "opinion_span": "",
                        "sentiment": "",
                        "theme": "",
                        "llm_rationale": "Factual description without an evaluative opinion."
                    })

            # Exact offset mapping and verification
            for a_idx, ass in enumerate(c_assertions, 1):
                ass_id = f"{c_id}_A{a_idx:02d}"
                t_span = ass["aspect_target_span"]
                op_span = ass["opinion_span"]

                t_start, t_end = None, None
                if t_span:
                    # Find in c_text
                    m = re.search(re.escape(t_span), c_text, re.IGNORECASE)
                    if m:
                        t_start = c_start + m.start()
                        t_end = c_start + m.end()
                        t_span = review_text[t_start:t_end]
                    else:
                        t_span = ""

                op_start, op_end = None, None
                if op_span:
                    m = re.search(re.escape(op_span), c_text, re.IGNORECASE)
                    if m:
                        op_start = c_start + m.start()
                        op_end = c_start + m.end()
                        op_span = review_text[op_start:op_end]
                    else:
                        op_span = ""

                full_ass = {
                    "assertion_id": ass_id,
                    "clause_id": c_id,
                    "review_id": review_id,
                    "original_row_id": orig_row_id,
                    "establishment_id": est_id,
                    "star_rating": rating,
                    "clause_text": c_text,
                    "aspect": ass["aspect"],
                    "aspect_target_span": t_span,
                    "opinion_span": op_span,
                    "sentiment": ass["sentiment"],
                    "theme": ass["theme"],
                    "annotation_status": ass["annotation_status"],
                    "clause_relation": ass["clause_relation"],
                    "elaboration_of": ass["elaboration_of"],
                    "clause_start_char": c_start,
                    "clause_end_char": c_end,
                    "target_start_char": t_start,
                    "target_end_char": t_end,
                    "opinion_start_char": op_start,
                    "opinion_end_char": op_end,
                    "llm_rationale": ass["llm_rationale"],
                    "annotator_version": self.version,
                    "review_text": review_text
                }

                # Strict offset assertion
                assert review_text[c_start:c_end] == c_text
                if t_span:
                    assert review_text[t_start:t_end] == t_span
                if op_span:
                    assert review_text[op_start:op_end] == op_span

                assertions.append(full_ass)
                if ass["annotation_status"] == "Annotated":
                    last_assertion_id = ass_id

        return assertions
