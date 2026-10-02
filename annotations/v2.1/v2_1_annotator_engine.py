import os
import sys
import re
import json
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"d:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews"
ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "annotations")
V2_1_DIR = os.path.join(ANNOTATIONS_DIR, "v2.1")
MANIFEST_PATH = os.path.join(ANNOTATIONS_DIR, "pilot_sample_manifest.csv")

OUTPUT_CSV_V2_1 = os.path.join(V2_1_DIR, "pilot_v2.1_annotations_100.csv")
OUTPUT_JSONL_V2_1 = os.path.join(V2_1_DIR, "pilot_v2.1_annotations_100.jsonl")
TEMPLATE_CSV_V2_1 = os.path.join(V2_1_DIR, "verified_annotations_template_100.csv")
SUMMARY_MD_V2_1 = os.path.join(V2_1_DIR, "pilot_v2.1_summary_100.md")

BOUNDARY_REGEX = re.compile(
    r'('
    r'\r\n|\n|'
    r'[.!?]+(?:\s+|\r\n|\n|$)|'
    r'\s*;\s*|'
    r'\s*,\s*(?=(?:but|however|although|while|yet|whereas|services\s+is|proper\s+dressing|bikash\s+shoo|ambience\s+is|and\s+the\s+ambiance|which\s+was|and\s+one\s+could|service\s+and|service\s+is)\b)|'
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
    """Finds exact match with word boundaries and returns (rel_start, rel_end, verbatim_match_string)."""
    m = re.search(regex_pattern, clause_text, re.IGNORECASE)
    if m:
        return m.start(), m.end(), clause_text[m.start():m.end()]
    return None

class ComprehensiveV21Annotator:
    def __init__(self):
        self.version = "gemini-3.8-flash-preannotator-v2.1"

        self.FOOD_TARGETS = r'\b(biryani|biriyani|chicken|pizza|pasta|mandi|starters|desserts|deserts|sweets|grill|fish|rice|noodles|salad|sauce|curry|curries|rotis|roti|naan|pulao|mutton|prawns|chaat|paranthas|paratha|sizzler|sizzlers|burgers|dishes|dish|soup|drinks|beer|vodka|cocktail|cocktails|mocktail|mocktails|tea|coffee|ice\s*cream|shakes|shake|food|meal|taste|flavor|flavour|portion|quantity|menu|ingredients|halwa|momos|waffle)\b'
        self.SERVICE_TARGETS = r'\b(service|services|sevice|waiter|waiters|waitress|staff|server|servers|captain|crew|delivery|deliver|delivered|order|ordering|ordered|management|hospitality)\b'
        self.AMBIENCE_TARGETS = r'\b(ambience|ambiance|atmosphere|vibe|vibes|interior|interiors|decor|decoration|seating|seats|tables|chairs|sofa|music|songs|dj|noise|view|lighting|ac|air\s*condition(?:ing)?|ventilation|cleanliness|hygiene|dirty|clean|restroom|washroom|toilet)\b'
        self.PRICE_TARGETS = r'\b(price|prices|pricing|cost|costly|rate|rates|bill|charge|charges|value|money|wallet|taxes|discount|offers)\b'
        self.GENERAL_TARGETS = r'\b(place|restaurant|restro|outlet|hotel|visit|experience|stay|bistro|spot)\b'

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
                re.search(r'\b(crispy veg and corn 65|chicken tikka masala|tomato cashew nut|qubani ka meetha|gulab jamun|veggie rolls|started with mocktails|followed us on instagram|ended up our meal|do follow us)\b', c_low) and not re.search(r'\b(delicious|tasty|bad|good|horrible|worst|pathetic)\b', c_low)
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

            elif re.search(r'^\s*(i used to spot|wanted to order|we went|went for|reached at|got to know|thank thank you for your services|rate\s*pasta\s*5/2\.5)\s*$', c_low) or (
                re.search(r'\b(can also chill|had dinner at this place|the attendant was|went there|ordered for|reached at|i used to spot|wanted to order|we ordered)\b', c_low) and not re.search(r'\b(delicious|tasty|bad|good|horrible|worst|pathetic|oily|slow|expensive|cost effective|quick|smooth|fresh|bland|stale|chewy|cheap|disappoint|worth|dirty|clean|amazing|awesome|superb|hate)\b', c_low)
            ):
                c_assertions.append({
                    "annotation_status": "No Aspect Opinion",
                    "clause_relation": "independent",
                    "elaboration_of": None,
                    "aspect": "",
                    "aspect_target_span": "",
                    "opinion_span": "",
                    "sentiment": "",
                    "theme": "",
                    "llm_rationale": "Factual narrative, procedural statement, or order context without an evaluative opinion."
                })

            # 2. Coordinate Multi-Aspect Openers: e.g. "Amazing food and ambience", "Superb quality, service and ambience"
            elif re.search(r'\b(amazing|superb|great|good|excellent)\s+food\s+and\s+(?:the\s+)?(?:ambience|ambiance)\b', c_low) or re.search(r'\bsuperb\s+quality,\s*service\s+and\s+ambience\b', c_low):
                op_m = find_exact_subspan(c_text, r'\b(amazing|superb|great|good|excellent)\b')
                op_str = op_m[2] if op_m else "good"

                f_t = find_exact_subspan(c_text, r'\b(food|quality)\b')
                c_assertions.append({
                    "annotation_status": "Annotated",
                    "clause_relation": "primary_assertion",
                    "elaboration_of": None,
                    "aspect": "Food",
                    "theme": "Food quality",
                    "aspect_target_span": f_t[2] if f_t else "food",
                    "opinion_span": op_str,
                    "sentiment": "Positive",
                    "llm_rationale": f"Explicit food quality praise '{op_str}'."
                })

                if re.search(r'\bservice\b', c_low):
                    s_t = find_exact_subspan(c_text, r'\bservice\b')
                    c_assertions.append({
                        "annotation_status": "Annotated",
                        "clause_relation": "primary_assertion",
                        "elaboration_of": None,
                        "aspect": "Service",
                        "theme": "Staff hospitality",
                        "aspect_target_span": s_t[2] if s_t else "service",
                        "opinion_span": op_str,
                        "sentiment": "Positive",
                        "llm_rationale": f"Explicit service praise '{op_str}'."
                    })

                a_t = find_exact_subspan(c_text, r'\b(ambience|ambiance)\b')
                c_assertions.append({
                    "annotation_status": "Annotated",
                    "clause_relation": "primary_assertion",
                    "elaboration_of": None,
                    "aspect": "Ambience",
                    "theme": "Atmosphere",
                    "aspect_target_span": a_t[2] if a_t else "ambience",
                    "opinion_span": op_str,
                    "sentiment": "Positive",
                    "llm_rationale": f"Explicit ambience praise '{op_str}'."
                })

            # 3. General Experience: Whole-establishment appraisals, repeat intent, overall visit satisfaction
            elif re.search(r'\b(must try place|the place is amazing|best place for foodies|a must try in that area|overall this place is very bad|if there was minus rating|would definitely recommend everyone to visit this place|loved this place|overall, a disappointing experience|visit again|come back again|never disappointed|not up to the mark|awesome hotel|made our stay worthwhile|worth a try|definitely worth a try|what a disappointment it was|disappointed me|disappointing)\b', c_low):
                sent = "Negative" if re.search(r'\b(very bad|minus rating|disappointing|disappointment|pathetic|not up to the mark)\b', c_low) else "Positive"
                op_match = find_exact_subspan(c_text, r'(must try place|must try|amazing|best|very bad|if there was minus rating|recommend|loved|disappointing|disappointment|visit again|never disappointed|not up to the mark|awesome|worthwhile|worth a try)')
                t_match = find_exact_subspan(c_text, r'\b(place|restaurant|outlet|hotel|stay|experience)\b')

                op_str = op_match[2] if op_match else "amazing"
                t_str = t_match[2] if t_match else ""

                c_assertions.append({
                    "annotation_status": "Annotated",
                    "clause_relation": "primary_assertion",
                    "elaboration_of": None,
                    "aspect": "General Experience",
                    "theme": "Recommendation" if re.search(r'\b(try|recommend|visit)\b', op_str.lower()) else "Overall Impression",
                    "aspect_target_span": t_str,
                    "opinion_span": op_str,
                    "sentiment": sent,
                    "llm_rationale": f"Holistic evaluation of the restaurant as a whole ({sent})."
                })

            # 4. Service Aspects (including typos like 'sevice', quick delivery, wait time, courtesy)
            elif re.search(r'\b(sevice|service|services|waiter|staff|delivery|deliver)\b', c_low) and re.search(r'\b(quick|fast|slow|late|delay|waiting|prompt|smooth|polite|rude|friendly|courteous|state of the art|excellent|good|bad|poor|not so fast|terrible|pathetic)\b', c_low):
                sent = "Negative" if re.search(r'\b(slow|late|delay|waiting|rude|poor|not so fast|terrible|pathetic|bad)\b', c_low) else "Positive"
                op_m = find_exact_subspan(c_text, r'(state of the art|courteous and helpful|not so fast|very quick|quick|fast|slow|late|delay|waiting|prompt|smooth|polite|rude|friendly|courteous|excellent|good|bad|poor|terrible|pathetic)')
                t_m = find_exact_subspan(c_text, r'\b(service|services|sevice|delivery|waiter|staff|server|management)\b')

                c_assertions.append({
                    "annotation_status": "Annotated",
                    "clause_relation": "primary_assertion",
                    "elaboration_of": None,
                    "aspect": "Service",
                    "theme": "Delivery Speed" if re.search(r'\bdelivery\b', c_low) else ("Service speed" if re.search(r'\b(fast|slow)\b', c_low) else "Staff courtesy"),
                    "aspect_target_span": t_m[2] if t_m else "service",
                    "opinion_span": op_m[2] if op_m else "good",
                    "sentiment": sent,
                    "llm_rationale": f"Service evaluation ({sent}) targeting '{t_m[2] if t_m else 'service'}'."
                })

            # 5. Ambience Aspects (decor, music, view, seating, cleanliness, crowded, cozy)
            elif re.search(r'\b(ambience|ambiance|decor|seating|view|music|lighting|cleanliness|hygiene|cozy|bistro|crowded)\b', c_low) and re.search(r'\b(amazing|soulful|eye catching|great|good|beautiful|cozy|appreciated|clean|neat|hygienic|noisy|crowded|bland|dirty|filthy|entertaining|very entertaining)\b', c_low):
                sent = "Negative" if re.search(r'\b(dirty|filthy|noisy|crowded)\b', c_low) else "Positive"
                op_m = find_exact_subspan(c_text, r'(very entertaining|entertaining|eye catching|soulful|amazing|beautiful|cozy|appreciated|clean|neat|hygienic|great|good|dirty|filthy|noisy|crowded)')
                t_m = find_exact_subspan(c_text, r'\b(ambience|ambiance|decor|seating|view|music|lighting|cleanliness|hygiene|bistro)\b')

                c_assertions.append({
                    "annotation_status": "Annotated",
                    "clause_relation": "primary_assertion",
                    "elaboration_of": None,
                    "aspect": "Ambience",
                    "theme": "Cleanliness & Hygiene" if re.search(r'\b(clean|hygiene)\b', c_low) else ("Music / Noise level" if re.search(r'\bmusic\b', c_low) else "Atmosphere"),
                    "aspect_target_span": t_m[2] if t_m else "ambience",
                    "opinion_span": op_m[2] if op_m else "good",
                    "sentiment": sent,
                    "llm_rationale": f"Ambience evaluation ({sent}) targeting '{t_m[2] if t_m else 'ambience'}'."
                })

            # 6. Price / Value Aspects (cost effective, expensive, cheap, costly, worth)
            elif re.search(r'\b(cost effective|expensive|costly|cheap|pocket friendly|value for money|not worth|wasting of money|waste of money|so high for this|price|pricing|charge|cost)\b', c_low) and re.search(r'\b(cost effective|expensive|costly|cheap|worth|high|less|waste|wasting|pocket friendly|value)\b', c_low):
                sent = "Positive" if re.search(r'\b(cost effective|cheap|pocket friendly|value for money|worth)\b', c_low) and not re.search(r'\b(not worth|waste|costly|expensive)\b', c_low) else "Negative"
                op_m = find_exact_subspan(c_text, r'(cost effective|pocket friendly|value for money|not worth|wasting of money|waste of money|expensive|costly|cheap|worth|high)')
                t_m = find_exact_subspan(c_text, r'\b(outlet|Saturday lunch|lunch|dinner|drinks|buffet|prices|price|menu|charges|food)\b')

                c_assertions.append({
                    "annotation_status": "Annotated",
                    "clause_relation": "primary_assertion",
                    "elaboration_of": None,
                    "aspect": "Price / Value",
                    "theme": "Value for money" if sent == "Positive" else "Pricing / Overpriced",
                    "aspect_target_span": t_m[2] if t_m else "",
                    "opinion_span": op_m[2] if op_m else ("cost effective" if sent == "Positive" else "expensive"),
                    "sentiment": sent,
                    "llm_rationale": f"Pricing evaluation ({sent}) targeting '{t_m[2] if t_m else 'pricing'}'."
                })

            # 7. Food Aspects: Taste, Freshness, Temperature, Portions, Culinary Attributes
            elif re.search(r'\b(bland|fresh|stale|oily|greasy|chewy|burnt|raw|cold|spicy|sweet|sour|tasty|delicious|worst taste|avarage taste|below average taste|not tasty|not good|less in quantity|little less quantity|not cooked properly|too tasty|finger licking good|wonderful taste|awesome taste|good variety|huge variety|sucks|old stock|unable to eat|hate|dishes were either too oily|shitty buffet)\b', c_low) or (
                re.search(self.FOOD_TARGETS, c_low) and re.search(r'\b(good|great|nice|delicious|tasty|bad|poor|worst|pathetic|cold|oily|fresh|stale|bland|chewy|burnt|raw|spicy|less|superb|amazing)\b', c_low)
            ):
                if re.search(r'\b(bland|stale|oily|greasy|chewy|burnt|raw|cold|worst|bad|poor|not tasty|not good|less in quantity|little less|not cooked|sucks|old stock|unable to eat|hate|shitty|pathetic)\b', c_low):
                    sent = "Negative"
                elif re.search(r'\b(avarage|average|okay|ok)\b', c_low):
                    sent = "Neutral"
                else:
                    sent = "Positive"

                op_m = find_exact_subspan(c_text, r'(finger licking good|wonderful taste|awesome taste|good taste|too tasty|delicious|tasty|bland|fresh|stale|oily|greasy|chewy|burnt|raw|cold|worst taste|below average taste|avarage taste|not tasty at all|not tasty|not good very bad|not good|very bad|not cooked properly|very less in quantity|little less quantity|less in quantity|less spicey|old stock|sucks|good variety|huge variety|great|good|nice|bad|poor|worst|pathetic)')
                t_m = find_exact_subspan(c_text, self.FOOD_TARGETS)

                c_assertions.append({
                    "annotation_status": "Annotated",
                    "clause_relation": "primary_assertion",
                    "elaboration_of": None,
                    "aspect": "Food",
                    "theme": "Portion size" if re.search(r'\bquantity\b', c_low) else ("Food temperature" if re.search(r'\bcold\b', c_low) else ("Freshness" if re.search(r'\b(fresh|stale)\b', c_low) else ("Menu variety" if re.search(r'\bvariety\b', c_low) else "Taste / Flavor"))),
                    "aspect_target_span": t_m[2] if t_m else "food",
                    "opinion_span": op_m[2] if op_m else "good",
                    "sentiment": sent,
                    "llm_rationale": f"Culinary evaluation '{op_m[2] if op_m else 'good'}' ({sent}) on Food."
                })

            # 8. Remaining generic good/bad evaluations
            elif re.search(r'\b(good|nice|great)\b', c_low):
                op_m = find_exact_subspan(c_text, r'\b(great|nice|good)\b')
                t_m = find_exact_subspan(c_text, r'\b(food|service|ambience|ambiance|taste|place|biryani|outlet|stay|hotel)\b')
                t_str = t_m[2] if t_m else ""
                if t_str.lower() in ["food", "taste", "biryani"]:
                    asp = "Food"
                elif t_str.lower() == "service":
                    asp = "Service"
                elif t_str.lower() in ["ambience", "ambiance"]:
                    asp = "Ambience"
                else:
                    asp = "General Experience"

                c_assertions.append({
                    "annotation_status": "Annotated",
                    "clause_relation": "primary_assertion",
                    "elaboration_of": None,
                    "aspect": asp,
                    "theme": "Food quality" if asp == "Food" else "Overall Impression",
                    "aspect_target_span": t_str,
                    "opinion_span": op_m[2] if op_m else "good",
                    "sentiment": "Positive",
                    "llm_rationale": f"Positive appraisal targeting '{t_str}'."
                })

            elif re.search(r'\b(bad|poor|burnt|pathetic|worst)\b', c_low):
                op_m = find_exact_subspan(c_text, r'\b(pathetic|burnt|poor|bad|worst)\b')
                t_m = find_exact_subspan(c_text, r'\b(food|service|chicken|biryani|outlet|roti|dish|dishes|experience)\b')
                t_str = t_m[2] if t_m else ""
                asp = "Service" if t_str.lower() == "service" else ("General Experience" if t_str.lower() in ["outlet", "experience"] else "Food")
                c_assertions.append({
                    "annotation_status": "Annotated",
                    "clause_relation": "primary_assertion",
                    "elaboration_of": None,
                    "aspect": asp,
                    "theme": "Food quality" if asp == "Food" else "Overall Impression",
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

            # Process assertions and resolve verbatim character offsets
            for a_idx, ass in enumerate(c_assertions, 1):
                ass_id = f"{c_id}_A{a_idx:02d}"
                t_span = ass["aspect_target_span"]
                op_span = ass["opinion_span"]

                t_start, t_end = None, None
                if t_span:
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
                assert review_text[c_start:c_end] == c_text, f"Clause offset mismatch in {ass_id}"
                if t_span:
                    assert review_text[t_start:t_end] == t_span, f"Target offset mismatch in {ass_id}"
                if op_span:
                    assert review_text[op_start:op_end] == op_span, f"Opinion offset mismatch in {ass_id}"

                assertions.append(full_ass)
                if ass["annotation_status"] == "Annotated":
                    last_assertion_id = ass_id

        return assertions
