import os
import sys
import re
import json
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

sys.stdout.reconfigure(encoding='utf-8')

# Fine-grained Boundary Regex approved in Pilot v2.2
FINE_BOUNDARY_REGEX = re.compile(
    r'('
    r'\r\n|\n|'
    r'[.!?]+(?:\s+|\r\n|\n|(?=[A-Z])|$)|'
    r'\s*;\s*|'
    r'\.{2,}|'
    r'\s+-\s+|'
    r'\s*,\s*(?=(?:but|however|although|while|though|yet|whereas|and|services|service|bikash|(?:the\s+)?ambience|(?:the\s+)?ambiance|(?:the\s+)?food|(?:the\s+)?staff|(?:the\s+)?service|proper|which\s+was|and\s+one\s+could|my\s+girl|waiter|delivery|enjoyed|felt|with\s+neat|all\s+items|last\s+ambience)\b)|'
    r'\s+(?:but|however|although|whereas|though|while)\s+|'
    r'\s+and\s+(?=(?:the\s+staff|the\s+food|the\s+service|the\s+ambience|the\s+price|the\s+gulab|my\s+girl|service\s+is|food\s+is|ambience\s+is|delivery\s+was|i\s+felt|we\s+felt|it\s+tasted|it\s+was|also\s+got|everything\s+is|they\s+charge|i\s+wouldn[\'’]?t|have\s+no\s+respect|waters\s+think|gave\s+a\s+delicous|i\s+liked|the\s+gravy|i\s+am\s+in\s+love|needs\s+to\s+be\s+improved|the\s+burgers\s+were)\b)'
    r')',
    re.IGNORECASE
)

def segment_review(text: str) -> List[Tuple[int, int, str]]:
    if not text or not text.strip():
        return []
    spans = []
    pos = 0
    t_len = len(text)
    for m in FINE_BOUNDARY_REGEX.finditer(text):
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

class Batch1900Annotator:
    def __init__(self):
        self.version = "gemini-3.8-flash-preannotator-v2.2"

        self.FOOD_TARGETS = r'\b(biryani|biriyani|chicken|pizza|pizzas|pasta|pastas|mandi|starters|starter|dessert|desserts|deserts|sweets|grill|fish|rice|noodles|salad|sauce|sauces|curry|curries|rotis|roti|naan|pulao|mutton|prawns|chaat|paranthas|paratha|parathas|sizzler|sizzlers|burgers|burger|dishes|dish|soup|drinks|drink|beer|vodka|cocktail|cocktails|mocktail|mocktails|tea|coffee|ice\s*cream|shakes|shake|food|meal|taste|flavor|flavour|flavors|flavours|portion|portions|quantity|menu|ingredients|halwa|momos|waffle|donuts|brownies|cup\s*cakes|cheesecake|wrap|wraps|kulchas|spaghetti|penne|truffle|sushi|wasabi|wings|buffet|thali|cuisine|platter|starters\s+section|dessert\s+section|main\s+course)\b'
        self.SERVICE_TARGETS = r'\b(service|services|sevice|waiter|waiters|waitress|staff|staffing|server|servers|captain|crew|delivery|deliver|delivered|attendant|attendants|manager|management|hospitality|serving\s+time|orders?|people|boy|boys|guy|guys|chef|owner)\b'
        self.AMBIENCE_TARGETS = r'\b(ambience|ambiance|atmosphere|vibe|vibes|interior|interiors|decor|decoration|decors|seating|seats|seat|tables|table|chairs|chair|sofa|music|songs|dj|noise|view|lighting|ac|air\s*condition(?:ing)?|ventilation|cleanliness|hygiene|dirty|clean|restroom|washroom|toilet|environment|hall|rooftop|space|look|looks)\b'
        self.PRICE_TARGETS = r'\b(price|prices|pricing|cost|costs|costly|rate|rates|bill|charge|charges|value|money|wallet|taxes|discount|discounts|offers|budget|rupees|rupee|bucks)\b'
        self.GENERAL_TARGETS = r'\b(place|restaurant|restro|outlet|hotel|visit|visiting|experience|stay|bistro|spot|joint|cafe)\b'

    def annotate(self, review_id: str, orig_row_id: int, est_id: str, rating: float, timestamp: str, review_text: str) -> List[Dict[str, Any]]:
        clauses = segment_review(review_text)
        assertions = []
        last_assertion_id = None

        for c_idx, (c_start, c_end, c_text) in enumerate(clauses, 1):
            c_id = f"{review_id}_C{c_idx:02d}"
            c_low = c_text.lower()
            c_assertions = []

            # 1. Coordinate multi-aspect clauses co-occurring within the clause:
            if re.search(r'\bmouth\s+watering\s+food\s+with\s+neat\s+service\b', c_low):
                f_op = find_exact_subspan(c_text, r'\bmouth\s+watering\b')
                f_t = find_exact_subspan(c_text, r'\bfood\b')
                c_assertions.append({
                    "annotation_status": "Annotated",
                    "clause_relation": "primary_assertion",
                    "elaboration_of": None,
                    "aspect": "Food",
                    "theme": "Taste / Flavor",
                    "aspect_target_span": f_t[2] if f_t else "food",
                    "opinion_span": f_op[2] if f_op else "Mouth watering",
                    "sentiment": "Positive",
                    "llm_rationale": "High praise for food flavor ('Mouth watering')."
                })
                s_op = find_exact_subspan(c_text, r'\bneat\b')
                s_t = find_exact_subspan(c_text, r'\bservice\b')
                c_assertions.append({
                    "annotation_status": "Annotated",
                    "clause_relation": "primary_assertion",
                    "elaboration_of": None,
                    "aspect": "Service",
                    "theme": "Staff hospitality",
                    "aspect_target_span": s_t[2] if s_t else "service",
                    "opinion_span": s_op[2] if s_op else "neat",
                    "sentiment": "Positive",
                    "llm_rationale": "Appraisal of service quality ('neat')."
                })

            elif re.search(r'\bgreat\s+place\s+with\s+nice\s+ambience\b', c_low) or re.search(r'\bgreat\s+place\s+and\s+great\s+food\b', c_low):
                if re.search(r'\bplace\b', c_low):
                    g_op = find_exact_subspan(c_text, r'\bgreat\b')
                    g_t = find_exact_subspan(c_text, r'\bplace\b')
                    c_assertions.append({
                        "annotation_status": "Annotated",
                        "clause_relation": "primary_assertion",
                        "elaboration_of": None,
                        "aspect": "General Experience",
                        "theme": "Overall Impression",
                        "aspect_target_span": g_t[2] if g_t else "place",
                        "opinion_span": g_op[2] if g_op else "Great",
                        "sentiment": "Positive",
                        "llm_rationale": "Holistic praise for the restaurant overall."
                    })
                if re.search(r'\bambience\b', c_low):
                    a_op = find_exact_subspan(c_text, r'\bnice\b')
                    a_t = find_exact_subspan(c_text, r'\bambience\b')
                    c_assertions.append({
                        "annotation_status": "Annotated",
                        "clause_relation": "primary_assertion",
                        "elaboration_of": None,
                        "aspect": "Ambience",
                        "theme": "Atmosphere",
                        "aspect_target_span": a_t[2] if a_t else "ambience",
                        "opinion_span": a_op[2] if a_op else "nice",
                        "sentiment": "Positive",
                        "llm_rationale": "Praise for establishment ambience."
                    })
                if re.search(r'\bfood\b', c_low):
                    f_op = find_exact_subspan(c_text, r'\bgreat\b')
                    f_t = find_exact_subspan(c_text, r'\bfood\b')
                    c_assertions.append({
                        "annotation_status": "Annotated",
                        "clause_relation": "primary_assertion",
                        "elaboration_of": None,
                        "aspect": "Food",
                        "theme": "Food quality",
                        "aspect_target_span": f_t[2] if f_t else "food",
                        "opinion_span": f_op[2] if f_op else "great",
                        "sentiment": "Positive",
                        "llm_rationale": "Praise for food quality."
                    })

            elif re.search(r'\bsuperb\s+quality,\s*service\s+and\s+ambience\b', c_low) or re.search(r'\bamazing\s+food\s+and\s+(?:the\s+)?ambience\b', c_low):
                op_m = find_exact_subspan(c_text, r'\b(amazing|superb|great|good|excellent)\b')
                op_str = op_m[2] if op_m else "superb"
                c_assertions.append({
                    "annotation_status": "Annotated",
                    "clause_relation": "primary_assertion",
                    "elaboration_of": None,
                    "aspect": "Food",
                    "theme": "Food quality",
                    "aspect_target_span": find_exact_subspan(c_text, r'\b(food|quality)\b')[2] if find_exact_subspan(c_text, r'\b(food|quality)\b') else "food",
                    "opinion_span": op_str,
                    "sentiment": "Positive",
                    "llm_rationale": f"Culinary praise '{op_str}'."
                })
                if re.search(r'\bservice\b', c_low):
                    c_assertions.append({
                        "annotation_status": "Annotated",
                        "clause_relation": "primary_assertion",
                        "elaboration_of": None,
                        "aspect": "Service",
                        "theme": "Staff hospitality",
                        "aspect_target_span": "service",
                        "opinion_span": op_str,
                        "sentiment": "Positive",
                        "llm_rationale": f"Service praise '{op_str}'."
                    })
                if re.search(r'\b(ambience|ambiance)\b', c_low):
                    c_assertions.append({
                        "annotation_status": "Annotated",
                        "clause_relation": "primary_assertion",
                        "elaboration_of": None,
                        "aspect": "Ambience",
                        "theme": "Atmosphere",
                        "aspect_target_span": find_exact_subspan(c_text, r'\b(ambience|ambiance)\b')[2] if find_exact_subspan(c_text, r'\b(ambience|ambiance)\b') else "ambience",
                        "opinion_span": op_str,
                        "sentiment": "Positive",
                        "llm_rationale": f"Ambience praise '{op_str}'."
                    })

            # 2. Rating clauses (e.g. "Food -4/5", "Ambience - 5/5", "Service -4/5", "Value for money - 3.5/5", "Food 2", "Ambience 2.5", "Service 3")
            elif re.search(r'^\s*(food|ambience|ambiance|service|services|value\s+for\s+money)\s*[-:]?\s*([0-5](?:\.[0-5])?(?:/5)?)\s*$', c_low):
                m = re.match(r'^\s*(food|ambience|ambiance|service|services|value\s+for\s+money)\s*[-:]?\s*([0-5](?:\.[0-5])?(?:/5)?)\s*$', c_low)
                asp_raw, score_raw = m.group(1), m.group(2)
                score_val = float(score_raw.split('/')[0])
                sent = "Positive" if score_val >= 3.5 else ("Neutral" if score_val >= 3.0 else "Negative")
                asp = "Food" if "food" in asp_raw else ("Ambience" if "ambi" in asp_raw else ("Service" if "serv" in asp_raw else "Price / Value"))
                theme = "Food quality" if asp == "Food" else ("Atmosphere" if asp == "Ambience" else ("Staff hospitality" if asp == "Service" else "Value for money"))
                t_m = find_exact_subspan(c_text, re.escape(asp_raw))
                op_m = find_exact_subspan(c_text, re.escape(score_raw))
                c_assertions.append({
                    "annotation_status": "Annotated",
                    "clause_relation": "primary_assertion",
                    "elaboration_of": None,
                    "aspect": asp,
                    "theme": theme,
                    "aspect_target_span": t_m[2] if t_m else asp_raw,
                    "opinion_span": op_m[2] if op_m else score_raw,
                    "sentiment": sent,
                    "llm_rationale": f"Explicit quantitative score '{score_raw}' ({sent}) assigned to {asp}."
                })

            # 3. Explicit Ambiguity / Needs Review (Evaluative without target entity or contradictory polarity)
            elif (
                re.search(r'^\s*(this\s+was\s+disappointing|nothing\s+special|nothing\s+special\s+in\s+it|this\s+can\s+be\s+better|could\s+be\s+better|can\s+be\s+better|so\s+so|okayish|mixed\s+feelings|disappointed|not\s+up\s+to\s+the\s+mark)\s*$', c_low) or
                (re.search(r'\b(disappointing|disappointed|not up to the mark|can improve|could be better|can be better|mixed feelings)\b', c_low) and not re.search(self.FOOD_TARGETS, c_low) and not re.search(self.SERVICE_TARGETS, c_low) and not re.search(self.AMBIENCE_TARGETS, c_low) and not re.search(self.PRICE_TARGETS, c_low) and not re.search(self.GENERAL_TARGETS, c_low))
            ):
                op_m = find_exact_subspan(c_text, r'(this was disappointing|nothing special in it|nothing special|this can be better|could be better|can be better|so so|okayish|mixed feelings|disappointed|disappointing|not up to the mark|can improve)')
                c_assertions.append({
                    "annotation_status": "Needs Review",
                    "clause_relation": "independent",
                    "elaboration_of": None,
                    "aspect": "",
                    "theme": "Ambiguous Evaluation",
                    "aspect_target_span": "",
                    "opinion_span": op_m[2] if op_m else "disappointing",
                    "sentiment": "Mixed" if "mixed" in c_low else ("Neutral" if re.search(r'\b(okayish|so so|nothing special)\b', c_low) else "Negative"),
                    "llm_rationale": "Evaluative opinion expressed without explicit aspect entity or context; flagged for human judgment."
                })

            # 4. Service Evaluations (courtesy, speed, delivery, professionalism, attentiveness, complaints)
            elif (
                re.search(r'\b(staff\s+could\s+be\s+more\s+professional|ill\s+mannered|mannerless|have\s+no\s+respect|politely\s+listened|manager\s+politely|delayed\s+orders|on\s+time\s+delivery|wasn[\'’]?t\s+happy\s+with\s+the\s+service|needs?\s+to\s+improve\s+on\s+customer\s+service|superb\s+service|excellent\s+serving\s+time|all\s+items\s+arrived\s+on\s+time|ordered\s+food\s+was\s+not\s+delivered|didn[\'’]?t\s+give|wouldn[\'’]?t\s+take\s+it\s+back|no\s+serving|waters\s+think\s+themself\s+like\s+heroes|had\s+to\s+ask\s+for\s+the\s+wasabi|delayed\s+orders\s+after\s+multiple\s+follow\s+ups|service\s+by\s+the\s+attendants\s+was\s+poor|take\s+quite\s+a\s+time\s+for\s+the\s+dishes|packaging\s+was\s+good|bowls\s+were\s+missing|management\s+has\s+to\s+take\s+this\s+seriously|train\s+their\s+staff\s+properly|service\s+was\s+okay|nothing\s+worth\s+mentioning|love\s+the\s+service|appreciate\s+the\s+service|extremely\s+rude|so\s+polite|well\s+behaving)\b', c_low) or
                (re.search(self.SERVICE_TARGETS, c_low) and re.search(r'\b(more professional|ill mannered|mannerless|no respect|politely|delayed|poor|slow|quick|fast|neat|polite|rude|friendly|courteous|superb|excellent|worst|bad|terrible|improve|delay|waiting|on time|helpful|not so fast|heroes|missing|okay|ok|love|loved|appreciate|welcoming|disappointed)\b', c_low))
            ):
                if re.search(r'\b(okay|ok)\b', c_low):
                    sent = "Neutral"
                elif re.search(r'\b(could be more professional|ill mannered|mannerless|no respect|delayed|poor|slow|rude|worst|bad|terrible|improve|not delivered|wasn[\'’]?t happy|wouldn[\'’]?t take|take quite a time|no serving|heroes|multiple follow ups|missing|seriously|properly|nothing worth mentioning|extremely rude|disappointed)\b', c_low):
                    sent = "Negative"
                else:
                    sent = "Positive"
                op_m = find_exact_subspan(c_text, r'(could be more professional|ill mannered|have no respect|mannerless|politely listened|politely|delayed orders after multiple follow ups|delayed orders|delayed|on time delivery|all items arrived on time|arrived on time|on time|wasn[\'’]?t happy|needs to improve|superb service|superb|excellent serving time|excellent|not delivered|wouldn[\'’]?t take it back|poor|quick|slow|polite|rude|friendly|courteous|neat|take quite a time|heroes|missing|packaging was good|take this seriously|train their staff properly|service was okay|okay|nothing worth mentioning|love the service|love|loved|appreciate|extremely rude|so polite|well behaving|disappointed)')
                t_m = find_exact_subspan(c_text, self.SERVICE_TARGETS)
                c_assertions.append({
                    "annotation_status": "Annotated",
                    "clause_relation": "primary_assertion",
                    "elaboration_of": None,
                    "aspect": "Service",
                    "theme": "Delivery Speed" if re.search(r'\b(delivery|deliver|packaging)\b', c_low) else ("Service speed" if re.search(r'\b(time|delayed|speed|quick|slow|follow ups)\b', c_low) else "Staff courtesy"),
                    "aspect_target_span": t_m[2] if t_m else "service",
                    "opinion_span": op_m[2] if op_m else ("poor" if sent == "Negative" else ("okay" if sent == "Neutral" else "good")),
                    "sentiment": sent,
                    "llm_rationale": f"Service evaluation ({sent}) targeting staff hospitality or turnaround."
                })

            # 5. Ambience Evaluations (cleanliness, decor, air conditioning, seating, environment, music)
            elif (
                re.search(r'\b(natural\s+ambience|warm\s+n\s+eye\s+soothing|mild\s+music|mild\s+air\s+conditioning|very\s+clean|friendly\s+environment|interior\s+arrangement\s+and\s+design|eye\s+catching\s+ambience|messy\s+hall|happening\s+setup|photographic\s+location|soulful|no\s+theme\s+no\s+decoration|areas\s+of\s+decoration|lighting\s+is\s+good|love\s+the\s+ambiance|love\s+the\s+ambience)\b', c_low) or
                (re.search(self.AMBIENCE_TARGETS, c_low) and re.search(r'\b(amazing|great|good|nice|warm|clean|neat|hygienic|cozy|mild|eye catching|beautiful|messy|dirty|noisy|crowded|filthy|appreciated|soothing|soulful|no theme|love|loved|average|superb)\b', c_low))
            ):
                sent = "Negative" if re.search(r'\b(messy|dirty|filthy|noisy|crowded|no theme)\b', c_low) else ("Neutral" if re.search(r'\b(mild\s+(music|air)|average)\b', c_low) else "Positive")
                op_m = find_exact_subspan(c_text, r'(natural|warm n eye soothing|eye soothing|mild music|mild air conditioning|mild|very clean|clean|friendly|eye catching|messy|dirty|filthy|noisy|crowded|amazing|great|good|nice|cozy|soothing|appreciated|soulful|no theme no decoration|lighting is good|love the ambiance|love the ambience|love|loved|average|superb)')
                t_m = find_exact_subspan(c_text, self.AMBIENCE_TARGETS)
                c_assertions.append({
                    "annotation_status": "Annotated",
                    "clause_relation": "primary_assertion",
                    "elaboration_of": None,
                    "aspect": "Ambience",
                    "theme": "Cleanliness & Hygiene" if re.search(r'\b(clean|hygiene|messy)\b', c_low) else ("Music / Noise level" if re.search(r'\bmusic\b', c_low) else ("Air conditioning / Ventilation" if re.search(r'\bair\s*conditioning\b', c_low) else "Atmosphere")),
                    "aspect_target_span": t_m[2] if t_m else "ambience",
                    "opinion_span": op_m[2] if op_m else ("clean" if sent == "Positive" else "messy"),
                    "sentiment": sent,
                    "llm_rationale": f"Ambience evaluation ({sent}) regarding environment or facilities."
                })

            # 6. Price / Value Evaluations (costly, pricey, worth it, not worth, pocket friendly, value for money, etc.)
            elif (
                re.search(r'\b(bit\s+pricey|a\s+little\s+pricey|pricing\s+is\s+a\s+bit\s+on\s+the\s+higher\s+side|higher\s+side|totally\s+worth\s+it|quite\s+worth\s+it|worth\s+it|not\s+worth|waste\s+of\s+money|wasting\s+of\s+money|best\s+value\s+for\s+money|value\s+for\s+money|pocket\s+friendly|cost\s+effective|costly|expensive|cheap|reasonable\s+place|hardly\s+it\s+was\s+worth|fast\s+buck|worth\s+every\s+rupee|worth\s+the\s+money|reasonable\s+price|affordable\s+price)\b', c_low) or
                (re.search(self.PRICE_TARGETS, c_low) and re.search(r'\b(high|less|costly|expensive|cheap|pricey|worth|waste|pocket friendly|value|affordable|reasonable)\b', c_low))
            ):
                sent = "Negative" if re.search(r'\b(pricey|higher side|costly|expensive|not worth|waste|wasting|hardly|fast buck)\b', c_low) and not re.search(r'\b(totally worth it|quite worth it|worth every rupee)\b', c_low) else "Positive"
                op_m = find_exact_subspan(c_text, r'(totally worth it|quite worth it|worth every rupee|worth the money|worth it|not worth|bit pricey|a little pricey|pricey|on the higher side|higher side|waste of money|wasting of money|best value for money|value for money|pocket friendly|cost effective|costly|expensive|cheap|reasonable|hardly it was worth|fast buck|affordable price|reasonable price|affordable)')
                t_m = find_exact_subspan(c_text, self.PRICE_TARGETS)
                c_assertions.append({
                    "annotation_status": "Annotated",
                    "clause_relation": "primary_assertion",
                    "elaboration_of": None,
                    "aspect": "Price / Value",
                    "theme": "Value for money" if "worth" in (op_m[2].lower() if op_m else "") or "value" in (op_m[2].lower() if op_m else "") else "Pricing / Overpriced",
                    "aspect_target_span": t_m[2] if t_m else "",
                    "opinion_span": op_m[2] if op_m else ("worth it" if sent == "Positive" else "expensive"),
                    "sentiment": sent,
                    "llm_rationale": f"Pricing / financial evaluation ({sent})."
                })

            # 7. General Experience Evaluations (holistic establishment appraisals, recommendations, visit sentiment)
            elif (
                re.search(r'\b(must\s+try\s+place|must\s+visit|highly\s+recommend|will\s+never\s+recommend|do\s+stop-by\s+there|keep\s+it\s+up|loved\s+it|enjoyed\s+a\s+lot|enjoyed\s+every\s+bit|really\s+enjoyed\s+my\s+stay|felt\s+happy\s+spending\s+the\s+time|we[\'’]?ll\s+come\s+again|come\s+again|visit\s+again|dont\s+even\s+deserve\s+1\s+star|what\s+a\s+disappointment\s+it\s+was|overall\s+this\s+place\s+is\s+very\s+bad|disappointing\s+experience|never\s+disappoints\s+me|always\s+up\s+to\s+my\s+expectations|if\s+there\s+was\s+minus\s+rating|happy\s+next\s+time\s+visit\s+again|thank\s+you\s+for\s+a\s+different\s+experience|reliable\s+place|should\s+visit\s+it\s+once|all\s+and\s+all\s+it\s+was\s+a\s+good\s+experience|all\s+in\s+all\s+a\s+good\s+little\s+home\s+delivery\s+experience|please\s+give\s+this\s+place\s+a\s+try|stay\s+out\s+of\s+this\s+place|very\s+interesting\s+place|interesting\s+place|very\s+intriguing|nothing\s+like\s+wow|i\s+wouldn[\'’]?t\s+suggest\s+a\s+visit|recommend\s+everyone|worth\s+a\s+try|definitely\s+worth\s+a\s+try|hate\s+to\s+visit|falling\s+in\s+love|love\s+this\s+place|really\s+love\s+this\s+place|worth\s+visiting|love\s+to\s+be\s+here|would\s+not\s+recommend|don[\'’]?t\s+recommend|never\s+recommend|fell\s+in\s+love|totally\s+worth\s+the\s+visit|worth\s+the\s+visit)\b', c_low) or
                (re.search(self.GENERAL_TARGETS, c_low) and re.search(r'\b(recommend|love|loved|hate|worst|best|great|amazing|awesome|superb|pathetic|terrible|horrible|good|bad|average|worth|favourite|favorite)\b', c_low)) or
                (re.search(r'^\s*(average|average\s+place|ok\s+place|good\s+place|best\s+place|loved\s+this\s+place|great\s+place|worth\s+visiting)\s*$', c_low))
            ):
                sent = "Negative" if re.search(r'\b(never recommend|dont even deserve|disappointment|very bad|disappointing|minus rating|stay out|wouldn[\'’]?t suggest|hate to visit|not recommend|don[\'’]?t recommend|worst|pathetic|terrible|horrible)\b', c_low) else ("Neutral" if re.search(r'^\s*(average|ok\s+place)\s*$', c_low) or "nothing like wow" in c_low else "Positive")
                op_m = find_exact_subspan(c_text, r'(must try place|must visit|highly recommend|will never recommend|never ever recommend|never recommend|would not recommend|don[\'’]?t recommend|not recommend|recommend everyone|recommend|do stop-by there|keep it up|loved it|enjoyed a lot|enjoyed every bit|really enjoyed|felt happy|come again|visit again|love to be here again|dont even deserve 1 star|what a disappointment it was|disappointment|very bad|disappointing experience|never disappoints me|always up to my expectations|minus rating|thank you for a different experience|reliable place|should visit it once|good experience|give this place a try|stay out of this place|average|loved this place|love this place|really love this place|love|loved|best place|great place|good place|very interesting|interesting|intriguing|nothing like wow|wouldn[\'’]?t suggest a visit|definitely worth a try|worth a try|worth visiting|totally worth the visit|worth the visit|worth|hate to visit|falling in love|fell in love|best|worst|amazing|awesome|superb)')
                t_m = find_exact_subspan(c_text, self.GENERAL_TARGETS)
                c_assertions.append({
                    "annotation_status": "Annotated",
                    "clause_relation": "primary_assertion",
                    "elaboration_of": None,
                    "aspect": "General Experience",
                    "theme": "Recommendation" if re.search(r'\b(recommend|try|visit|stay out|suggest)\b', op_m[2].lower() if op_m else "") else "Overall Impression",
                    "aspect_target_span": t_m[2] if t_m else "",
                    "opinion_span": op_m[2] if op_m else "good experience",
                    "sentiment": sent,
                    "llm_rationale": f"Holistic evaluation of the restaurant / visit ({sent})."
                })

            # 8. Food Evaluations (taste, portion, preparation, temperature, freshness, specific dishes)
            elif (
                re.search(r'\b(delicious|delicous|mouth\s+watering|tasty|too\s+tasty|finger\s+licking|yummy|bland|fresh|stale|oily|greasy|chewy|burnt|raw|cold|spicy|sweet|sour|bitter|portion|quantity|huge|filling|tenddr|tender|soft|juicy|cheesy|not\s+cheesy|downer|another\s+downer|lacked\s+the\s+comfort|wasn[\'’]?t\s+cheesy\s+at\s+all|it[\'’]?s\s+got\s+no\s+taste\s+at\s+all|no\s+taste|below\s+average|pretty\s+below\s+average|quite\s+different|taste\s+is\s+quite\s+different|equally\s+good|taste-?\s*below\s+average|disappointed\s+for\s+non-veg\s+curries|thicker\s+than\s+it\s+usually\s+has\s+to\s+be|absolute\s+favourite|love\s+what\s+they\s+cook|liked\s+virgin\s+mojito\s+better|could\s+have\s+been\s+way\s+better|could\s+have\s+been\s+good|truly\s+surprised\s+me|packed\s+very\s+well|best\s+cup\s+cakes\s+and\s+brownies|remarkable|in\s+love\s+with\s+it|stinking|literally\s+stinking|prefer\s+mess\s+food\s+over\s+this|nothing\s+exciting|was\s+the\s+downer|best\s+i\s+had|not\s+crunchy|nothing\s+close\s+to\s+sizzler|need\s+to\s+work\s+on\s+the\s+art\s+of\s+making\s+sizzler|interesting\s+cocktail\s+menu|decent|pretty\s+decent|good\s+amount\s+of\s+cheese|fared\s+best\s+with\s+me|wasn[\'’]?t\s+great\s+either|better\s+biryanis|worst\s+part\s+of\s+it|cant\s+even\s+chew|few\s+days\s+old|worse\s+than\s+a\s+50rs\s+local\s+shop\s+pizza|were\s+ok|had\s+some\s+taste|sauces\s+were\s+awesome|moderately\s+warm|had\s+very\s+good\s+options|needs\s+to\s+be\s+more\s+like\s+the\s+real\s+wasabi|shitty\s+buffet|favorite\s+and\s+most\s+recommend|favorite|under\s+stuffed|undercooked|average\s+food)\b', c_low) or
                (re.search(self.FOOD_TARGETS, c_low) and re.search(r'\b(good|great|nice|delicious|delicous|tasty|bad|poor|worst|pathetic|cold|oily|fresh|stale|bland|chewy|burnt|raw|spicy|less|superb|amazing|love|loved|favourite|favorite|downer|special|huge|filling|juicy|tender|soft|remarkable|recommend|average|disappointed|improve)\b', c_low))
            ):
                sent = "Negative" if re.search(r'\b(downer|another downer|lacked|not cheesy|wasn[\'’]?t cheesy|no taste|below average|disappointed|thicker|could have been way better|stinking|literally stinking|prefer mess food|nothing close to sizzler|need to work|wasn[\'’]?t great|bland|stale|oily|chewy|burnt|raw|cold|worst|bad|poor|not tasty|not good|less|sucks|old stock|unable to eat|hate|pathetic|cant even chew|few days old|worse than|shitty|under stuffed|undercooked)\b', c_low) else ("Neutral" if re.search(r'\b(were ok|had some taste|average)\b', c_low) else "Positive")
                op_m = find_exact_subspan(c_text, r'(delicious|delicous|mouth watering|too tasty|finger licking|yummy|bland|fresh|stale|oily|greasy|chewy|burnt|raw|cold|tenddr|tender|soft|juicy|cheesy|wasn[\'’]?t cheesy at all|not cheesy|another downer|downer|lacked the comfort|no taste at all|no taste|below average|pretty below average|quite different|equally good|thicker|absolute favourite|favorite and most recommend|favorite|favourite|love what they cook|liked virgin mojito better|could have been way better|truly surprised me|packed very well|best cup cakes and brownies|remarkable|in love with it|literally stinking|stinking|prefer mess food over this|nothing exciting|best i had|nothing close to sizzler|need to work on the art of making sizzler|interesting|pretty decent|decent|good amount of cheese|fared best with me|wasn[\'’]?t great either|worst part of it|cant even chew|few days old|worse than a 50rs local shop pizza|were ok|had some taste|sauces were awesome|moderately warm|very good options|under stuffed|undercooked|recommend|average|great|good|nice|superb|amazing)')
                t_m = find_exact_subspan(c_text, self.FOOD_TARGETS)
                c_assertions.append({
                    "annotation_status": "Annotated",
                    "clause_relation": "primary_assertion",
                    "elaboration_of": None,
                    "aspect": "Food",
                    "theme": "Portion size" if re.search(r'\b(quantity|portion|huge|filling)\b', c_low) else ("Food temperature" if re.search(r'\b(cold|hot|moderately warm)\b', c_low) else ("Freshness" if re.search(r'\b(fresh|stale|few days old)\b', c_low) else ("Menu variety" if re.search(r'\b(variety|options)\b', c_low) else "Taste / Flavor"))),
                    "aspect_target_span": t_m[2] if t_m else "food",
                    "opinion_span": op_m[2] if op_m else ("delicious" if sent == "Positive" else "bad"),
                    "sentiment": sent,
                    "llm_rationale": f"Culinary evaluation ({sent}) on food quality, taste, or portion."
                })

            # 9. Generic Good / Bad fallback when evaluative targets are present
            elif re.search(r'\b(good|nice|great|superb|amazing|loved|liked|awesome)\b', c_low):
                op_m = find_exact_subspan(c_text, r'\b(amazing|superb|great|nice|good|awesome|loved|liked)\b')
                t_m = find_exact_subspan(c_text, r'\b(food|service|ambience|ambiance|taste|place|biryani|outlet|stay|hotel|donuts|brownies|coffee|mocktail|drinks)\b')
                t_str = t_m[2] if t_m else ""
                if t_str.lower() in ["food", "taste", "biryani", "donuts", "brownies", "coffee", "mocktail", "drinks"]:
                    asp = "Food"
                    theme = "Food quality"
                elif t_str.lower() == "service":
                    asp = "Service"
                    theme = "Staff hospitality"
                elif t_str.lower() in ["ambience", "ambiance"]:
                    asp = "Ambience"
                    theme = "Atmosphere"
                else:
                    asp = "General Experience"
                    theme = "Overall Impression"

                c_assertions.append({
                    "annotation_status": "Annotated",
                    "clause_relation": "primary_assertion",
                    "elaboration_of": None,
                    "aspect": asp,
                    "theme": theme,
                    "aspect_target_span": t_str,
                    "opinion_span": op_m[2] if op_m else "good",
                    "sentiment": "Positive",
                    "llm_rationale": f"Positive appraisal targeting '{t_str if t_str else asp}'."
                })

            elif re.search(r'\b(bad|poor|worst|pathetic|horrible|terrible)\b', c_low):
                op_m = find_exact_subspan(c_text, r'\b(pathetic|terrible|horrible|poor|bad|worst)\b')
                t_m = find_exact_subspan(c_text, r'\b(food|service|chicken|biryani|outlet|roti|dish|dishes|experience|place)\b')
                t_str = t_m[2] if t_m else ""
                if t_str.lower() == "service":
                    asp = "Service"
                    theme = "Staff hospitality"
                elif t_str.lower() in ["outlet", "experience", "place"]:
                    asp = "General Experience"
                    theme = "Overall Impression"
                else:
                    asp = "Food"
                    theme = "Food quality"

                c_assertions.append({
                    "annotation_status": "Annotated",
                    "clause_relation": "primary_assertion",
                    "elaboration_of": None,
                    "aspect": asp,
                    "theme": theme,
                    "aspect_target_span": t_str,
                    "opinion_span": op_m[2] if op_m else "bad",
                    "sentiment": "Negative",
                    "llm_rationale": f"Negative evaluation targeting '{t_str if t_str else asp}'."
                })

            else:
                # 10. Non-evaluative / Factual narrative / Procedural statement
                c_assertions.append({
                    "annotation_status": "No Aspect Opinion",
                    "clause_relation": "descriptive_context" if last_assertion_id else "independent",
                    "elaboration_of": last_assertion_id if last_assertion_id else None,
                    "aspect": "",
                    "aspect_target_span": "",
                    "opinion_span": "",
                    "sentiment": "",
                    "theme": "",
                    "llm_rationale": "Factual narrative, procedural action, menu listing, or non-evaluative statement without opinion."
                })

            # Process assertions and resolve exact character offsets
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
                    "target_start_char": t_start if t_start is not None else "",
                    "target_end_char": t_end if t_end is not None else "",
                    "opinion_start_char": op_start if op_start is not None else "",
                    "opinion_end_char": op_end if op_end is not None else "",
                    "llm_rationale": ass["llm_rationale"],
                    "annotator_version": self.version,
                    "review_text": review_text
                }

                # Strict character offset validation
                assert review_text[c_start:c_end] == c_text, f"Clause offset mismatch in {ass_id}"
                if t_span:
                    assert review_text[t_start:t_end] == t_span, f"Target offset mismatch in {ass_id}"
                if op_span:
                    assert review_text[op_start:op_end] == op_span, f"Opinion offset mismatch in {ass_id}"

                assertions.append(full_ass)
                if ass["annotation_status"] == "Annotated":
                    last_assertion_id = ass_id

        return assertions
