import re
import sys
from typing import List, Dict, Any, Optional, Tuple

class AspectAnnotator:
    """
    High-precision pre-annotator for establishment reviews.
    Extracts aspect-opinion assertions from segmented clauses while adhering strictly
    to the DineSense AI annotation schema, taxonomy, and verification guidelines.
    """

    # Controlled Aspect Categories
    ASPECT_FOOD = "Food / Dining"
    ASPECT_SERVICE = "Staff / Service"
    ASPECT_FACILITIES = "Facilities / Amenities"
    ASPECT_PRICE = "Price / Value"
    ASPECT_CLEANLINESS = "Cleanliness"
    ASPECT_LOCATION = "Location"
    ASPECT_BOOKING = "Booking / Check-in / Check-out"
    ASPECT_SAFETY = "Safety / Security"
    ASPECT_OTHER = "Other"

    # Lexicon patterns for targets
    FOOD_TARGETS = [
        r'\bbiriyani\b', r'\bbiryani\b', r'\bshawarma\b', r'\bchicken\b', r'\bpulao\b', r'\bmutton\b',
        r'\bfish\b', r'\bprawns?\b', r'\bstarters?\b', r'\bdesserts?\b', r'\bice\s*cream\b', r'\bshakes?\b',
        r'\bdrinks?\b', r'\bbeverages?\b', r'\bbeer\b', r'\bvodka\b', r'\bcocktails?\b', r'\bmocktails?\b',
        r'\brotis?\b', r'\bnaans?\b', r'\bcurry\b', r'\bcurries\b', r'\bpaneer\b', r'\bdosa\b', r'\bidli\b',
        r'\bsambar\b', r'\bchaat\b', r'\bparanthas?\b', r'\bsizzlers?\b', r'\bburgers?\b', r'\bpizza\b',
        r'\bsalad\b', r'\bdressing\b', r'\bsauce\b', r'\bbuffet\b', r'\bspread\b', r'\bdishes\b', r'\bitem[s]?\b',
        r'\bfood\b', r'\btaste\b', r'\bflavor[s]?\b', r'\bflavour[s]?\b', r'\bportion[s]?\b', r'\bquantity\b',
        r'\bmenu\b', r'\bmeal\b', r'\blunch\b', r'\bdinner\b', r'\bbrunch\b', r'\bbreakfast\b',
        r'\bappetizers?\b', r'\bsoup[s]?\b', r'\btea\b', r'\bchai\b', r'\bcoffee\b', r'\bbread\b',
        r'\bveg\b', r'\bnon[-\s]?veg\b', r'\bsweets?\b', r'\bchutney\b', r'\bgravy\b'
    ]

    SERVICE_TARGETS = [
        r'\bstaff\b', r'\bservice\b', r'\bservices\b', r'\bwaiter[s]?\b', r'\bwaitress\b', r'\bmanager\b',
        r'\bserver[s]?\b', r'\bcaptain\b', r'\bcrew\b', r'\bpersonnel\b', r'\bpeople\b', r'\bboys\b',
        r'\battendant\b', r'\bhospitality\b', r'\bordering\b', r'\border[s]?\b', r'\bdelivery\b',
        r'\bmanagement\b', r'\bteam\b', r'\bguy[s]?\b'
    ]

    FACILITIES_TARGETS = [
        r'\bambience\b', r'\bambiance\b', r'\batmosphere\b', r'\benvironment\b', r'\bvibe[s]?\b',
        r'\binterior[s]?\b', r'\bdecor\b', r'\bdecoration\b', r'\bseating\b', r'\bseats?\b', r'\btables?\b',
        r'\bchairs?\b', r'\bmusic\b', r'\bsongs?\b', r'\bdj\b', r'\bair\s*condition(?:ing|er)?\b', r'\bac\b',
        r'\blighting\b', r'\bsmoking\s*area\b', r'\broof\s*top\b', r'\bplace\b', r'\bspace\b',
        r'\brestroom[s]?\b', r'\bwashroom[s]?\b', r'\btoilet[s]?\b', r'\bparking\b', r'\bvalet\b',
        r'\bscreening\b', r'\bscreen\b'
    ]

    PRICE_TARGETS = [
        r'\bprice[s]?\b', r'\bpricing\b', r'\bcost\b', r'\brate[s]?\b', r'\bbill\b', r'\bcharge[s]?\b',
        r'\bvalue\b', r'\bmoney\b', r'\bwallet\b', r'\btaxes?\b', r'\bdiscount[s]?\b', r'\boffers?\b'
    ]

    CLEANLINESS_TARGETS = [
        r'\bcleanliness\b', r'\bhygiene\b', r'\bhair\b', r'\binsects?\b', r'\bcockroach\b', r'\bfly\b',
        r'\bflies\b', r'\bplates?\b', r'\bglasses?\b', r'\bspoons?\b', r'\bcutlery\b', r'\bfork\b'
    ]

    LOCATION_TARGETS = [
        r'\blocation\b', r'\barea\b', r'\broad\b', r'\bspot\b', r'\bview\b'
    ]

    BOOKING_TARGETS = [
        r'\breservation\b', r'\bbooking\b', r'\bqueue\b', r'\bwaitlist\b'
    ]

    SAFETY_TARGETS = [
        r'\bsafety\b', r'\bsecurity\b', r'\bguard\b', r'\bfood\s*poisoning\b', r'\bsick\b', r'\bhospital\b'
    ]

    # Opinion patterns
    POS_OPINIONS = [
        r'\bdelicious\b', r'\byummy\b', r'\bawesome\b', r'\bamazing\b', r'\bexcellent\b', r'\bfantastic\b',
        r'\bgreat\b', r'\bvery\s+good\b', r'\btoo\s+good\b', r'\bquite\s+good\b', r'\bgood\b', r'\btasty\b',
        r'\btooo\s+good\b', r'\bso\s+good\b', r'\blove[d]?\b', r'\bliked\b', r'\blike\b', r'\bperfection\b',
        r'\bperfect\b', r'\bwell\s+cooked\b', r'\bfresh\b', r'\bcrispy\b', r'\bjuicy\b', r'\btender\b',
        r'\bflavourful\b', r'\bflavorful\b', r'\bmouth\s*watering\b', r'\bsuperb\b', r'\bwonderful\b',
        r'\bfriendly\b', r'\bcourteous\b', r'\bhelpful\b', r'\bprompt\b', r'\bquick\b', r'\bfast\b',
        r'\bpolite\b', r'\battentive\b', r'\bcooperative\b', r'\bpleasant\b', r'\bpeaceful\b',
        r'\bbeautiful\b', r'\blovely\b', r'\bcozy\b', r'\bcost\s+effective\b', r'\baffordable\b',
        r'\bworth\b', r'\bvalue\s+for\s+money\b', r'\breasonable\b', r'\bpocket\s+friendly\b',
        r'\bcheap\b', r'\bhygienic\b', r'\bclean\b', r'\bneat\b', r'\bspotless\b', r'\bwell\s+maintained\b',
        r'\brecommend\b', r'\bmust\s+visit\b', r'\bmust\s+try\b', r'\bdecent\b', r'\bnice\b',
        r'\bnever\s+disappoint[s]?\b', r'\bnot\s+disappoint[s]?\b'
    ]

    NEG_OPINIONS = [
        r'\bterrible\b', r'\bhorrible\b', r'\bworst\b', r'\bpathetic\b', r'\bvery\s+bad\b', r'\bbad\b',
        r'\bpoor\b', r'\btasteless\b', r'\bbland\b', r'\boily\b', r'\btoo\s+oily\b', r'\bvery\s+oily\b',
        r'\bgreasy\b', r'\bsalty\b', r'\btoo\s+salty\b', r'\bstale\b', r'\bcold\b', r'\blukewarm\b',
        r'\braw\b', r'\bundercooked\b', r'\bovercooked\b', r'\bburnt\b', r'\bhard\b', r'\brubber\b',
        r'\brubbery\b', r'\bchewy\b', r'\bdry\b', r'\bspoiled\b', r'\brotten\b', r'\bsour\b',
        r'\bsmelly\b', r'\bhorrid\b', r'\bdisappointing\b', r'\bdisappointment\b', r'\bdisappointed\b',
        r'\brude\b', r'\barrogant\b', r'\bslow\b', r'\bvery\s+slow\b', r'\btoo\s+slow\b', r'\blate\b',
        r'\bdelay[ed]?\b', r'\binattentive\b', r'\bignored\b', r'\bcareless\b', r'\bunprofessional\b',
        r'\bexpensive\b', r'\btoo\s+expensive\b', r'\bvery\s+expensive\b', r'\boverpriced\b',
        r'\bcostly\b', r'\bnot\s+worth\b', r'\bwaste\s+of\s+money\b', r'\bloot\b', r'\brip\s*off\b',
        r'\bdirty\b', r'\bunhygienic\b', r'\bfilthy\b', r'\bsmell\b', r'\bstinking\b', r'\bmessy\b',
        r'\bnoisy\b', r'\bcrowded\b', r'\bsuffocating\b', r'\bcongested\b', r'\bcramped\b',
        r'\buncomfortable\b', r'\bnot\s+good\b', r'\bnot\s+tasty\b', r'\bnot\s+fresh\b',
        r'\bnot\s+clean\b', r'\bnot\s+friendly\b', r'\bnot\s+worth\s+it\b', r'\bunable\s+to\s+eat\b',
        r'\bhate\b', r'\bhated\b', r'\bdislike\b', r'\bavoid\b', r'\bnever\s+again\b'
    ]

    NEU_OPINIONS = [
        r'\baverage\b', r'\bokay\b', r'\bok\b', r'\bmediocre\b', r'\bso\s*so\b', r'\bnormal\b',
        r'\bmoderate\b', r'\bjust\s+fine\b', r'\bfine\b'
    ]

    def __init__(self, annotator_version: str = "gemini-3.8-flash-preannotator-v1.0"):
        self.annotator_version = annotator_version

    def _find_spans(self, text: str, patterns: List[str]) -> List[Tuple[int, int, str]]:
        found = []
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                found.append((m.start(), m.end(), text[m.start():m.end()]))
        # Deduplicate and sort
        found = sorted(list(set(found)), key=lambda x: (x[0], -(x[1] - x[0])))
        # Remove nested subsets
        filtered = []
        for span in found:
            if not any(span[0] >= existing[0] and span[1] <= existing[1] for existing in filtered):
                filtered.append(span)
        return filtered

    def _determine_aspect(self, target_text: str, clause_text: str) -> Tuple[str, str]:
        """Maps target and clause context to aspect category and default theme."""
        t_low = target_text.lower()
        c_low = clause_text.lower()

        # Food checks
        for pat in self.FOOD_TARGETS:
            if re.search(pat, t_low) or (not t_low and re.search(pat, c_low)):
                if any(w in t_low or w in c_low for w in ['drink', 'beer', 'vodka', 'cocktail', 'mocktail', 'beverage', 'tea', 'coffee', 'chai']):
                    return self.ASPECT_FOOD, "Drink / Beverage quality"
                if any(w in t_low or w in c_low for w in ['menu', 'variety', 'options', 'choices']):
                    return self.ASPECT_FOOD, "Menu variety"
                if any(w in t_low or w in c_low for w in ['portion', 'quantity', 'size']):
                    return self.ASPECT_FOOD, "Portion size"
                if any(w in t_low or w in c_low for w in ['temperature', 'cold', 'hot', 'warm']):
                    return self.ASPECT_FOOD, "Food temperature"
                if any(w in t_low or w in c_low for w in ['taste', 'flavor', 'flavour', 'spicy', 'sweet', 'salt']):
                    return self.ASPECT_FOOD, "Taste / Flavor"
                return self.ASPECT_FOOD, "Food quality"

        # Service checks
        for pat in self.SERVICE_TARGETS:
            if re.search(pat, t_low) or (not t_low and re.search(pat, c_low)):
                if any(w in c_low for w in ['quick', 'fast', 'slow', 'late', 'delay', 'wait', 'time']):
                    return self.ASPECT_SERVICE, "Service speed / Wait time"
                if any(w in c_low for w in ['polite', 'rude', 'friendly', 'courteous', 'arrogant', 'attitude', 'helpful', 'humble']):
                    return self.ASPECT_SERVICE, "Staff courtesy"
                if any(w in c_low for w in ['wrong', 'missing', 'mistake', 'accuracy']):
                    return self.ASPECT_SERVICE, "Order accuracy"
                return self.ASPECT_SERVICE, "Staff courtesy"

        # Facilities checks
        for pat in self.FACILITIES_TARGETS:
            if re.search(pat, t_low) or (not t_low and re.search(pat, c_low)):
                if any(w in t_low or w in c_low for w in ['music', 'song', 'dj', 'noise', 'sound']):
                    return self.ASPECT_FACILITIES, "Music / Noise level"
                if any(w in t_low or w in c_low for w in ['seating', 'seat', 'chair', 'table', 'sofa', 'comfort']):
                    return self.ASPECT_FACILITIES, "Seating / Comfort"
                if any(w in t_low or w in c_low for w in ['ac', 'air condition', 'ventilation', 'suffocat']):
                    return self.ASPECT_FACILITIES, "Air conditioning / Ventilation"
                if any(w in t_low or w in c_low for w in ['parking', 'valet']):
                    return self.ASPECT_FACILITIES, "Parking"
                if any(w in t_low or w in c_low for w in ['washroom', 'restroom', 'toilet']):
                    return self.ASPECT_FACILITIES, "Washroom / Restroom facilities"
                return self.ASPECT_FACILITIES, "Ambience / Atmosphere"

        # Price checks
        for pat in self.PRICE_TARGETS:
            if re.search(pat, t_low) or (not t_low and re.search(pat, c_low)):
                if any(w in c_low for w in ['expensive', 'costly', 'overpriced', 'cheap', 'price']):
                    return self.ASPECT_PRICE, "Pricing / Overpriced"
                return self.ASPECT_PRICE, "Value for money"

        # Cleanliness checks
        for pat in self.CLEANLINESS_TARGETS:
            if re.search(pat, t_low) or (not t_low and re.search(pat, c_low)):
                return self.ASPECT_CLEANLINESS, "General hygiene"

        # Location checks
        for pat in self.LOCATION_TARGETS:
            if re.search(pat, t_low) or (not t_low and re.search(pat, c_low)):
                return self.ASPECT_LOCATION, "Accessibility / Ease of finding"

        # Booking checks
        for pat in self.BOOKING_TARGETS:
            if re.search(pat, t_low) or (not t_low and re.search(pat, c_low)):
                return self.ASPECT_BOOKING, "Reservation / Table booking"

        # Safety checks
        for pat in self.SAFETY_TARGETS:
            if re.search(pat, t_low) or (not t_low and re.search(pat, c_low)):
                return self.ASPECT_SAFETY, "Food safety / Health"

        return self.ASPECT_OTHER, "Overall experience"

    def annotate_clause(
        self,
        review_id: str,
        clause_id: str,
        clause_text: str,
        clause_start_char: int,
        clause_end_char: int,
        review_text: str
    ) -> List[Dict[str, Any]]:
        """
        Extracts one or more aspect-opinion assertions from a clause.
        Returns a list of assertion dictionaries.
        """
        c_low = clause_text.lower()

        # Step 1: Detect opinions
        pos_ops = self._find_spans(clause_text, self.POS_OPINIONS)
        neg_ops = self._find_spans(clause_text, self.NEG_OPINIONS)
        neu_ops = self._find_spans(clause_text, self.NEU_OPINIONS)

        # Check for non-evaluative activity / purely descriptive / procedural text
        # (e.g. "We went with family", "had Saturday lunch", "One can also chill with friends and or parents")
        has_opinions = bool(pos_ops or neg_ops or neu_ops)

        if not has_opinions:
            # Check if there is an explicit evaluative adjective or verb missing from lexicon
            # If purely descriptive, output No Aspect Opinion
            return [{
                "assertion_id": f"{clause_id}_A01",
                "clause_id": clause_id,
                "clause_text": clause_text,
                "clause_start_char": clause_start_char,
                "clause_end_char": clause_end_char,
                "annotation_status": "No Aspect Opinion",
                "aspect": "",
                "aspect_target_span": "",
                "target_start_char": None,
                "target_end_char": None,
                "opinion_span": "",
                "opinion_start_char": None,
                "opinion_end_char": None,
                "sentiment": "",
                "theme": "",
                "llm_rationale": "Descriptive, procedural, or observational statement without an evaluative aspect opinion.",
                "annotator_version": self.annotator_version
            }]

        # Step 2: Detect Targets across aspect lexicons
        all_target_patterns = (
            self.FOOD_TARGETS + self.SERVICE_TARGETS + self.FACILITIES_TARGETS +
            self.PRICE_TARGETS + self.CLEANLINESS_TARGETS + self.LOCATION_TARGETS +
            self.BOOKING_TARGETS + self.SAFETY_TARGETS
        )
        targets = self._find_spans(clause_text, all_target_patterns)

        # Step 3: Check for multiple distinct aspects in clause (e.g., "The staff is friendly and ambiance is awesome")
        # If multiple targets and multiple opinions exist, pair them
        assertions = []
        paired_opinions = set()
        paired_targets = set()

        # Check coordinate structure: Target1 + Opinion1 and Target2 + Opinion2
        if len(targets) >= 2 and (len(pos_ops) + len(neg_ops) + len(neu_ops)) >= 2:
            # Attempt to pair nearest target to each opinion
            all_ops = [(s, e, t, "Positive") for s, e, t in pos_ops] + \
                      [(s, e, t, "Negative") for s, e, t in neg_ops] + \
                      [(s, e, t, "Neutral") for s, e, t in neu_ops]
            all_ops = sorted(all_ops, key=lambda x: x[0])

            for op_s, op_e, op_text, sent in all_ops:
                # Find closest target to this opinion
                best_t = None
                best_dist = 9999
                for t_s, t_e, t_text in targets:
                    dist = abs(op_s - t_s)
                    if dist < best_dist and (t_s, t_e) not in paired_targets:
                        best_dist = dist
                        best_t = (t_s, t_e, t_text)

                if best_t:
                    paired_targets.add((best_t[0], best_t[1]))
                    paired_opinions.add((op_s, op_e))
                    t_str = best_t[2]
                    aspect, theme = self._determine_aspect(t_str, clause_text)
                    
                    t_abs_start = clause_start_char + best_t[0]
                    t_abs_end = clause_start_char + best_t[1]
                    op_abs_start = clause_start_char + op_s
                    op_abs_end = clause_start_char + op_e

                    assertions.append({
                        "aspect": aspect,
                        "theme": theme,
                        "aspect_target_span": t_str,
                        "target_start_char": t_abs_start,
                        "target_end_char": t_abs_end,
                        "opinion_span": op_text,
                        "opinion_start_char": op_abs_start,
                        "opinion_end_char": op_abs_end,
                        "sentiment": sent,
                        "annotation_status": "Annotated",
                        "llm_rationale": f"Explicit aspect '{aspect}' with target '{t_str}' and opinion '{op_text}' ({sent})."
                    })

        # If not paired or single assertion
        if not assertions:
            # Pick primary opinion
            primary_op = None
            sentiment = "Positive"
            if neg_ops:
                primary_op = neg_ops[0]
                sentiment = "Negative"
            elif pos_ops:
                primary_op = pos_ops[0]
                sentiment = "Positive"
            elif neu_ops:
                primary_op = neu_ops[0]
                sentiment = "Neutral"

            # Check if there is contrast or conflict in the single clause
            status = "Annotated"
            if pos_ops and neg_ops:
                status = "Needs Review"
                sentiment = "Unclear"

            # Pick target
            t_str = ""
            t_abs_start = None
            t_abs_end = None
            if targets:
                # Find closest target to primary opinion
                best_t = targets[0]
                if primary_op:
                    best_t = min(targets, key=lambda t: abs(t[0] - primary_op[0]))
                t_str = best_t[2]
                t_abs_start = clause_start_char + best_t[0]
                t_abs_end = clause_start_char + best_t[1]
            elif "cost effective" in c_low or "expensive" in c_low or "pocket friendly" in c_low:
                # Check for explicit meal/session mentioned in clause as described in correction 3
                for meal in ['lunch', 'dinner', 'saturday lunch', 'brunch', 'breakfast', 'buffet', 'food']:
                    idx = c_low.find(meal)
                    if idx != -1:
                        t_str = clause_text[idx:idx+len(meal)]
                        t_abs_start = clause_start_char + idx
                        t_abs_end = t_abs_start + len(meal)
                        break

            aspect, theme = self._determine_aspect(t_str, clause_text)

            op_str = primary_op[2] if primary_op else ""
            op_abs_start = (clause_start_char + primary_op[0]) if primary_op else None
            op_abs_end = (clause_start_char + primary_op[1]) if primary_op else None

            rationale = f"Evaluative opinion '{op_str}' indicating {sentiment} sentiment towards aspect '{aspect}'"
            if t_str:
                rationale += f" targeting '{t_str}'."
            else:
                rationale += " with implicit target."

            assertions.append({
                "aspect": aspect,
                "theme": theme,
                "aspect_target_span": t_str,
                "target_start_char": t_abs_start,
                "target_end_char": t_abs_end,
                "opinion_span": op_str,
                "opinion_start_char": op_abs_start,
                "opinion_end_char": op_abs_end,
                "sentiment": sentiment,
                "annotation_status": status,
                "llm_rationale": rationale
            })

        # Add IDs and common fields
        result = []
        for idx, ass in enumerate(assertions, 1):
            ass_dict = {
                "assertion_id": f"{clause_id}_A{idx:02d}",
                "clause_id": clause_id,
                "clause_text": clause_text,
                "clause_start_char": clause_start_char,
                "clause_end_char": clause_end_char,
                **ass,
                "annotator_version": self.annotator_version
            }
            result.append(ass_dict)

        return result
