import sys
import json
import pandas as pd
import re

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv(r"annotations\v2.1\pilot_v2.1_annotations_100.csv", keep_default_na=False)

no_aspect_df = df[df['annotation_status'] == 'No Aspect Opinion'].copy()
print(f"Total 'No Aspect Opinion' clauses in v2.1: {len(no_aspect_df)}")

# Let's inspect a structured sample (or all 391) and classify them based on linguistic analysis
# Categories:
# A: Missed Specific Aspect (Food, Service, Ambience, Price / Value)
# B: Missed General Experience (Holistic restaurant appraisal, repeat intent, recommendation)
# C: Truly No Aspect Opinion (Factual, descriptive, procedural, dish listing, boilerplate)
# D: Ambiguous / Borderline (Needs Review)

FOOD_INDICATORS = r'\b(food|biryani|chicken|pizza|pasta|mandi|dessert|curry|roti|taste|flavor|flavour|tasty|delicious|bland|fresh|stale|oily|chewy|burnt|raw|cold|spicy|sweet|sour|quantity|portion|gravy|sambar|idli|dosa|chutney|soup|tea|coffee|drinks|beer|cocktail|smoothie|shake|parantha|halwa|paneer|prawns|fish|mutton|rolls|starter|starters|buffet|appetizer|appetizers)\b'
SERVICE_INDICATORS = r'\b(service|services|sevice|waiter|waiters|waitress|staff|server|servers|delivery|delivered|order|ordering|management|hospitality|attentive|polite|rude|slow|quick|fast|delay|waiting|prompt|courteous|ignored|mix-up|refund)\b'
AMBIENCE_INDICATORS = r'\b(ambience|ambiance|atmosphere|vibe|vibes|interior|interiors|decor|decoration|seating|seats|tables|chairs|music|dj|noise|view|lighting|ac|cleanliness|hygiene|dirty|clean|restroom|washroom|cozy|bistro|crowded)\b'
PRICE_INDICATORS = r'\b(price|prices|pricing|cost|costly|rate|rates|bill|charge|charges|value|money|wallet|expensive|cheap|affordable|worth|pocket friendly|overpriced)\b'
GENERAL_INDICATORS = r'\b(place|restaurant|restro|outlet|hotel|visit|experience|recommend|must visit|loved|disappoint|disappointment|favourite|favorite|satisfied|satisfying|terrible|awesome|amazing|superb|horrible|avoid|come back|again|happy soul|happy tummy|overall)\b'

EVALUATIVE_CUES = r'\b(good|bad|great|nice|awesome|amazing|superb|delicious|tasty|terrible|horrible|pathetic|worst|poor|best|loved|disappoint|disappointment|disappointed|satisfying|happy|unhappy|fine|average|bland|fresh|stale|oily|slow|quick|fast|polite|rude|cheap|expensive|costly|worth|crowded|cozy|dirty|clean|neat|cold|hot|burnt|chewy|tender|soft|hard|heavy|light|appealing|refreshing|favourite|favorite|recommend|special)\b'

findings = []

for idx, r in no_aspect_df.iterrows():
    c_text = r['clause_text']
    c_low = c_text.lower()
    rev_id = r['review_id']
    c_id = r['clause_id']

    # Check for boilerplate or pure narrative
    is_boilerplate = bool(re.search(r'\b(do follow us|instagram|zomato gold|review#2|at that time|while coming back|used to spot|ordered through zomato|ordered from swiggy|got to know|rate\s*pasta\s*5/2\.5)\b', c_low))
    has_eval = bool(re.search(EVALUATIVE_CUES, c_low))

    # Detailed classification
    classification = "C: Truly No Aspect Opinion"
    proposed_aspect = ""
    proposed_opinion = ""
    notes = ""

    if is_boilerplate and not has_eval:
        classification = "C: Truly No Aspect Opinion"
        notes = "Boilerplate / social handle / platform metadata"
    elif not has_eval and not re.search(r'\b(competi|ultimate pleasure|degrad|authent|finger lick|eye catch|special)\b', c_low):
        classification = "C: Truly No Aspect Opinion"
        notes = "Factual narrative, procedural step, or plain dish list without evaluative words"
    else:
        # Check if it has genuine evaluative opinion
        # Is it specific aspect?
        if re.search(FOOD_INDICATORS, c_low) and re.search(EVALUATIVE_CUES, c_low):
            classification = "A: Missed Specific Aspect (Food)"
            proposed_aspect = "Food"
            notes = "Contains food target and evaluative attribute"
        elif re.search(SERVICE_INDICATORS, c_low) and re.search(EVALUATIVE_CUES, c_low):
            classification = "A: Missed Specific Aspect (Service)"
            proposed_aspect = "Service"
            notes = "Contains service target and evaluative attribute"
        elif re.search(AMBIENCE_INDICATORS, c_low) and re.search(EVALUATIVE_CUES, c_low):
            classification = "A: Missed Specific Aspect (Ambience)"
            proposed_aspect = "Ambience"
            notes = "Contains ambience target and evaluative attribute"
        elif re.search(PRICE_INDICATORS, c_low) and re.search(EVALUATIVE_CUES, c_low):
            classification = "A: Missed Specific Aspect (Price / Value)"
            proposed_aspect = "Price / Value"
            notes = "Contains pricing target and evaluative attribute"
        elif re.search(GENERAL_INDICATORS, c_low) and re.search(EVALUATIVE_CUES, c_low):
            classification = "B: Missed General Experience"
            proposed_aspect = "General Experience"
            notes = "Evaluates overall restaurant, repeat intent, or holistic visit"
        elif re.search(r'\b(ok|okay|fine|average|special|different|normal|not allowed|competi|ultimate pleasure)\b', c_low):
            classification = "D: Ambiguous / Borderline (Needs Review)"
            notes = "Context-dependent, mixed, or borderline evaluative statement"
        else:
            classification = "C: Truly No Aspect Opinion"
            notes = "Descriptive context"

    findings.append({
        "review_id": rev_id,
        "clause_id": c_id,
        "clause_text": c_text,
        "classification": classification,
        "proposed_aspect": proposed_aspect,
        "notes": notes
    })

findings_df = pd.DataFrame(findings)
print("\nClassification breakdown across all 391 'No Aspect Opinion' clauses:")
print(findings_df['classification'].value_counts())
