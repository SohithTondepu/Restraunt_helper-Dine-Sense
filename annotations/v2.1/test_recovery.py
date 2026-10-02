import sys
import pandas as pd
import re

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv(r"annotations\v2.1\pilot_v2.1_annotations_100.csv", keep_default_na=False)

no_aspect_df = df[df['annotation_status'] == 'No Aspect Opinion']

# Broader lexical categories
FOOD_TERMS = r'\b(food|biryani|chicken|pizza|pasta|mandi|starters|desserts|sweets|grill|fish|rice|noodles|salad|sauce|taste|flavor|flavour|tasty|delicious|bland|fresh|stale|oily|chewy|burnt|raw|cold|spicy|sweet|sour|quantity|portion)\b'
SERVICE_TERMS = r'\b(service|services|sevice|waiter|staff|delivery|deliver|manager|polite|rude|slow|quick|fast|delay|wait|waiting|order|ordered)\b'
AMBIENCE_TERMS = r'\b(ambience|ambiance|atmosphere|seating|seats|view|music|decor|interior|lighting|noise|cleanliness|hygiene|dirty|clean|crowded|cozy|bistro)\b'
PRICE_TERMS = r'\b(price|prices|pricing|cost|costly|expensive|cheap|charge|worth|wallet|money|bill|value)\b'
GENERAL_TERMS = r'\b(experience|place|restaurant|outlet|hotel|recommend|visit|disappoint|disappointment|worst|amazing|awesome|love|loved|hate|pathetic|superb|terrible|horrible)\b'

misclassified = []
for idx, r in no_aspect_df.iterrows():
    c_text = r['clause_text']
    c_low = c_text.lower()
    
    # Check if there is an evaluative indicator
    has_eval = re.search(r'\b(amazing|awesome|good|bad|worst|bland|fresh|stale|oily|tasty|delicious|quick|slow|polite|rude|worth|disappoint|hate|superb|pathetic|costly|expensive|cheap|high|low|less|clean|dirty|cold|hot|burn|chewy|tender)\b', c_low)
    
    if has_eval:
        matched_aspect = []
        if re.search(FOOD_TERMS, c_low): matched_aspect.append("Food")
        if re.search(SERVICE_TERMS, c_low): matched_aspect.append("Service")
        if re.search(AMBIENCE_TERMS, c_low): matched_aspect.append("Ambience")
        if re.search(PRICE_TERMS, c_low): matched_aspect.append("Price / Value")
        if re.search(GENERAL_TERMS, c_low): matched_aspect.append("General Experience")
        
        if matched_aspect:
            misclassified.append((r['assertion_id'], r['clause_text'], matched_aspect, has_eval.group(0)))

print(f"Total verified misclassified evaluative clauses in No Aspect Opinion: {len(misclassified)}")
print("\nSample recovered assertions with clear aspects:")
for ass_id, text, aspects, op in misclassified[:15]:
    print(f"[{ass_id}] -> Opinion: '{op}' | Likely Aspect(s): {aspects}")
    print(f"   Text: {repr(text)}")
