import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv(r"annotations\v2.1\pilot_v2.1_annotations_100.csv", keep_default_na=False)

no_aspect_df = df[df['annotation_status'] == 'No Aspect Opinion']
print(f"Total No Aspect Opinion clauses: {len(no_aspect_df)}")

# Look for clauses that contain sentiment or aspect words but got marked as No Aspect Opinion
suspects = []
sentiment_keywords = [
    'good', 'bad', 'great', 'nice', 'delicious', 'tasty', 'worst', 'poor', 'slow', 'quick',
    'fast', 'cold', 'hot', 'oily', 'spicy', 'sweet', 'salt', 'bland', 'fresh', 'stale',
    'expensive', 'cheap', 'cost', 'clean', 'dirty', 'rude', 'polite', 'friendly', 'courteous',
    'loved', 'disappoint', 'awesome', 'amazing', 'fine', 'recommend', 'worth', 'pathetic',
    'degrading', 'improve', 'problem', 'delay', 'wait', 'wrong', 'burn', 'chewy', 'tender'
]

for idx, r in no_aspect_df.iterrows():
    c_low = r['clause_text'].lower()
    matches = [kw for kw in sentiment_keywords if kw in c_low]
    if matches:
        suspects.append((r['review_id'], r['clause_id'], r['clause_text'], matches))

print(f"Found {len(suspects)} suspect clauses containing opinion keywords that were marked 'No Aspect Opinion':\n")
for rev_id, c_id, text, matches in suspects[:30]:
    print(f"[{c_id}] ({rev_id}) matched keywords: {matches}")
    print(f"   Text: {repr(text)}\n")
