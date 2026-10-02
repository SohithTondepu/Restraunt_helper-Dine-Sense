import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv(r'annotations\v2.2\pilot_v2.2_annotations_100.csv', keep_default_na=False)

check_revs = ['REV_00564', 'REV_02328', 'REV_06351', 'REV_07067', 'REV_08213', 'REV_06813']
for r_id in check_revs:
    print(f'=== {r_id} ===')
    sub = df[df['review_id'] == r_id]
    for _, r in sub.iterrows():
        print(f"[{r['assertion_id']}] [{r['clause_start_char']}:{r['clause_end_char']}] ({r['annotation_status']}) Aspect: {r['aspect']} | Target: {repr(r['aspect_target_span'])} | Op: {repr(r['opinion_span'])} | Sent: {r['sentiment']}")
        print(f"   Clause: {repr(r['clause_text'])}")
    print()
