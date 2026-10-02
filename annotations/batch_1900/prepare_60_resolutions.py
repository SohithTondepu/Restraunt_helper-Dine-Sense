import pandas as pd
import json

df = pd.read_csv(r'D:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews\annotations\batch_1900\annotations_1900.csv')
nr = df[df['annotation_status'] == 'Needs Review']

rows_info = []
for idx, r in nr.iterrows():
    rows_info.append({
        'assertion_id': r['assertion_id'],
        'clause_id': r['clause_id'],
        'review_id': r['review_id'],
        'clause_text': r['clause_text'],
        'clause_start_char': int(r['clause_start_char']),
        'clause_end_char': int(r['clause_end_char']),
        'review_text': r['review_text']
    })

with open(r'D:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews\annotations\batch_1900\needs_review_info.json', 'w', encoding='utf-8') as f:
    json.dump(rows_info, f, indent=2, ensure_ascii=False)

print(f"Dumped {len(rows_info)} items.")
