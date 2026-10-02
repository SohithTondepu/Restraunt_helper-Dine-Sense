import pandas as pd
import json
import numpy as np
import shutil
import os
from validate_resolutions import resolutions

# 1. Update annotations_1900.csv
df_1900 = pd.read_csv(r'D:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews\annotations\batch_1900\annotations_1900.csv')

updated_count = 0
for idx, row in df_1900.iterrows():
    aid = row['assertion_id']
    if aid in resolutions:
        res = resolutions[aid]
        c_start = int(row['clause_start_char'])
        rev_text = str(row['review_text'])
        
        df_1900.at[idx, 'aspect'] = res['aspect']
        df_1900.at[idx, 'sentiment'] = res['sentiment']
        df_1900.at[idx, 'theme'] = res['theme']
        df_1900.at[idx, 'llm_rationale'] = res['rationale']
        df_1900.at[idx, 'annotation_status'] = 'Annotated'
        
        op = res['opinion_span']
        df_1900.at[idx, 'opinion_span'] = op
        op_start = rev_text.find(op, c_start)
        op_end = op_start + len(op)
        df_1900.at[idx, 'opinion_start_char'] = float(op_start)
        df_1900.at[idx, 'opinion_end_char'] = float(op_end)
        
        tg = res['target_span']
        if tg != '':
            df_1900.at[idx, 'aspect_target_span'] = tg
            tg_start = rev_text.find(tg, c_start)
            tg_end = tg_start + len(tg)
            df_1900.at[idx, 'target_start_char'] = float(tg_start)
            df_1900.at[idx, 'target_end_char'] = float(tg_end)
        else:
            df_1900.at[idx, 'aspect_target_span'] = np.nan
            df_1900.at[idx, 'target_start_char'] = np.nan
            df_1900.at[idx, 'target_end_char'] = np.nan
            
        updated_count += 1

print(f"Updated {updated_count} rows in df_1900.")
df_1900.to_csv(r'D:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews\annotations\batch_1900\annotations_1900.csv', index=False, encoding='utf-8')
print("Saved annotations_1900.csv")

# 2. Update annotations_1900.jsonl
with open(r'D:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews\annotations\batch_1900\annotations_1900.jsonl', 'w', encoding='utf-8') as f:
    for _, row in df_1900.iterrows():
        d = row.to_dict()
        for k, v in d.items():
            if pd.isna(v):
                d[k] = None
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
print("Saved annotations_1900.jsonl")

# 3. Update verified_annotations_template_1900.csv
df_1900.to_csv(r'D:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews\annotations\batch_1900\verified_annotations_template_1900.csv', index=False, encoding='utf-8')
print("Saved verified_annotations_template_1900.csv")

# 4. Update combined 2,000 dataset
# Load pilot v2.2 and combine with updated df_1900
df_pilot = pd.read_csv(r'D:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews\annotations\v2.2\pilot_v2.2_annotations_100.csv')
df_2000 = pd.concat([df_pilot, df_1900], ignore_index=True)
print(f"Combined df_2000 has {len(df_2000)} rows.")

# Save annotations_2000_final.csv
final_csv_path = r'D:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews\annotations\combined_2000\annotations_2000_final.csv'
df_2000.to_csv(final_csv_path, index=False, encoding='utf-8')
print(f"Saved {final_csv_path}")

# Try to save annotations_2000.csv directly
orig_csv_path = r'D:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews\annotations\combined_2000\annotations_2000.csv'
try:
    df_2000.to_csv(orig_csv_path, index=False, encoding='utf-8')
    print("Saved annotations_2000.csv successfully!")
except Exception as e:
    print(f"Notice: annotations_2000.csv is currently open in Excel ({e}). Created annotations_2000_final.csv as clean copy.")

# Save annotations_2000.jsonl
with open(r'D:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews\annotations\combined_2000\annotations_2000.jsonl', 'w', encoding='utf-8') as f:
    for _, row in df_2000.iterrows():
        d = row.to_dict()
        for k, v in d.items():
            if pd.isna(v):
                d[k] = None
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
print("Saved annotations_2000.jsonl")

# 5. Full assertion check
assert len(df_1900[df_1900['annotation_status'] == 'Needs Review']) == 0, "Still has Needs Review in 1900!"
assert len(df_2000[df_2000['annotation_status'] == 'Needs Review']) == 0, "Still has Needs Review in 2000!"
print("Status check passed: 0 Needs Review cases remaining across all datasets!")
