import pandas as pd
from v2_annotator_engine import V2Annotator

manifest = pd.read_csv(r'annotations\pilot_sample_manifest.csv', keep_default_na=False)
annotator = V2Annotator()

count = 0
for _, r in manifest.iterrows():
    recs = annotator.annotate(r['review_id'], r['original_row_id'], r['establishment_id'], r['star_rating'], r['review_timestamp'], r['review_text'])
    for rec in recs:
        t = rec['aspect_target_span']
        ts = rec['target_start_char']
        op = rec['opinion_span']
        ops = rec['opinion_start_char']
        if t and (ts is None or pd.isna(ts)):
            print(f"Target offset missing in {rec['assertion_id']}: target={repr(t)} in clause={repr(rec['clause_text'])}")
            count += 1
        if op and (ops is None or pd.isna(ops)):
            print(f"Opinion offset missing in {rec['assertion_id']}: opinion={repr(op)} in clause={repr(rec['clause_text'])}")
            count += 1

print(f"Total offset missing issues: {count}")
