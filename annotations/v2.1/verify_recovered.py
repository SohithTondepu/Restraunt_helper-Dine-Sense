import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv(r"annotations\v2.1\pilot_v2.1_annotations_100.csv", keep_default_na=False)

check_ids = ['REV_00457_C03', 'REV_00641_C03', 'REV_01169_C01', 'REV_01902_C01', 'REV_02612_C05', 'REV_02612_C02', 'REV_03108_C11', 'REV_03108_C26', 'REV_03272_C04']

print("Checking previously misclassified clauses in new v2.1 output:")
for cid in check_ids:
    sub = df[df['clause_id'].str.startswith(cid)]
    for _, r in sub.iterrows():
        print(f"[{r['assertion_id']}] Clause: {repr(r['clause_text'])}")
        print(f"   Status: {r['annotation_status']} | Aspect: {r['aspect']} | Target: {repr(r['aspect_target_span'])} | Opinion: {repr(r['opinion_span'])} | Sent: {r['sentiment']}")
        print(f"   Rationale: {r['llm_rationale']}\n")
