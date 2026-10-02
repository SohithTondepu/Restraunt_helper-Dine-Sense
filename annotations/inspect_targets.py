import pandas as pd

df = pd.read_csv(r"annotations\pilot_llm_annotations.csv")
target_revs = ['REV_01800', 'REV_05070', 'REV_06534', 'REV_08836', 'REV_02912']

for r_id in target_revs:
    sub = df[df['review_id'] == r_id]
    print(f"\n==================== {r_id} ({sub['establishment_id'].iloc[0]}) ====================")
    print("Full Review Text:\n", repr(sub['review_text'].iloc[0]))
    print("\nV1 Pilot Annotations:")
    for _, row in sub.iterrows():
        print(f"  [{row['assertion_id']}] Clause: {repr(row['clause_text'])}")
        print(f"      Status: {row['annotation_status']} | Aspect: {row['aspect']} | Theme: {row['theme']}")
        print(f"      Target: {repr(row['aspect_target_span'])} | Opinion: {repr(row['opinion_span'])} | Sentiment: {row['sentiment']}")
        print(f"      Rationale: {row['llm_rationale']}")
