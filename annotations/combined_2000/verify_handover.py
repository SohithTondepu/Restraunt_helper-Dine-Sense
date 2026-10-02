import pandas as pd

path = r'D:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews\annotations\combined_2000\annotations_2000_final.csv'
df = pd.read_csv(path)

print("=== DATASET COUNTS ===")
print("Reviews:", df['review_id'].nunique())
print("Establishments:", df['establishment_id'].nunique())
print("Clauses:", df['clause_id'].nunique())
print("Assertions:", len(df))
print("\nAnnotation Status Breakdown:")
print(df['annotation_status'].value_counts())

print("\nAspect Breakdown:")
print(df['aspect'].value_counts())

print("\nSentiment Breakdown:")
print(df['sentiment'].value_counts())

# Multi-assertion clauses
clause_counts = df['clause_id'].value_counts()
multi_clauses = clause_counts[clause_counts > 1]
print(f"\n=== MULTI-ASSERTION CLAUSES ({len(multi_clauses)} clauses) ===")
for cid, cnt in multi_clauses.items():
    sub = df[df['clause_id'] == cid]
    print(f"\nClause ID: {cid} (Assertions: {cnt})")
    print(f"Clause text: {repr(sub.iloc[0]['clause_text'])}")
    for _, r in sub.iterrows():
        print(f"  -> Assertion ID: {r['assertion_id']} | Aspect: {r['aspect']} | Sentiment: {r['sentiment']} | Target: {repr(r['aspect_target_span'])} | Opinion: {repr(r['opinion_span'])}")

# Mixed assertion
mixed = df[df['sentiment'] == 'Mixed']
print(f"\n=== MIXED SENTIMENT ASSERTION ({len(mixed)} rows) ===")
for _, r in mixed.iterrows():
    print(f"Assertion ID: {r['assertion_id']}")
    print(f"Review ID: {r['review_id']}")
    print(f"Clause ID: {r['clause_id']}")
    print(f"Clause text: {repr(r['clause_text'])}")
    print(f"Aspect: {r['aspect']}")
    print(f"Sentiment: {r['sentiment']}")
    print(f"Target: {repr(r['aspect_target_span'])}")
    print(f"Opinion: {repr(r['opinion_span'])}")
    print(f"Rationale: {r['llm_rationale']}")
