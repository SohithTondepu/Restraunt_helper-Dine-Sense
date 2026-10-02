import pandas as pd

df = pd.read_csv(r"annotations\v2\pilot_v2_annotations_100.csv", keep_default_na=False)

review_stats = df.groupby('review_id').agg(
    total_assertions=('assertion_id', 'count'),
    annotated_count=('annotation_status', lambda s: (s == 'Annotated').sum()),
    no_aspect_count=('annotation_status', lambda s: (s == 'No Aspect Opinion').sum())
).reset_index()

zero_aspect = review_stats[review_stats['annotated_count'] == 0]
print(f"Total reviews in pilot: {len(review_stats)}")
print(f"Reviews with >= 1 aspect assertion: {len(review_stats[review_stats['annotated_count'] > 0])}")
print(f"Reviews with ZERO aspect assertions (100% No Aspect Opinion): {len(zero_aspect)}")

if len(zero_aspect) > 0:
    print("\n--- Zero-Aspect Reviews in Pilot ---")
    for _, r in zero_aspect.iterrows():
        rev_id = r['review_id']
        sample_row = df[df['review_id'] == rev_id].iloc[0]
        print(f"\n[{rev_id}] {sample_row['establishment_id']} ({sample_row['star_rating']}*):")
        print(f"Full Text: {repr(sample_row['review_text'])}")
        clauses = df[df['review_id'] == rev_id]
        print("Clauses:")
        for _, c in clauses.iterrows():
            print(f"  [{c['clause_id']}] Status: {c['annotation_status']} | Text: {repr(c['clause_text'])}")
