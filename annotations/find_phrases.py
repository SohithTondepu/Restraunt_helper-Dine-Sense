import pandas as pd

df = pd.read_csv(r"annotations\pilot_llm_annotations.csv")
print("Total rows in pilot_llm_annotations:", len(df))
print("Total reviews:", df["review_id"].nunique())

phrases = ["not good", "crispy", "fine-dine", "fine dine", "fine", "MUST TRY", "must try", "delicious", "very bad", "chill", "cold", "oily", "expensive", "cost effective"]
for phrase in phrases:
    matches = df[df["review_text"].str.contains(phrase, case=False, na=False)]
    revs = matches["review_id"].unique().tolist()
    print(f"Phrase '{phrase}': {len(revs)} reviews -> {revs[:5]}")
