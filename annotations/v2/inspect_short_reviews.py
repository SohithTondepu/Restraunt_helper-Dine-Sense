import pandas as pd

manifest = pd.read_csv(r"annotations\pilot_sample_manifest.csv", keep_default_na=False)

print("Checking short / purely procedural reviews in 100 pilot:")
for i, r in manifest.iterrows():
    text = r['review_text']
    # Check if there are no typical sentiment words at all
    words = text.lower().split()
    if len(words) <= 10:
        print(f"[{r['review_id']}] ({r['star_rating']}*) {repr(text)}")
