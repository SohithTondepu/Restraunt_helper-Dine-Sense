import pandas as pd

df = pd.read_csv(r'D:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews\annotations\batch_1900\annotations_1900.csv')
nr = df[df['annotation_status'] == 'Needs Review']
print(f"Total Needs Review: {len(nr)}")

for idx, r in nr.iterrows():
    print(f"{r['assertion_id']} | {r['clause_id']} | {repr(r['clause_text'])}")
