import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"d:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews"
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "annotations", "pilot_sample_manifest.csv")
manifest = pd.read_csv(MANIFEST_PATH, keep_default_na=False)

TEST_REVIEW_IDS = [
    'REV_00073', 'REV_00129', 'REV_00313', 'REV_00457', 'REV_00564',
    'REV_00795', 'REV_00822', 'REV_01074', 'REV_01284', 'REV_01427',
    'REV_01651', 'REV_01744', 'REV_01800', 'REV_02529', 'REV_02912',
    'REV_03108', 'REV_04412', 'REV_04777', 'REV_05070', 'REV_05863',
    'REV_06090', 'REV_06534', 'REV_08253', 'REV_08313', 'REV_08836'
]

sub = manifest[manifest['review_id'].isin(TEST_REVIEW_IDS)]
print("Total test reviews loaded:", len(sub))
for _, r in sub.iterrows():
    print(f"\n--- [{r['review_id']}] {r['establishment_id']} ({r['star_rating']}*) ---")
    print(repr(r['review_text']))
