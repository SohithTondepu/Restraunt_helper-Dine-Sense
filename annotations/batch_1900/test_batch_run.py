import os
import sys
import json
import re
import pandas as pd
import importlib.util

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"d:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews"
ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "annotations")
V2_2_DIR = os.path.join(ANNOTATIONS_DIR, "v2.2")
BATCH_1900_DIR = os.path.join(ANNOTATIONS_DIR, "batch_1900")
MANIFEST_1900_PATH = os.path.join(BATCH_1900_DIR, "sample_manifest_1900.csv")

# Dynamically import v2_2_annotator_engine
spec = importlib.util.spec_from_file_location("v2_2_annotator_engine", os.path.join(V2_2_DIR, "v2_2_annotator_engine.py"))
engine_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine_mod)

manifest_1900 = pd.read_csv(MANIFEST_1900_PATH, keep_default_na=False)
print(f"Loaded 1900 manifest: {len(manifest_1900)} reviews.")

annotator = engine_mod.ComprehensiveV22Annotator()

all_assertions = []
for idx, row in manifest_1900.iterrows():
    rev_id = row['review_id']
    orig_id = row['original_row_id']
    est_id = row['establishment_id']
    rating = row['star_rating']
    timestamp = row['review_timestamp']
    text = row['review_text']

    ass_list = annotator.annotate(rev_id, orig_id, est_id, rating, timestamp, text)
    all_assertions.extend(ass_list)

print(f"Generated {len(all_assertions)} assertions across {len(manifest_1900)} reviews.")
df = pd.DataFrame(all_assertions)
print("\n--- Initial Counts ---")
print("Status counts:")
print(df['annotation_status'].value_counts())
print("\nAspect distribution (Annotated):")
print(df[df['annotation_status'] == 'Annotated']['aspect'].value_counts())
print("\nSentiment distribution (Annotated):")
print(df[df['annotation_status'] == 'Annotated']['sentiment'].value_counts())
