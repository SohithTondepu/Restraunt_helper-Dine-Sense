import os
import sys
import json
import pandas as pd
import importlib.util

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"d:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews"
ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "annotations")
V2_2_DIR = os.path.join(ANNOTATIONS_DIR, "v2.2")
MANIFEST_PATH = os.path.join(ANNOTATIONS_DIR, "pilot_sample_manifest.csv")

OUTPUT_CSV = os.path.join(V2_2_DIR, "pilot_v2.2_annotations_100.csv")
OUTPUT_JSONL = os.path.join(V2_2_DIR, "pilot_v2.2_annotations_100.jsonl")
TEMPLATE_CSV = os.path.join(V2_2_DIR, "verified_annotations_template_100.csv")
SUMMARY_MD = os.path.join(V2_2_DIR, "pilot_v2.2_summary_100.md")

# Dynamically import v2_2_annotator_engine
spec = importlib.util.spec_from_file_location("v2_2_annotator_engine", os.path.join(V2_2_DIR, "v2_2_annotator_engine.py"))
engine_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine_mod)

manifest = pd.read_csv(MANIFEST_PATH, keep_default_na=False)
print(f"Loaded pilot manifest with {len(manifest)} reviews.")

annotator = engine_mod.ComprehensiveV22Annotator()

all_assertions = []
for idx, row in manifest.iterrows():
    rev_id = row['review_id']
    orig_id = row['original_row_id']
    est_id = row['establishment_id']
    rating = row['star_rating']
    timestamp = row['review_timestamp']
    text = row['review_text']

    ass_list = annotator.annotate(rev_id, orig_id, est_id, rating, timestamp, text)
    all_assertions.extend(ass_list)

df = pd.DataFrame(all_assertions)
df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print(f"Saved {len(df)} assertions to {OUTPUT_CSV}")

# Save JSONL
with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
    for ass in all_assertions:
        f.write(json.dumps(ass, ensure_ascii=False) + '\n')
print(f"Saved JSONL to {OUTPUT_JSONL}")

# Save verification template
tmpl_cols = [
    'assertion_id', 'clause_id', 'review_id', 'establishment_id', 'star_rating',
    'clause_text', 'aspect', 'aspect_target_span', 'opinion_span', 'sentiment',
    'theme', 'annotation_status', 'clause_relation', 'elaboration_of',
    'verification_status', 'verified_aspect', 'verified_target_span',
    'verified_opinion_span', 'verified_sentiment', 'verified_theme',
    'correction_notes', 'llm_rationale', 'clause_start_char', 'clause_end_char',
    'review_text'
]
tmpl_df = df.copy()
tmpl_df['verification_status'] = 'pending'
tmpl_df['verified_aspect'] = ''
tmpl_df['verified_target_span'] = ''
tmpl_df['verified_opinion_span'] = ''
tmpl_df['verified_sentiment'] = ''
tmpl_df['verified_theme'] = ''
tmpl_df['correction_notes'] = ''

tmpl_df = tmpl_df[tmpl_cols]
tmpl_df.to_csv(TEMPLATE_CSV, index=False, encoding='utf-8')
print(f"Saved verification template to {TEMPLATE_CSV}")

# Verification of offsets
for idx, r in df.iterrows():
    rt = r['review_text']
    cs, ce = int(r['clause_start_char']), int(r['clause_end_char'])
    assert rt[cs:ce] == r['clause_text'], f"Clause offset mismatch in {r['assertion_id']}"
    if r['aspect_target_span']:
        ts, te = int(r['target_start_char']), int(r['target_end_char'])
        assert rt[ts:te] == r['aspect_target_span'], f"Target offset mismatch in {r['assertion_id']}"
    if r['opinion_span']:
        os_c, oe = int(r['opinion_start_char']), int(r['opinion_end_char'])
        assert rt[os_c:oe] == r['opinion_span'], f"Opinion offset mismatch in {r['assertion_id']}"

print("All 100% character offsets verified verbatim against source review text!")
print("\n--- Summary Statistics ---")
print("Status counts:")
print(df['annotation_status'].value_counts())
print("\nAspect distribution:")
print(df[df['annotation_status'] == 'Annotated']['aspect'].value_counts())
print("\nSentiment distribution:")
print(df[df['annotation_status'] == 'Annotated']['sentiment'].value_counts())
