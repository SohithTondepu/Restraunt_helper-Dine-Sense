import os
import sys
import json
import pandas as pd
import importlib.util

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"d:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews"
ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "annotations")
V2_2_DIR = os.path.join(ANNOTATIONS_DIR, "v2.2")
BATCH_1900_DIR = os.path.join(ANNOTATIONS_DIR, "batch_1900")
COMBINED_DIR = os.path.join(ANNOTATIONS_DIR, "combined_2000")

MANIFEST_1900_PATH = os.path.join(BATCH_1900_DIR, "sample_manifest_1900.csv")
PILOT_CSV_PATH = os.path.join(V2_2_DIR, "pilot_v2.2_annotations_100.csv")

OUTPUT_CSV_1900 = os.path.join(BATCH_1900_DIR, "annotations_1900.csv")
OUTPUT_JSONL_1900 = os.path.join(BATCH_1900_DIR, "annotations_1900.jsonl")
TEMPLATE_CSV_1900 = os.path.join(BATCH_1900_DIR, "verified_annotations_template_1900.csv")
SUMMARY_MD_1900 = os.path.join(BATCH_1900_DIR, "batch_1900_summary.md")

OUTPUT_CSV_2000 = os.path.join(COMBINED_DIR, "annotations_2000.csv")
OUTPUT_JSONL_2000 = os.path.join(COMBINED_DIR, "annotations_2000.jsonl")
SUMMARY_MD_2000 = os.path.join(COMBINED_DIR, "summary_2000.md")

os.makedirs(BATCH_1900_DIR, exist_ok=True)
os.makedirs(COMBINED_DIR, exist_ok=True)

# Dynamically import batch_1900_annotator_engine
spec = importlib.util.spec_from_file_location("batch_1900_annotator_engine", os.path.join(BATCH_1900_DIR, "batch_1900_annotator_engine.py"))
engine_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine_mod)

def run():
    print(f"Loading 1,900 manifest: {MANIFEST_1900_PATH}")
    manifest_1900 = pd.read_csv(MANIFEST_1900_PATH, keep_default_na=False)
    print(f"Loaded {len(manifest_1900)} reviews across {manifest_1900['establishment_id'].nunique()} establishments.")

    annotator = engine_mod.Batch1900Annotator()

    all_assertions_1900 = []
    print("Executing fine segmentation and annotation on 1,900 reviews...")
    for idx, row in manifest_1900.iterrows():
        rev_id = row['review_id']
        orig_id = row['original_row_id']
        est_id = row['establishment_id']
        rating = row['star_rating']
        timestamp = row['review_timestamp']
        text = row['review_text']

        ass_list = annotator.annotate(rev_id, orig_id, est_id, rating, timestamp, text)
        all_assertions_1900.extend(ass_list)

    df_1900 = pd.DataFrame(all_assertions_1900)
    print(f"Total assertions generated for 1,900 reviews: {len(df_1900)}")

    # 100% strict offset verification
    print("Verifying 100% character offsets verbatim against source review text...")
    for idx, r in df_1900.iterrows():
        rt = r['review_text']
        aid = r['assertion_id']
        cs, ce = int(r['clause_start_char']), int(r['clause_end_char'])
        assert rt[cs:ce] == r['clause_text'], f"Clause offset mismatch in {aid}"
        if r['aspect_target_span']:
            ts, te = int(r['target_start_char']), int(r['target_end_char'])
            assert rt[ts:te] == r['aspect_target_span'], f"Target offset mismatch in {aid}"
        if r['opinion_span']:
            os_c, oe = int(r['opinion_start_char']), int(r['opinion_end_char'])
            assert rt[os_c:oe] == r['opinion_span'], f"Opinion offset mismatch in {aid}"
    print("ALL character offsets verified verbatim with 100% precision!")

    # Save CSV and JSONL for 1,900
    df_1900.to_csv(OUTPUT_CSV_1900, index=False, encoding='utf-8')
    print(f"Saved {len(df_1900)} assertions to {OUTPUT_CSV_1900}")

    with open(OUTPUT_JSONL_1900, 'w', encoding='utf-8') as f:
        for ass in all_assertions_1900:
            f.write(json.dumps(ass, ensure_ascii=False) + '\n')
    print(f"Saved JSONL to {OUTPUT_JSONL_1900}")

    # Save verification template for 1,900
    tmpl_cols = [
        'assertion_id', 'clause_id', 'review_id', 'establishment_id', 'star_rating',
        'clause_text', 'aspect', 'aspect_target_span', 'opinion_span', 'sentiment',
        'theme', 'annotation_status', 'clause_relation', 'elaboration_of',
        'verification_status', 'verified_aspect', 'verified_target_span',
        'verified_opinion_span', 'verified_sentiment', 'verified_theme',
        'correction_notes', 'llm_rationale', 'clause_start_char', 'clause_end_char',
        'review_text'
    ]
    tmpl_df = df_1900.copy()
    tmpl_df['verification_status'] = 'pending'
    tmpl_df['verified_aspect'] = ''
    tmpl_df['verified_target_span'] = ''
    tmpl_df['verified_opinion_span'] = ''
    tmpl_df['verified_sentiment'] = ''
    tmpl_df['verified_theme'] = ''
    tmpl_df['correction_notes'] = ''
    tmpl_df = tmpl_df[tmpl_cols]
    tmpl_df.to_csv(TEMPLATE_CSV_1900, index=False, encoding='utf-8')
    print(f"Saved verification template to {TEMPLATE_CSV_1900}")

    # Build combined 2,000 dataset
    print(f"\nBuilding combined 2,000-review dataset using pilot from: {PILOT_CSV_PATH}")
    df_pilot = pd.read_csv(PILOT_CSV_PATH, keep_default_na=False)
    print(f"Loaded approved pilot: {len(df_pilot)} assertions.")

    df_2000 = pd.concat([df_pilot, df_1900], ignore_index=True)
    # Sort deterministically by original_row_id, then assertion_id
    df_2000 = df_2000.sort_values(['original_row_id', 'assertion_id']).reset_index(drop=True)
    print(f"Total assertions in combined 2,000 dataset: {len(df_2000)}")

    df_2000.to_csv(OUTPUT_CSV_2000, index=False, encoding='utf-8')
    print(f"Saved combined 2,000 CSV to {OUTPUT_CSV_2000}")

    with open(OUTPUT_JSONL_2000, 'w', encoding='utf-8') as f:
        for record in df_2000.to_dict(orient='records'):
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"Saved combined 2,000 JSONL to {OUTPUT_JSONL_2000}")

    # Summary Statistics for 1,900 Batch
    print("\n==========================================")
    print("=== BATCH 1,900 SUMMARY STATISTICS ===")
    print("==========================================")
    print(f"Total Reviews: {len(manifest_1900)}")
    print(f"Total Clauses: {df_1900['clause_id'].nunique()}")
    print(f"Total Assertions: {len(df_1900)}")
    print("\nAnnotation Status Distribution:")
    print(df_1900['annotation_status'].value_counts())
    print("\nAspect Distribution (Annotated):")
    print(df_1900[df_1900['annotation_status'] == 'Annotated']['aspect'].value_counts())
    print("\nSentiment Distribution (Annotated):")
    print(df_1900[df_1900['annotation_status'] == 'Annotated']['sentiment'].value_counts())

    print("\n==========================================")
    print("=== COMBINED 2,000 SUMMARY STATISTICS ===")
    print("==========================================")
    print(f"Total Reviews: {df_2000['review_id'].nunique()}")
    print(f"Total Clauses: {df_2000['clause_id'].nunique()}")
    print(f"Total Assertions: {len(df_2000)}")
    print("\nCombined Annotation Status Distribution:")
    print(df_2000['annotation_status'].value_counts())
    print("\nCombined Aspect Distribution (Annotated):")
    print(df_2000[df_2000['annotation_status'] == 'Annotated']['aspect'].value_counts())
    print("\nCombined Sentiment Distribution (Annotated):")
    print(df_2000[df_2000['annotation_status'] == 'Annotated']['sentiment'].value_counts())

if __name__ == '__main__':
    run()
