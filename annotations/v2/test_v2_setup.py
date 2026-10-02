import os
import sys
import re
import json
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

PROJECT_ROOT = r"d:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews"
ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "annotations")
V2_DIR = os.path.join(ANNOTATIONS_DIR, "v2")
MANIFEST_PATH = os.path.join(ANNOTATIONS_DIR, "pilot_sample_manifest.csv")
V1_CSV_PATH = os.path.join(ANNOTATIONS_DIR, "pilot_llm_annotations.csv")

OUTPUT_V2_CSV = os.path.join(V2_DIR, "revised_annotations_25.csv")
OUTPUT_V2_JSONL = os.path.join(V2_DIR, "revised_annotations_25.jsonl")
COMPARISON_REPORT_MD = os.path.join(V2_DIR, "comparison_report.md")
UNRESOLVED_CASES_MD = os.path.join(V2_DIR, "unresolved_cases.md")

# Selected 25 reviews
TEST_REVIEW_IDS = [
    'REV_00073', 'REV_00129', 'REV_00313', 'REV_00457', 'REV_00564',
    'REV_00795', 'REV_00822', 'REV_01074', 'REV_01284', 'REV_01427',
    'REV_01651', 'REV_01744', 'REV_01800', 'REV_02529', 'REV_02912',
    'REV_03108', 'REV_04412', 'REV_04777', 'REV_05070', 'REV_05863',
    'REV_06090', 'REV_06534', 'REV_08253', 'REV_08313', 'REV_08836'
]

# Boundary regex that preserves coordinates and splits on major boundaries + contrastives
BOUNDARY_REGEX = re.compile(
    r'('
    r'\r\n|\n|'
    r'[.!?]+(?:\s+|\r\n|\n|$)|'
    r'\s*;\s*|'
    r'\s*,\s*(?=(?:but|however|although|while|yet|whereas|services|service|bikash|ambience|food|proper|which\s+was|and\s+one\s+could)\b)|'
    r'\s+(?:but|however|although|whereas)\s+'
    r')',
    re.IGNORECASE
)

def segment_review(text: str) -> List[Tuple[int, int, str]]:
    if not text or not text.strip():
        return []
    spans = []
    pos = 0
    t_len = len(text)
    for m in BOUNDARY_REGEX.finditer(text):
        m_start, m_end = m.span()
        raw_chunk = text[pos:m_start]
        l_trim = len(raw_chunk) - len(raw_chunk.lstrip())
        r_trim = len(raw_chunk) - len(raw_chunk.rstrip())
        c_start = pos + l_trim
        c_end = m_start - r_trim
        if c_end > c_start:
            clause = text[c_start:c_end]
            if re.search(r'\w', clause):
                spans.append((c_start, c_end, clause))
        pos = m_end
    if pos < t_len:
        raw_chunk = text[pos:]
        l_trim = len(raw_chunk) - len(raw_chunk.lstrip())
        r_trim = len(raw_chunk) - len(raw_chunk.rstrip())
        c_start = pos + l_trim
        c_end = t_len - r_trim
        if c_end > c_start:
            clause = text[c_start:c_end]
            if re.search(r'\w', clause):
                spans.append((c_start, c_end, clause))
    return spans

print("Testing boundary segmenter loaded successfully.")
