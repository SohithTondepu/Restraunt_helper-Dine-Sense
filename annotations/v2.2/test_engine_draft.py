import os
import sys
import re
import json
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"d:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews"
ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "annotations")
V2_2_DIR = os.path.join(ANNOTATIONS_DIR, "v2.2")
MANIFEST_PATH = os.path.join(ANNOTATIONS_DIR, "pilot_sample_manifest.csv")

FINE_BOUNDARY_REGEX = re.compile(
    r'('
    r'\r\n|\n|'
    r'[.!?]+(?:\s+|\r\n|\n|$)|'
    r'\s*;\s*|'
    r'\.{2,}|'
    r'\s+-\s+|'
    r'\s*,\s*(?=(?:but|however|although|while|though|yet|whereas|and|services|service|bikash|ambience|ambiance|food|proper|which\s+was|and\s+one\s+could|my\s+girl|staff|waiter|delivery|enjoyed|felt|with\s+neat|all\s+items|last\s+ambience)\b)|'
    r'\s+(?:but|however|although|whereas|though|while)\s+|'
    r'\s+and\s+(?=(?:the\s+staff|the\s+food|the\s+service|the\s+ambience|the\s+price|the\s+gulab|my\s+girl|service\s+is|food\s+is|ambience\s+is|delivery\s+was|i\s+felt|we\s+felt|it\s+tasted|it\s+was|also\s+got|everything\s+is|they\s+charge|i\s+wouldn[\'’]?t|have\s+no\s+respect|waters\s+think|gave\s+a\s+delicous|i\s+liked|the\s+gravy|i\s+am\s+in\s+love|needs\s+to\s+be\s+improved)\b)'
    r')',
    re.IGNORECASE
)

def segment_review(text: str) -> List[Tuple[int, int, str]]:
    if not text or not text.strip():
        return []
    spans = []
    pos = 0
    t_len = len(text)
    for m in FINE_BOUNDARY_REGEX.finditer(text):
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

def find_subspan(clause_text: str, regex_pattern: str) -> Optional[Tuple[int, int, str]]:
    m = re.search(regex_pattern, clause_text, re.IGNORECASE)
    if m:
        return m.start(), m.end(), clause_text[m.start():m.end()]
    return None

manifest = pd.read_csv(MANIFEST_PATH, keep_default_na=False)
print(f"Loaded manifest: {len(manifest)} reviews")
