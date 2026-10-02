import sys
import os
import re
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"d:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews"
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "annotations", "pilot_sample_manifest.csv")
manifest = pd.read_csv(MANIFEST_PATH, keep_default_na=False)

# Enhanced fine-grained segmenter regex:
# Splits on:
# 1. Line breaks: \r\n, \n
# 2. Terminal punctuation: . ! ?
# 3. Semicolons and colons
# 4. Dashes/ellipses: --, ..., -
# 5. Coordinating conjunctions joining clauses:
#    " and " (when connecting distinct predicates/clauses)
#    " but ", " however ", " although ", " while ", " though ", " whereas ", " yet "
# 6. Commas followed by conjunctions or distinct aspect subjects
FINE_BOUNDARY_REGEX = re.compile(
    r'('
    r'\r\n|\n|'
    r'[.!?]+(?:\s+|\r\n|\n|$)|'
    r'\s*;\s*|'
    r'\.{2,}|'
    r'\s+-\s+|'
    r'\s*,\s*(?=(?:but|however|although|while|though|yet|whereas|and|services|service|bikash|ambience|food|proper|which\s+was|and\s+one\s+could|my\s+girl|staff|waiter|delivery)\b)|'
    r'\s+(?:but|however|although|whereas|though|while)\s+|'
    r'\s+and\s+(?=(?:the\s+staff|the\s+food|the\s+service|the\s+ambience|the\s+price|my\s+girl|service\s+is|food\s+is|ambience\s+is|delivery\s+was|i\s+felt|we\s+felt|it\s+tasted|it\s+was|also\s+got|everything\s+is|they\s+charge|i\s+wouldn[\'’]?t)\b)'
    r')',
    re.IGNORECASE
)

def fine_segment(text: str):
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

total_clauses = 0
for idx, r in manifest.iterrows():
    cls = fine_segment(r['review_text'])
    total_clauses += len(cls)
    for s, e, t in cls:
        assert r['review_text'][s:e] == t

print(f"Total reviews: {len(manifest)}")
print(f"Fine-grained clauses extracted: {total_clauses} (previously 648)")
