import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

# Run the audit and inspect examples for each group
from audit_all_391 import findings_df

groups = [
    "A: Missed Specific Aspect (Food)",
    "A: Missed Specific Aspect (Service)",
    "A: Missed Specific Aspect (Ambience)",
    "A: Missed Specific Aspect (Price / Value)",
    "B: Missed General Experience",
    "C: Truly No Aspect Opinion",
    "D: Ambiguous / Borderline (Needs Review)"
]

for g in groups:
    sub = findings_df[findings_df['classification'] == g]
    print(f"\n==================== {g} (Total: {len(sub)}) ====================")
    for idx, r in sub.head(5).iterrows():
        print(f"[{r['clause_id']}] ({r['review_id']}): {repr(r['clause_text'])}")
        print(f"    Notes: {r['notes']}\n")
