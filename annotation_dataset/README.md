# DineSense AI: Annotation Dataset & Artifacts

This folder consolidates all core annotation files, guidelines, schema specifications, and the resulting milestone dataset produced during the 2,000-review annotation process.

## Folder Contents

| File | Format | Description |
| :--- | :---: | :--- |
| `annotations_2000_final.csv` | CSV | Final 2,000-review milestone dataset (11,635 assertion-level rows across 100 establishments). 100% offset-verified. |
| `annotations_2000.jsonl` | JSONL | JSON Lines serialization of the 2,000-review annotated corpus. |
| `clause_evaluation_input_2000.csv` | CSV | Clean clause-level aggregation (11,631 unique clauses) mapping assertions to reference aspect label sets. |
| `annotation_guide.md` | Markdown | Comprehensive annotation guidelines defining the 5-aspect taxonomy (`Food`, `Service`, `Price / Value`, `Ambience`, `General Experience`) and disambiguation rules. |
| `annotation_schema.md` | Markdown | Technical data dictionary and schema documentation defining record hierarchy, span offsets, and sentiment conventions. |
| `summary_2000.md` | Markdown | High-level audit report and statistical distribution summary of the 2,000-review annotation milestone. |

## Dataset Hierarchy

```
Review (review_id, e.g., REV_00073)
  └── Clause (clause_id, e.g., REV_00073_C01)
        └── Assertion (assertion_id, e.g., REV_00073_C01_A01)
```

- **Reviews**: 2,000 unique reviews
- **Clauses**: 11,631 unique clauses
- **Assertions**: 11,635 total assertions (4 clauses contain 2 co-occurring assertions)
- **Aspect Taxonomy**:
  1. `Food`
  2. `Service`
  3. `Price / Value`
  4. `Ambience`
  5. `General Experience`
- **Non-Evaluative Status**: `No Aspect Opinion` (6,677 clauses / 57.4%)
