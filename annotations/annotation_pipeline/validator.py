import os
import pandas as pd
from typing import List, Dict, Any, Tuple

class AnnotationValidator:
    """
    Validation engine ensuring strict compliance with DineSense AI annotation standards.
    Validates character offsets, evidence spans, label domains, and hierarchical invariants.
    """

    ALLOWED_STATUSES = {"Annotated", "Needs Review", "No Aspect Opinion"}
    ALLOWED_ASPECTS = {
        "Food / Dining",
        "Staff / Service",
        "Facilities / Amenities",
        "Price / Value",
        "Cleanliness",
        "Location",
        "Booking / Check-in / Check-out",
        "Safety / Security",
        "Room",
        "Other",
        ""
    }
    ALLOWED_SENTIMENTS = {"Positive", "Negative", "Neutral", "Unclear", ""}

    def validate_dataset(
        self,
        assertions_df: pd.DataFrame,
        manifest_df: pd.DataFrame
    ) -> Tuple[bool, List[str]]:
        """
        Validates the entire assertions dataset against source reviews and schema rules.
        Returns (is_valid, error_list).
        """
        errors = []

        # Check review count
        manifest_reviews = set(manifest_df['review_id'])
        annotated_reviews = set(assertions_df['review_id'])
        if manifest_reviews != annotated_reviews:
            missing = manifest_reviews - annotated_reviews
            extra = annotated_reviews - manifest_reviews
            if missing:
                errors.append(f"Missing annotations for reviews: {list(missing)[:5]}")
            if extra:
                errors.append(f"Unexpected extra reviews in annotations: {list(extra)[:5]}")

        # Check unique assertion IDs
        dup_assertions = assertions_df[assertions_df['assertion_id'].duplicated()]['assertion_id'].tolist()
        if dup_assertions:
            errors.append(f"Duplicate assertion IDs found: {dup_assertions[:5]}")

        # Map review text for offset verification
        review_map = dict(zip(manifest_df['review_id'], manifest_df['review_text']))

        for idx, row in assertions_df.iterrows():
            rev_id = row['review_id']
            ass_id = row['assertion_id']
            full_text = review_map.get(rev_id)

            if full_text is None:
                errors.append(f"[{ass_id}] Review ID '{rev_id}' not found in manifest.")
                continue

            # 1. Clause offset verification
            c_start = int(row['clause_start_char'])
            c_end = int(row['clause_end_char'])
            c_text = row['clause_text']

            if c_start < 0 or c_end > len(full_text) or c_start >= c_end:
                errors.append(f"[{ass_id}] Invalid clause offset range: [{c_start}, {c_end}) for text len {len(full_text)}")
            elif full_text[c_start:c_end] != c_text:
                errors.append(f"[{ass_id}] Clause text offset mismatch:\nExpected: {repr(c_text)}\nActual:   {repr(full_text[c_start:c_end])}")

            # 2. Annotation status domain
            status = row['annotation_status']
            if status not in self.ALLOWED_STATUSES:
                errors.append(f"[{ass_id}] Invalid annotation_status: '{status}'")

            # 3. No Aspect Opinion invariants
            if status == "No Aspect Opinion":
                if pd.notna(row['aspect']) and str(row['aspect']).strip():
                    errors.append(f"[{ass_id}] 'No Aspect Opinion' record must have empty aspect, got '{row['aspect']}'")
                if pd.notna(row['sentiment']) and str(row['sentiment']).strip():
                    errors.append(f"[{ass_id}] 'No Aspect Opinion' record must have empty sentiment, got '{row['sentiment']}'")
            else:
                # 4. Aspect domain
                aspect = str(row['aspect']) if pd.notna(row['aspect']) else ""
                if aspect not in self.ALLOWED_ASPECTS or not aspect:
                    errors.append(f"[{ass_id}] Invalid aspect category: '{aspect}'")

                # 5. Sentiment domain
                sent = str(row['sentiment']) if pd.notna(row['sentiment']) else ""
                if sent not in self.ALLOWED_SENTIMENTS or not sent:
                    errors.append(f"[{ass_id}] Invalid sentiment: '{sent}'")

            # 6. Target span offset verification
            if pd.notna(row['aspect_target_span']) and str(row['aspect_target_span']).strip():
                t_span = str(row['aspect_target_span'])
                if pd.isna(row['target_start_char']) or pd.isna(row['target_end_char']):
                    errors.append(f"[{ass_id}] Target span present '{t_span}' but target offsets are missing.")
                else:
                    t_start = int(row['target_start_char'])
                    t_end = int(row['target_end_char'])
                    if full_text[t_start:t_end] != t_span:
                        errors.append(f"[{ass_id}] Target span offset mismatch:\nExpected: {repr(t_span)}\nActual:   {repr(full_text[t_start:t_end])}")

            # 7. Opinion span offset verification
            if pd.notna(row['opinion_span']) and str(row['opinion_span']).strip():
                op_span = str(row['opinion_span'])
                if pd.isna(row['opinion_start_char']) or pd.isna(row['opinion_end_char']):
                    errors.append(f"[{ass_id}] Opinion span present '{op_span}' but opinion offsets are missing.")
                else:
                    op_start = int(row['opinion_start_char'])
                    op_end = int(row['opinion_end_char'])
                    if full_text[op_start:op_end] != op_span:
                        errors.append(f"[{ass_id}] Opinion span offset mismatch:\nExpected: {repr(op_span)}\nActual:   {repr(full_text[op_start:op_end])}")

        is_valid = len(errors) == 0
        return is_valid, errors
