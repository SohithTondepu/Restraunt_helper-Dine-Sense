import re
from typing import List, Tuple

class ClauseSegmenter:
    """
    Splits review text into coherent, evidence-bearing clause spans
    while strictly preserving exact character offsets against the raw text.
    Handles CRLF, punctuation, sentence breaks, and contrastive markers.
    """
    
    # Boundary markers: newlines, sentence-ending punctuation (.!?), semicolons,
    # contrastive conjunctions (but, however, although, yet, whereas),
    # and coordinating conjunctions connecting distinct aspect phrases.
    BOUNDARY_REGEX = re.compile(
        r'('
        r'\r\n|\n|'                           # Line breaks
        r'[.!?]+(?:\s+|\r\n|\n|$)|'           # Sentence terminal punctuation
        r'\s*;\s*|'                           # Semicolons
        r'\s*,\s*(?=(?:but|however|although|while|yet|whereas|and\s+also)\b)|' # Comma before contrast
        r'\s+(?:but|however|although|whereas)\s+' # Strong contrastive conjunctions
        r')',
        re.IGNORECASE
    )

    def segment(self, review_text: str) -> List[Tuple[int, int, str]]:
        """
        Segments review_text into clauses.
        Returns a list of tuples: (clause_start_char, clause_end_char, clause_text)
        Invariants: review_text[clause_start_char:clause_end_char] == clause_text
        """
        if not review_text or not review_text.strip():
            return []

        spans: List[Tuple[int, int, str]] = []
        pos = 0
        text_len = len(review_text)

        for match in self.BOUNDARY_REGEX.finditer(review_text):
            m_start, m_end = match.span()
            raw_chunk = review_text[pos:m_start]

            # Calculate trimmed offsets to avoid leading/trailing whitespace
            # while keeping exact index alignment with original review_text
            l_strip = len(raw_chunk) - len(raw_chunk.lstrip())
            r_strip = len(raw_chunk) - len(raw_chunk.rstrip())
            
            c_start = pos + l_strip
            c_end = m_start - r_strip

            if c_end > c_start:
                clause_str = review_text[c_start:c_end]
                # Filter out pure punctuation remnants
                if re.search(r'\w', clause_str):
                    spans.append((c_start, c_end, clause_str))

            pos = m_end

        # Process trailing chunk
        if pos < text_len:
            raw_chunk = review_text[pos:]
            l_strip = len(raw_chunk) - len(raw_chunk.lstrip())
            r_strip = len(raw_chunk) - len(raw_chunk.rstrip())
            c_start = pos + l_strip
            c_end = text_len - r_strip

            if c_end > c_start:
                clause_str = review_text[c_start:c_end]
                if re.search(r'\w', clause_str):
                    spans.append((c_start, c_end, clause_str))

        # Fallback if no valid clauses found
        if not spans and review_text.strip():
            l_strip = len(review_text) - len(review_text.lstrip())
            r_strip = len(review_text) - len(review_text.rstrip())
            spans.append((l_strip, text_len - r_strip, review_text[l_strip:text_len - r_strip]))

        return spans
