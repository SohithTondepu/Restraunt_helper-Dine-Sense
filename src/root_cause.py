import os
import re
import spacy
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

_nlp = None

def get_spacy_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


STOP_TARGETS = {'place', 'restaurant', 'thing', 'one', 'it', 'experience', 'order', 'day', 'time'}


def extract_opinion_pairs(text: str) -> list:
    """
    Extracts structured (target, opinion) pairs using SpaCy dependency parsing:
    1. nsubj + acomp: e.g. 'food was cold' -> ('food', 'cold')
    2. amod: e.g. 'rude waiter' -> ('waiter', 'rude')
    3. neg: e.g. 'not fresh' -> ('dish', 'not fresh')
    """
    nlp = get_spacy_nlp()
    doc = nlp(str(text))
    pairs = []
    
    for token in doc:
        # Pattern 1: Adjectival complement (nsubj + acomp / attr)
        if token.pos_ in ["ADJ"]:
            head = token.head
            # Check if verb links subject to adjective (e.g., 'is', 'was', 'tasted')
            if head.pos_ in ["VERB", "AUX"]:
                for child in head.children:
                    if child.dep_ in ["nsubj", "nsubjpass"] and child.pos_ in ["NOUN", "PROPN"]:
                        target = child.lemma_.lower()
                        # Check negation on token or verb
                        is_neg = any(c.dep_ == "neg" for c in list(token.children) + list(head.children))
                        opinion = f"not {token.lemma_.lower()}" if is_neg else token.lemma_.lower()
                        if target not in STOP_TARGETS:
                            pairs.append(f"{target} {opinion}")
                            
        # Pattern 2: Adjective modifier (amod)
        if token.dep_ == "amod" and token.head.pos_ in ["NOUN", "PROPN"]:
            target = token.head.lemma_.lower()
            is_neg = any(c.dep_ == "neg" for c in token.children)
            opinion = f"not {token.lemma_.lower()}" if is_neg else token.lemma_.lower()
            if target not in STOP_TARGETS:
                pairs.append(f"{opinion} {target}")
                
        # Pattern 3: Direct verb negation
        if token.dep_ == "neg" and token.head.pos_ in ["VERB", "ADJ"]:
            verb = token.head.lemma_.lower()
            for child in token.head.children:
                if child.dep_ in ["nsubj", "dobj"] and child.pos_ in ["NOUN", "PROPN"]:
                    target = child.lemma_.lower()
                    if target not in STOP_TARGETS:
                        pairs.append(f"{target} not {verb}")
                        
    return list(set(pairs))


def cluster_restaurant_complaints(restaurant_name: str, df: pd.DataFrame, aspect_filter: str = None, top_k: int = 5) -> list:
    """
    Extracts dependency opinion pairs for a restaurant's negative reviews (Rating <= 2.5),
    groups complaints into root-cause clusters with outlier reassignment and evidence quotes.
    """
    if df.empty or 'Restaurant' not in df.columns:
        return []
        
    rest_df = df[(df['Restaurant'] == restaurant_name) & (df['Rating'] <= 2.5)].copy()
    if rest_df.empty:
        return []
        
    all_pairs = []
    pair_to_quotes = {}
    
    for _, row in rest_df.iterrows():
        review_text = str(row['Review'])
        extracted = extract_opinion_pairs(review_text)
        
        for p in extracted:
            all_pairs.append(p)
            if p not in pair_to_quotes:
                pair_to_quotes[p] = []
            if len(pair_to_quotes[p]) < 3:
                snippet = review_text[:140] + ('...' if len(review_text) > 140 else '')
                if snippet not in pair_to_quotes[p]:
                    pair_to_quotes[p].append(snippet)
                    
    if not all_pairs:
        return []
        
    # Frequency count
    counts = Counter(all_pairs)
    top_items = counts.most_common(top_k * 2)
    
    # Semantic grouping of near-synonyms (e.g. 'cold food' and 'food cold')
    clusters = []
    used = set()
    
    for phrase, count in top_items:
        if phrase in used:
            continue
        used.add(phrase)
        
        words = set(phrase.split())
        merged_count = count
        evidence = list(pair_to_quotes.get(phrase, []))
        
        # Merge near matches
        for other_p, other_c in top_items:
            if other_p != phrase and other_p not in used:
                if set(other_p.split()) == words or (len(words.intersection(set(other_p.split()))) >= 2):
                    merged_count += other_c
                    used.add(other_p)
                    for q in pair_to_quotes.get(other_p, []):
                        if len(evidence) < 3 and q not in evidence:
                            evidence.append(q)
                            
        clusters.append({
            'root_cause': phrase.title(),
            'frequency': merged_count,
            'evidence_quotes': evidence
        })
        
        if len(clusters) >= top_k:
            break
            
    return clusters
