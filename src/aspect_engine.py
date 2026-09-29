import os
import re
import spacy
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
HF_MODEL_DIR = os.path.join(MODELS_DIR, 'distilbert_sentiment')
LEGACY_MODEL_PATH = os.path.join(MODELS_DIR, 'deberta_finetuned.pt')

_nlp = None
_model = None
_tokenizer = None


def get_spacy_nlp():
    """Lazy loader for SpaCy neural pipeline."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def get_transformer_model():
    """Loads the fine-tuned Transformer model and tokenizer."""
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
        
    if os.path.exists(HF_MODEL_DIR):
        try:
            _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_DIR)
            _model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_DIR)
            _model.eval()
            return _model, _tokenizer
        except Exception as e:
            print(f"Notice loading HF model dir: {e}")
            
    if os.path.exists(LEGACY_MODEL_PATH):
        try:
            _tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            _model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
            _model.load_state_dict(torch.load(LEGACY_MODEL_PATH, map_location='cpu'))
            _model.eval()
            return _model, _tokenizer
        except Exception as e:
            print(f"Notice loading legacy pt: {e}")
            
    return None, None


# Curated 4 Business Core Aspects for Restaurant Reviews
ASPECT_LEXICON = {
    'Food': {
        'food', 'taste', 'dish', 'dishes', 'meal', 'meals', 'chicken', 'biryani', 'curry',
        'gravy', 'portion', 'portions', 'quantity', 'quality', 'fresh', 'cold', 'stale',
        'spicy', 'bland', 'flavor', 'flavors', 'flavour', 'sweet', 'salt', 'salty', 'oil',
        'sauce', 'dessert', 'starters', 'rice', 'roti', 'naan', 'drink', 'beverage', 'pizza',
        'burger', 'pasta', 'soup', 'salad', 'paneer', 'mutton', 'kebab', 'tandoori'
    },
    'Service': {
        'service', 'staff', 'waiter', 'waitress', 'server', 'servers', 'manager', 'hospitality',
        'behavior', 'behaviour', 'attitude', 'rude', 'polite', 'courteous', 'prompt', 'slow',
        'quick', 'fast', 'delay', 'delayed', 'wait', 'waiting', 'minutes', 'order', 'serve',
        'served', 'bill', 'billing', 'management', 'response', 'attendant', 'friendly'
    },
    'Price': {
        'price', 'prices', 'pricing', 'cost', 'costs', 'costly', 'expensive', 'cheap',
        'reasonable', 'overpriced', 'value', 'worth', 'worthwhile', 'money', 'budget',
        'pocket', 'pocket-friendly', 'affordable', 'bill', 'charge', 'charges', 'overcharged'
    },
    'Ambience': {
        'ambience', 'ambiance', 'atmosphere', 'vibe', 'vibes', 'decor', 'interior', 'interiors',
        'lighting', 'music', 'seating', 'seat', 'seats', 'table', 'tables', 'space', 'spacious',
        'clean', 'cleanliness', 'hygiene', 'dirty', 'smell', 'odor', 'ac', 'air conditioning',
        'crowd', 'crowded', 'noisy', 'noise', 'quiet', 'view', 'location', 'parking'
    }
}

SERVICE_PRIORITY_TERMS = {
    'wait', 'waiting', 'delay', 'delayed', 'slow', 'minute', 'minutes', 'hour', 'hours',
    'took long', 'served late', 'rude waiter', 'poor service', 'ignored'
}

# Dense Semantic Aspect Centroid Descriptions
ASPECT_DESCRIPTIONS = {
    'Food': 'food taste delicious flavor recipe meal dishes cuisine freshness cooking portion ingredients savory culinary',
    'Service': 'service staff waiter waitress hospitality server prompt rude friendly delay response attentive manager billing order attentiveness',
    'Price': 'price pricing expensive cheap affordable money cost bill value for money overpriced pocket friendly charges budget worth economic',
    'Ambience': 'ambience atmosphere decor interior music lighting vibe seating view noise crowd comfortable cleanliness aesthetic temperature acoustic'
}

_sentence_model = None
_aspect_embeddings = None
_aspect_names = list(ASPECT_DESCRIPTIONS.keys())


def get_sentence_model():
    """Lazy loader for SentenceTransformer to enable dense semantic aspect routing."""
    global _sentence_model, _aspect_embeddings
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            _aspect_embeddings = _sentence_model.encode(list(ASPECT_DESCRIPTIONS.values()), normalize_embeddings=True)
        except Exception as e:
            pass
    return _sentence_model, _aspect_embeddings


def split_into_clauses(text: str) -> list:
    """
    RST-Inspired Discourse Clause Segmentation:
    1. Segments sentences via SpaCy parser.
    2. Identifies Concessive Discourse Markers (RST Satellites): 'despite', 'in spite of', 'even though', 'although'.
    3. Identifies Contrastive & Coordinate Conjunctions (RST Nuclei): 'but', 'however', 'yet', 'nevertheless', ';'.
    4. Strips clausal boundary artifacts, preserving coherent grammatical clauses.
    """
    text = str(text).strip()
    if not text:
        return []
        
    nlp = get_spacy_nlp()
    doc = nlp(text)
    
    # Combined regex pattern for RST Concessive Satellites + Adversative Conjunctions + Punctuation
    rst_split_regex = re.compile(
        r'(?<=[.!?;\n—])|'
        r'\b(?:despite(?:\s+the\s+fact\s+that)?|in\s+spite\s+of|regardless\s+of|even\s+though|even\s+if|although|though|whereas|while)\b|'
        r'\b(?:but|however|yet|nevertheless|nonetheless|on\s+the\s+other\s+hand|on\s+the\s+contrary|except\s+that|except\s+for)\b',
        re.IGNORECASE
    )
    
    leading_conjunction_cleaner = re.compile(
        r'^(?:and|but|or|so|yet|because|although|though|despite|while|however|whereas)\s+',
        re.IGNORECASE
    )
    
    concessive_start_regex = re.compile(
        r'^(?:despite(?:\s+the\s+fact\s+that)?|in\s+spite\s+of|regardless\s+of|even\s+though|even\s+if|although|though|while)\s+(.+?),\s*',
        re.IGNORECASE
    )
    
    clauses = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        # Transform leading concessive comma boundary into explicit clausal break
        sent_text = concessive_start_regex.sub(r'\1 ; ', sent_text)
        
        parts = rst_split_regex.split(sent_text)
        for p in parts:
            if not p:
                continue
            cleaned_p = p.strip(' ,;:-—')
            # Clean leading conjunction residue
            cleaned_p = leading_conjunction_cleaner.sub('', cleaned_p).strip()
            if len(cleaned_p.split()) >= 2:
                clauses.append(cleaned_p)
                
    if not clauses and text:
        clauses = [text]
    return clauses


def match_aspect_hybrid(clause_text: str, similarity_threshold: float = 0.22) -> set:
    """
    Hybrid Aspect Matcher:
    Tier 1: O(1) Fast Exact Lexicon & Latency Priority Matching.
    Tier 2: Dense Semantic Cosine Similarity (via all-MiniLM-L6-v2) for implicit aspects & metaphors.
    """
    clause_lower = clause_text.lower()
    words = set(re.findall(r'\b[a-zA-Z]{3,}\b', clause_lower))
    detected = set()
    
    # Priority: Wait/latency terms route to Service
    if any(term in clause_lower for term in SERVICE_PRIORITY_TERMS):
        detected.add('Service')
        
    # Tier 1: Lexicon matching
    for aspect, keywords in ASPECT_LEXICON.items():
        if aspect == 'Service' and 'Service' in detected:
            continue
        if words.intersection(keywords):
            detected.add(aspect)
            
    # If Tier 1 found explicit aspect(s), return immediately
    if detected:
        return detected
        
    # Tier 2: Dense Semantic Embedding Similarity (Handles implicit metaphors like 'cost an arm and a leg')
    sent_model, aspect_vecs = get_sentence_model()
    if sent_model is not None and aspect_vecs is not None:
        try:
            clause_vec = sent_model.encode([clause_text], normalize_embeddings=True)
            # Dot product of normalized vectors = cosine similarity
            sims = np.dot(aspect_vecs, clause_vec.T).flatten()
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            
            if best_score >= similarity_threshold:
                detected.add(_aspect_names[best_idx])
        except Exception:
            pass
            
    return detected


def predict_clause_sentiment(clause_text: str) -> dict:
    """
    Executes a PyTorch forward pass on the fine-tuned Transformer model to predict
    probabilities: [Negative, Neutral, Positive].
    """
    model, tokenizer = get_transformer_model()
    if model is not None and tokenizer is not None:
        try:
            inputs = tokenizer(clause_text, truncation=True, padding=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()
                
            neg_p = round(float(probs[0]), 3)
            neu_p = round(float(probs[1]), 3)
            pos_p = round(float(probs[2]), 3)
            
            label_idx = int(torch.argmax(outputs.logits, dim=1).item())
            labels = ['Negative', 'Neutral', 'Positive']
            pred_label = labels[label_idx]
            
            return {
                'sentiment': pred_label,
                'confidence': round(max([neg_p, neu_p, pos_p]), 3),
                'probs': {'Negative': neg_p, 'Neutral': neu_p, 'Positive': pos_p},
                'polarity_score': round(pos_p - neg_p, 3)
            }
        except Exception:
            pass
            
    # Deterministic lexical fallback if model unavailable
    lower = clause_text.lower()
    neg_triggers = ['bad', 'worst', 'horrible', 'poor', 'slow', 'cold', 'dirty', 'rude', 'overpriced', 'stale']
    pos_triggers = ['good', 'great', 'excellent', 'amazing', 'delicious', 'fresh', 'tasty', 'friendly', 'love', 'best']
    
    neg_count = sum(1 for w in neg_triggers if w in lower)
    pos_count = sum(1 for w in pos_triggers if w in lower)
    
    if pos_count > neg_count:
        return {'sentiment': 'Positive', 'confidence': 0.8, 'probs': {'Negative': 0.1, 'Neutral': 0.2, 'Positive': 0.7}, 'polarity_score': 0.6}
    elif neg_count > pos_count:
        return {'sentiment': 'Negative', 'confidence': 0.8, 'probs': {'Negative': 0.7, 'Neutral': 0.2, 'Positive': 0.1}, 'polarity_score': -0.6}
    else:
        return {'sentiment': 'Neutral', 'confidence': 0.6, 'probs': {'Negative': 0.2, 'Neutral': 0.6, 'Positive': 0.2}, 'polarity_score': 0.0}


def extract_aspects_from_review(review_text: str) -> list:
    """
    Upgraded ABSA pipeline:
    1. RST Discourse Segmentation (isolates concessive satellites & contrastive nuclei).
    2. Hybrid Semantic Matching (Tier 1 Lexicon + Tier 2 Dense Cosine Embeddings).
    3. Transformer clause-level sentiment inference.
    """
    clauses = split_into_clauses(review_text)
    aspect_findings = []
    
    for clause in clauses:
        detected_aspects = match_aspect_hybrid(clause)
        
        # If no specific aspect matched (neither explicit nor semantic), skip to keep high precision
        if not detected_aspects:
            continue
            
        # Get clause sentiment
        sent_info = predict_clause_sentiment(clause)
        
        for asp in detected_aspects:
            aspect_findings.append({
                'clause': clause,
                'aspect': asp,
                'sentiment': sent_info['sentiment'],
                'confidence': sent_info['confidence'],
                'polarity_score': sent_info['polarity_score'],
                'probs': sent_info['probs']
            })
            
    return aspect_findings


# Backwards compatibility aliases
analyze_aspect_sentiments = extract_aspects_from_review
ASPECT_MAP = ASPECT_LEXICON

