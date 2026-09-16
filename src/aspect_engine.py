import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from src.preprocessing import extract_nouns, clean_text

# Ensure VADER lexicon is available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

ASPECT_MAP = {
    'Food': {
        'food', 'taste', 'dish', 'dishes', 'meal', 'meals', 'pasta', 'burger', 'burgers',
        'pizza', 'chicken', 'sauce', 'flavor', 'flavors', 'drink', 'drinks', 'beverage',
        'dessert', 'desserts', 'coffee', 'tea', 'bread', 'meat', 'fish', 'soup', 'salad',
        'menu', 'portion', 'portions', 'quality', 'recipe', 'curry', 'biryani', 'ice'
    },
    'Service': {
        'service', 'staff', 'waiter', 'waitress', 'manager', 'host', 'hostess', 'server',
        'servers', 'attendant', 'time', 'hospitality', 'speed', 'attitude', 'behavior',
        'promptness', 'response', 'service quality'
    },
    'Price': {
        'price', 'prices', 'cost', 'costs', 'bill', 'money', 'value', 'cheap', 'expensive',
        'charge', 'charges', 'rate', 'rates', 'worth', 'pricing', 'amount', 'tax'
    },
    'Ambience / Location': {
        'ambience', 'ambiance', 'environment', 'decor', 'view', 'location', 'place',
        'music', 'seating', 'table', 'tables', 'atmosphere', 'vibe', 'vibes', 'room',
        'parking', 'interior', 'interiors', 'lighting', 'area'
    },
    'Cleanliness': {
        'clean', 'cleanliness', 'hygiene', 'washroom', 'toilet', 'bathroom', 'restroom',
        'tablecloth', 'tidy', 'dirty', 'neat', 'sanitation'
    }
}


def map_nouns_to_aspects(nouns: set) -> dict:
    """Maps extracted nouns to predefined aspect categories."""
    mapped = {aspect: [] for aspect in ASPECT_MAP}
    for noun in nouns:
        for aspect, keywords in ASPECT_MAP.items():
            if noun.lower() in keywords:
                mapped[aspect].append(noun)
    return {k: v for k, v in mapped.items() if v}


def _get_sia():
    return SentimentIntensityAnalyzer()


def analyze_aspect_sentiments(text: str, target_aspects=None) -> list:
    """
    Analyzes sentiment specifically bound to each detected aspect within the review text.
    Returns a list of dicts with aspect, matched_keywords, sentiment_label, compound_score, and snippet.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    
    sia = _get_sia()
    nouns = extract_nouns(text)
    mapped_aspects = map_nouns_to_aspects(nouns)
    
    # Split text into clauses/sentences to analyze local context per aspect
    clauses = [c.strip() for c in re.split(r'[,.!?;\n]|\bbut\b|\bAND\b|\band\b', cleaned) if c.strip()]
    
    results = []
    aspect_keys = target_aspects if target_aspects else ASPECT_MAP.keys()
    
    for aspect in aspect_keys:
        keywords = ASPECT_MAP.get(aspect, set())
        matched_words = [w for w in nouns if w.lower() in keywords]
        
        # If no explicit noun was extracted, check raw string matching for key aspect terms
        if not matched_words:
            matched_words = [kw for kw in keywords if kw in cleaned.split()]
            
        if matched_words:
            # Find relevant sentence clauses containing these aspect keywords
            relevant_clauses = [c for c in clauses if any(kw in c.split() for kw in matched_words)]
            context_text = " ".join(relevant_clauses) if relevant_clauses else cleaned
            
            scores = sia.polarity_scores(context_text)
            compound = scores['compound']
            
            if compound >= 0.05:
                label = 'Positive'
            elif compound <= -0.05:
                label = 'Negative'
            else:
                label = 'Neutral'
                
            results.append({
                'aspect': aspect,
                'matched_keywords': list(set(matched_words)),
                'sentiment': label,
                'score': round(compound, 3),
                'pos_score': round(scores['pos'], 2),
                'neg_score': round(scores['neg'], 2),
                'neu_score': round(scores['neu'], 2),
                'snippet': context_text[:120] + ('...' if len(context_text) > 120 else '')
            })
            
    return results
