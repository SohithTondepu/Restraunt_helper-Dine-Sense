import re
import nltk

# Ensure required NLTK resources are downloaded quietly without stderr noise
for resource in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass


def clean_text(text: str) -> str:
    """Cleans input text by lowercasing, removing URLs, email addresses, and extra whitespace."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove special HTML tags or noise
    text = re.sub(r'<.*?>', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_nouns(text: str) -> set:
    """Extracts unique noun tokens from cleaned text using NLTK POS tagging or fallback regex tokenizer."""
    cleaned = clean_text(text)
    if not cleaned:
        return set()
    
    try:
        tokens = nltk.word_tokenize(cleaned)
        tagged = nltk.pos_tag(tokens)
        nouns = {word for word, pos in tagged if pos.startswith('NN') and len(word) > 2}
        if nouns:
            return nouns
    except Exception:
        pass

    # Regex tokenization fallback if NLTK tagger is unavailable
    words = re.findall(r'\b[a-z]{3,}\b', cleaned)
    return set(words)
