import os
import joblib
import pandas as pd
from src.preprocessing import clean_text

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
LOGREG_PATH = os.path.join(MODEL_DIR, 'LogisticRegression.joblib')
TFIDF_PATH = os.path.join(MODEL_DIR, 'tfidf.pkl')


def predict_overall_sentiment(text: str) -> dict:
    """
    Predicts overall review sentiment rating using saved Logistic Regression / ML model,
    with fallback to VADER/heuristics if model pickle is unavailable.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return {'sentiment': 'Neutral', 'confidence': 0.5}
        
    if os.path.exists(LOGREG_PATH) and os.path.exists(TFIDF_PATH):
        try:
            model = joblib.load(LOGREG_PATH)
            vectorizer = joblib.load(TFIDF_PATH)
            
            vec = vectorizer.transform([cleaned])
            pred_rating = model.predict(vec)[0]
            
            if pred_rating >= 4:
                label = 'Positive'
            elif pred_rating <= 2:
                label = 'Negative'
            else:
                label = 'Neutral'
                
            return {
                'predicted_rating': float(pred_rating),
                'sentiment': label,
                'source': 'Trained LogisticRegression Baseline'
            }
        except Exception:
            pass
            
    # Fallback to simple rule/vader
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(cleaned)
        compound = scores['compound']
        label = 'Positive' if compound >= 0.05 else ('Negative' if compound <= -0.05 else 'Neutral')
        return {
            'predicted_rating': round(3.0 + compound * 2.0, 1),
            'sentiment': label,
            'source': 'VADER Polarity Fallback Engine'
        }
    except Exception:
        return {'predicted_rating': 3.0, 'sentiment': 'Neutral', 'source': 'Baseline Fallback'}
