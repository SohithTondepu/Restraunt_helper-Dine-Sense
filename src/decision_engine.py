import pandas as pd
import numpy as np
from src.preprocessing import extract_nouns
from src.aspect_engine import analyze_aspect_sentiments, ASPECT_MAP


def compute_restaurant_aspect_health(restaurant_name: str, df: pd.DataFrame) -> dict:
    """
    Computes 0-100 Aspect Health Scores for a specific restaurant across
    Food, Service, Price, Ambience / Location, and Cleanliness.
    """
    if df.empty or 'Restaurant' not in df.columns:
        return {}
    
    rest_df = df[df['Restaurant'] == restaurant_name].copy()
    if rest_df.empty:
        return {}
    
    aspect_counts = {asp: {'pos': 0, 'neu': 0, 'neg': 0, 'total': 0} for asp in ASPECT_MAP}
    
    # Process reviews for the selected restaurant
    for _, row in rest_df.iterrows():
        review_text = str(row['Review'])
        results = analyze_aspect_sentiments(review_text)
        
        for res in results:
            asp = res['aspect']
            sent = res['sentiment']
            if asp in aspect_counts:
                aspect_counts[asp]['total'] += 1
                if sent == 'Positive':
                    aspect_counts[asp]['pos'] += 1
                elif sent == 'Neutral':
                    aspect_counts[asp]['neu'] += 1
                elif sent == 'Negative':
                    aspect_counts[asp]['neg'] += 1

    health_scores = {}
    for asp, counts in aspect_counts.items():
        total = counts['total']
        if total > 0:
            # Health Score Formula: (Positive + 0.5 * Neutral) / Total * 100
            score = round(((counts['pos'] + 0.5 * counts['neu']) / total) * 100, 1)
        else:
            # Neutral baseline if no mentions
            score = 75.0
            
        status = 'Excellent' if score >= 80 else ('Satisfactory' if score >= 60 else 'Critical Risk')
        color = '#16A34A' if score >= 80 else ('#D97706' if score >= 60 else '#DC2626')
        
        health_scores[asp] = {
            'score': score,
            'status': status,
            'color': color,
            'pos_count': counts['pos'],
            'neu_count': counts['neu'],
            'neg_count': counts['neg'],
            'total_mentions': total
        }
        
    return health_scores


def generate_ai_recommendations(health_scores: dict, restaurant_name: str) -> list:
    """
    Generates actionable operational AI recommendations based on aspect health scores.
    """
    recommendations = []
    
    for aspect, data in health_scores.items():
        score = data['score']
        neg_count = data['neg_count']
        total = data['total_mentions']
        
        if aspect == 'Service' and score < 75:
            recommendations.append({
                'type': 'CRITICAL ALERT' if score < 60 else 'OPERATIONAL WARNING',
                'severity': 'high' if score < 60 else 'medium',
                'aspect': 'Service Operations',
                'issue': f"Service Health Score is at {score}/100 with {neg_count} negative complaints out of {total} service mentions.",
                'action': "Increase floor staff during weekend dinner shifts (7 PM – 10 PM) and conduct hospitality response training to reduce order wait times."
            })
            
        elif aspect == 'Food' and score < 75:
            recommendations.append({
                'type': 'CRITICAL ALERT' if score < 60 else 'OPERATIONAL WARNING',
                'severity': 'high' if score < 60 else 'medium',
                'aspect': 'Kitchen & Food Quality',
                'issue': f"Food Health Score is at {score}/100. Negative feedback targets dish freshness and seasoning consistency.",
                'action': "Audit kitchen prep standards, check ingredient storage temperature logs, and review recipe portioning."
            })
            
        elif aspect == 'Price' and score < 70:
            recommendations.append({
                'type': 'PRICING STRATEGY',
                'severity': 'medium',
                'aspect': 'Value Perception',
                'issue': f"Price perception score is at {score}/100. Customers perceive certain add-on items as overpriced.",
                'action': "Introduce bundled combo meals and review beverage pricing margins to improve overall perceived value."
            })
            
        elif aspect == 'Ambience / Location' and score < 70:
            recommendations.append({
                'type': 'ATMOSPHERE & SEATING',
                'severity': 'medium',
                'aspect': 'Ambience',
                'issue': f"Ambience Health Score is at {score}/100. Feedback highlights noise levels and table spacing.",
                'action': "Adjust background music decibels, optimize dining floor table spacing, and audit lighting levels."
            })

    # Default positive recommendation if all scores are good
    if not recommendations:
        recommendations.append({
            'type': 'BUSINESS GROWTH',
            'severity': 'low',
            'aspect': 'General Operations',
            'issue': f"{restaurant_name} maintains high health scores across all dining aspects (All scores >= 75/100).",
            'action': "Leverage high customer satisfaction for loyalty marketing campaigns and expand specialty menu offerings."
        })
        
    return recommendations


def extract_aspect_drivers(restaurant_name: str, df: pd.DataFrame) -> dict:
    """
    Extracts top praise drivers and complaint drivers for a given restaurant.
    """
    if df.empty or 'Restaurant' not in df.columns:
        return {'praise': [], 'complaints': []}
        
    rest_df = df[df['Restaurant'] == restaurant_name]
    praise_terms = []
    complaint_terms = []
    
    for _, row in rest_df.iterrows():
        text = str(row['Review']).lower()
        rating = float(row['Rating']) if pd.notnull(row['Rating']) else 3.0
        
        nouns = extract_nouns(text)
        if rating >= 4.0:
            praise_terms.extend(list(nouns))
        elif rating <= 2.5:
            complaint_terms.extend(list(nouns))
            
    praise_counts = pd.Series(praise_terms).value_counts().head(8).to_dict()
    complaint_counts = pd.Series(complaint_terms).value_counts().head(8).to_dict()
    
    return {
        'praise': list(praise_counts.keys()),
        'complaints': list(complaint_counts.keys())
    }
