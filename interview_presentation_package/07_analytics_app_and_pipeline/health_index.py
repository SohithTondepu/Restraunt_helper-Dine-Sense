import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from src.aspect_engine import extract_aspects_from_review, ASPECT_LEXICON

ASPECTS = ['Food', 'Service', 'Price', 'Ambience']


def compute_aspect_health(pos_count: float, neg_count: float, neu_count: float, global_prior: float = 0.35, k: float = 5.0) -> float:
    """
    Computes Empirical Bayes smoothed aspect health score (0 to 100).
    Supports time-decayed continuous observation weights as well as discrete counts.
    raw_polarity = (pos - neg) / (pos + neg + neu)  in [-1, 1]
    smoothed = (N_eff * raw + k * prior) / (N_eff + k)
    score_0_100 = 50 * (smoothed + 1)
    """
    n = float(pos_count + neg_count + neu_count)
    if n <= 1e-9:
        raw_polarity = float(global_prior)
    else:
        raw_polarity = float(pos_count - neg_count) / n
        
    smoothed = (n * raw_polarity + k * float(global_prior)) / float(n + k)
    score = round(float(50.0 * (smoothed + 1.0)), 1)
    return max(0.0, min(100.0, score))


def compute_restaurant_scorecard(
    restaurant_name: str,
    df: pd.DataFrame,
    aspect_results: list = None,
    half_life_days: float = None,
    time_decay_lambda: float = 0.0
) -> dict:
    """
    Builds the aspect health scorecard for a restaurant with optional exponential time decay.
    
    If half_life_days or time_decay_lambda > 0, reviews are weighted by:
        w_i = exp(-lambda * delta_t_i)
    where delta_t_i is review age in days from the most recent review timestamp.
    Recent operational improvements or regressions are given higher significance
    without freezing scores in ancient history.
    """
    rest_df = df[df['Restaurant'] == restaurant_name].copy()
    if rest_df.empty:
        return {}
        
    # Resolve decay lambda
    if half_life_days is not None and half_life_days > 0:
        decay_lambda = float(np.log(2.0) / float(half_life_days))
    else:
        decay_lambda = float(time_decay_lambda) if time_decay_lambda else 0.0

    # Calculate review-level weights if time decay is enabled
    review_weights = {}
    time_decay_applied = False
    
    if decay_lambda > 0.0:
        date_col = None
        for col in ['Time', 'Date', 'Timestamp', 'Review_Date', 'date', 'time']:
            if col in rest_df.columns:
                date_col = col
                break
                
        if date_col is not None:
            parsed_dates = pd.to_datetime(rest_df[date_col], errors='coerce')
            valid_mask = parsed_dates.notna()
            if valid_mask.any():
                max_date = parsed_dates[valid_mask].max()
                # Compute age in days
                delta_days = (max_date - parsed_dates).dt.total_seconds() / 86400.0
                median_delta = float(delta_days[valid_mask].median()) if valid_mask.any() else 0.0
                delta_days = delta_days.fillna(median_delta)
                w_series = np.exp(-decay_lambda * delta_days.values)
                for idx, w in zip(rest_df.index, w_series):
                    review_weights[idx] = float(w)
                time_decay_applied = True

        if not time_decay_applied:
            # Fallback: Sequential recency decay across dataframe index
            n_revs = len(rest_df)
            for i, idx in enumerate(rest_df.index):
                # Assume index 0 is newest
                delta_rank = float(i)
                w = np.exp(-decay_lambda * delta_rank)
                review_weights[idx] = float(w)
            time_decay_applied = True

    # Accumulate aspect counts (both raw and time-weighted)
    counts = {
        asp: {'pos': 0, 'neu': 0, 'neg': 0, 'total': 0, 'pos_wt': 0.0, 'neu_wt': 0.0, 'neg_wt': 0.0, 'total_wt': 0.0}
        for asp in ASPECTS
    }
    
    if aspect_results is not None:
        for res in aspect_results:
            asp = res.get('aspect')
            sent = res.get('sentiment')
            w = float(res.get('weight', 1.0))
            if asp in counts:
                counts[asp]['total'] += 1
                counts[asp]['total_wt'] += w
                if sent == 'Positive':
                    counts[asp]['pos'] += 1
                    counts[asp]['pos_wt'] += w
                elif sent == 'Neutral':
                    counts[asp]['neu'] += 1
                    counts[asp]['neu_wt'] += w
                elif sent == 'Negative':
                    counts[asp]['neg'] += 1
                    counts[asp]['neg_wt'] += w
    else:
        for idx, row in rest_df.iterrows():
            w = review_weights.get(idx, 1.0)
            findings = extract_aspects_from_review(str(row['Review']))
            for res in findings:
                asp = res.get('aspect')
                sent = res.get('sentiment')
                if asp in counts:
                    counts[asp]['total'] += 1
                    counts[asp]['total_wt'] += w
                    if sent == 'Positive':
                        counts[asp]['pos'] += 1
                        counts[asp]['pos_wt'] += w
                    elif sent == 'Neutral':
                        counts[asp]['neu'] += 1
                        counts[asp]['neu_wt'] += w
                    elif sent == 'Negative':
                        counts[asp]['neg'] += 1
                        counts[asp]['neg_wt'] += w
                        
    scorecard = {}
    weighted_sum = 0.0
    
    # Standard business weights: Food=0.40, Service=0.30, Price=0.15, Ambience=0.15
    weights = {'Food': 0.40, 'Service': 0.30, 'Price': 0.15, 'Ambience': 0.15}
    
    for asp in ASPECTS:
        c = counts[asp]
        # Use time-decayed weights if decay is active, otherwise raw counts
        p_wt = c['pos_wt'] if time_decay_applied else c['pos']
        n_wt = c['neg_wt'] if time_decay_applied else c['neg']
        u_wt = c['neu_wt'] if time_decay_applied else c['neu']
        
        score = compute_aspect_health(p_wt, n_wt, u_wt, global_prior=0.35, k=5.0)
        
        status = 'Excellent' if score >= 75.0 else ('Satisfactory' if score >= 55.0 else 'Critical Alert')
        color = '#16A34A' if score >= 75.0 else ('#D97706' if score >= 55.0 else '#DC2626')
        
        scorecard[asp] = {
            'score': score,
            'status': status,
            'color': color,
            'pos_mentions': c['pos'],
            'neu_mentions': c['neu'],
            'neg_mentions': c['neg'],
            'total_mentions': c['total'],
            'effective_weight': round(float(c['total_wt']), 2) if time_decay_applied else float(c['total'])
        }
        weighted_sum += score * weights[asp]
        
    overall_health = round(weighted_sum, 1)
    
    avg_stars = round(float(rest_df['Rating'].mean()), 2)
    total_reviews = len(rest_df)
    
    return {
        'restaurant': restaurant_name,
        'overall_health_index': overall_health,
        'average_stars': avg_stars,
        'total_reviews': total_reviews,
        'time_decay_applied': time_decay_applied,
        'half_life_days': half_life_days,
        'aspects': scorecard
    }


def validate_health_index(scorecards: list) -> dict:
    """
    Validates health index against actual customer star ratings using Spearman rank correlation.
    """
    if not scorecards:
        return {}
        
    indices = [s['overall_health_index'] for s in scorecards]
    stars = [s['average_stars'] for s in scorecards]
    
    rho, p_val = spearmanr(indices, stars)
    return {
        'spearman_correlation': round(float(rho), 4),
        'p_value': float(p_val),
        'statistically_significant': bool(p_val < 0.05)
    }
