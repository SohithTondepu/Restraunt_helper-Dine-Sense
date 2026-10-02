"""
DineSense AI - Core Analytics Aggregation Engine

This module implements the primary reporting and analytics aggregation layer
for aspect-based sentiment analysis of restaurant reviews.

Aggregation Principles:
1. Primary reporting unit is the unique (review_id, aspect) pair.
2. Multiple assertions for the same aspect within a single review are resolved:
   - Both Positive and Negative present -> 'Mixed'
   - Only Positive -> 'Positive'
   - Only Negative -> 'Negative'
   - Only Neutral  -> 'Neutral'
3. 'No Aspect Opinion' is an exclusion status, NOT a Neutral sentiment.
   It is strictly excluded from sentiment-rate denominators.
4. Underlying assertion-level predictions are fully preserved for traceability.
5. Reviews are never referred to as unique customers (no customer IDs exist).
6. Optional Empirical Bayes smoothing uses Beta-Binomial models with priors
   estimated directly from dataset-wide aspect distributions (Method of Moments).
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

VALID_ASPECTS = ['Food', 'Service', 'Price / Value', 'Ambience', 'General Experience']
VALID_SENTIMENTS = ['Positive', 'Negative', 'Neutral']


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ANNOTATIONS_PATH = os.path.join(PROJECT_ROOT, 'final_aspect_evaluation', 'modified_annotations_2000_corrected_mixed.csv')
DEFAULT_RAW_REVIEWS_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'Restaurant reviews.csv')


def load_and_prepare_data(
    annotations_path: Optional[str] = None,
    raw_reviews_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads assertion annotations and merges review metadata (Time, Reviewer)
    from the raw reviews corpus via original_row_id.

    Returns:
        df_assertions: Preserved assertion-level DataFrame with review dates.
        df_review_aspects: Primary aggregated DataFrame at (review_id, aspect) level.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(this_dir)
    grandparent_dir = os.path.dirname(parent_dir)

    search_dirs = [
        PROJECT_ROOT,
        os.getcwd(),
        this_dir,
        parent_dir,
        grandparent_dir
    ]

    if annotations_path is None or not os.path.exists(annotations_path):
        candidates = []
        for base in search_dirs:
            candidates.extend([
                os.path.join(base, 'data', 'processed', 'full_10000_reviews_assertions.csv'),
                os.path.join(base, '01_datasets_and_preprocessing', 'full_10000_reviews_assertions.csv'),
                os.path.join(base, 'interview_presentation_package', '01_datasets_and_preprocessing', 'full_10000_reviews_assertions.csv'),
                os.path.join(base, 'full_10000_reviews_assertions.csv'),
                os.path.join(base, 'data', 'processed', 'modified_annotations_2000_corrected_mixed.csv'),
                os.path.join(base, 'final_aspect_evaluation', 'modified_annotations_2000_corrected_mixed.csv'),
                os.path.join(base, '01_datasets_and_preprocessing', 'modified_annotations_2000_corrected_mixed.csv'),
                os.path.join(base, 'modified_annotations_2000_corrected_mixed.csv'),
            ])
        annotations_path = next((p for p in candidates if os.path.exists(p)), None)
        if not annotations_path:
            raise FileNotFoundError(f"Annotations file not found in candidate paths.")

    if raw_reviews_path is None or not os.path.exists(raw_reviews_path):
        raw_candidates = []
        for base in search_dirs:
            raw_candidates.extend([
                os.path.join(base, 'data', 'raw', 'Restaurant reviews.csv'),
                os.path.join(base, '01_datasets_and_preprocessing', 'Restaurant reviews.csv'),
                os.path.join(base, 'interview_presentation_package', '01_datasets_and_preprocessing', 'Restaurant reviews.csv'),
                os.path.join(base, 'Restaurant reviews.csv')
            ])
        raw_reviews_path = next((p for p in raw_candidates if os.path.exists(p)), None)

    df_assertions = pd.read_csv(annotations_path)

    # Standardize column names
    if 'establishment_id' in df_assertions.columns and 'Restaurant' not in df_assertions.columns:
        df_assertions['Restaurant'] = df_assertions['establishment_id']

    # Initialize or preserve metadata columns with correct dtypes
    if 'Review_Date' in df_assertions.columns and df_assertions['Review_Date'].notna().sum() > 0:
        df_assertions['Review_Date'] = pd.to_datetime(df_assertions['Review_Date'], errors='coerce')
    else:
        df_assertions['Review_Date'] = pd.Series(pd.NaT, index=df_assertions.index, dtype='datetime64[ns]')

    if 'YearMonth' in df_assertions.columns and df_assertions['YearMonth'].notna().sum() > 0:
        df_assertions['YearMonth'] = df_assertions['YearMonth'].astype(str).replace(['NaT', 'nan'], np.nan)
    else:
        df_assertions['YearMonth'] = df_assertions['Review_Date'].dt.to_period('M').astype(str).replace('NaT', np.nan)

    if 'Reviewer' not in df_assertions.columns:
        df_assertions['Reviewer'] = pd.Series('Anonymous', index=df_assertions.index, dtype=object)

    # If dates are missing, fallback to merging from raw reviews
    if df_assertions['Review_Date'].isna().sum() > (0.5 * len(df_assertions)) and raw_reviews_path and os.path.exists(raw_reviews_path):
        raw_df = pd.read_csv(raw_reviews_path)
        raw_df['Review_Date'] = pd.to_datetime(raw_df['Time'], errors='coerce')
        raw_df['YearMonth'] = raw_df['Review_Date'].dt.to_period('M').astype(str).replace('NaT', np.nan)

        if 'original_row_id' in df_assertions.columns:
            valid_mask = (df_assertions['original_row_id'] >= 0) & (df_assertions['original_row_id'] < len(raw_df))
            valid_indices = df_assertions.loc[valid_mask, 'original_row_id'].astype(int)

            df_assertions.loc[valid_mask, 'Review_Date'] = raw_df.loc[valid_indices, 'Review_Date'].values
            df_assertions.loc[valid_mask, 'YearMonth'] = raw_df.loc[valid_indices, 'YearMonth'].values
            if 'Reviewer' in raw_df.columns:
                df_assertions.loc[valid_mask, 'Reviewer'] = raw_df.loc[valid_indices, 'Reviewer'].values

    # Filter out invalid aspect categories
    df_assertions = df_assertions[df_assertions['aspect'].isin(VALID_ASPECTS)].copy()

    # Aggregate to unique review-aspect pairs
    df_review_aspects = aggregate_review_aspect_units(df_assertions)

    return df_assertions, df_review_aspects


def resolve_sentiment_conflict(sentiments: List[str]) -> str:
    """
    Resolves multiple assertion sentiments for the same review-aspect pair:
    - Contains both Positive and Negative -> 'Mixed'
    - Contains Positive, no Negative     -> 'Positive'
    - Contains Negative, no Positive     -> 'Negative'
    - Contains only Neutral              -> 'Neutral'
    - Empty or non-opinion only          -> None
    """
    valid_sents = set(s for s in sentiments if s in VALID_SENTIMENTS)
    if not valid_sents:
        return 'No Aspect Opinion'
    if 'Positive' in valid_sents and 'Negative' in valid_sents:
        return 'Mixed'
    if 'Positive' in valid_sents:
        return 'Positive'
    if 'Negative' in valid_sents:
        return 'Negative'
    if 'Neutral' in valid_sents:
        return 'Neutral'
    return 'No Aspect Opinion'


def aggregate_review_aspect_units(df_assertions: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates assertion records into unique (review_id, aspect) reporting units.
    Preserves restaurant, date, and rating metadata.
    """
    # Exclude No Aspect Opinion from sentiment consideration
    if 'annotation_status' in df_assertions.columns:
        status_mask = (df_assertions['annotation_status'] != 'No Aspect Opinion')
    else:
        status_mask = pd.Series(True, index=df_assertions.index)

    opinion_mask = status_mask & (df_assertions['sentiment'].isin(VALID_SENTIMENTS))

    df_opinions = df_assertions[opinion_mask].copy()

    # Group by review_id and aspect
    records = []
    grouped = df_opinions.groupby(['review_id', 'aspect'])

    for (review_id, aspect), grp in grouped:
        sents = grp['sentiment'].tolist()
        resolved_sent = resolve_sentiment_conflict(sents)

        first_row = grp.iloc[0]
        records.append({
            'review_id': review_id,
            'aspect': aspect,
            'Restaurant': first_row.get('Restaurant', first_row.get('establishment_id', 'Unknown')),
            'sentiment': resolved_sent,
            'star_rating': first_row.get('star_rating', np.nan),
            'Review_Date': first_row.get('Review_Date', pd.NaT),
            'YearMonth': first_row.get('YearMonth', np.nan),
            'Reviewer': first_row.get('Reviewer', 'Anonymous'),
            'clause_count': len(grp),
            'sample_clause': first_row.get('clause_text', '')
        })

    df_review_aspects = pd.DataFrame(records)
    return df_review_aspects


def calculate_aspect_metrics(
    df_review_aspects: pd.DataFrame,
    df_all_reviews: pd.DataFrame,
    restaurant: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculates aspect-wise metrics for a given restaurant or all restaurants.

    Metrics per aspect:
    - eligible_reviews: Total eligible reviews in the selected scope
    - aspect_mentions: Unique reviews mentioning this aspect
    - mention_rate: aspect_mentions / eligible_reviews
    - positive_count, negative_count, neutral_count, mixed_count
    - positive_pct, negative_pct, neutral_pct, mixed_pct (denominator: aspect_mentions)
    """
    # Filter scope
    if restaurant and restaurant != 'All Restaurants':
        df_ra_sub = df_review_aspects[df_review_aspects['Restaurant'] == restaurant]
        df_rev_sub = df_all_reviews[df_all_reviews['Restaurant'] == restaurant]
    else:
        df_ra_sub = df_review_aspects
        df_rev_sub = df_all_reviews

    total_eligible_reviews = df_rev_sub['review_id'].nunique()
    if total_eligible_reviews == 0:
        total_eligible_reviews = df_ra_sub['review_id'].nunique()

    aspect_rows = []

    for aspect in VALID_ASPECTS:
        ra_aspect = df_ra_sub[df_ra_sub['aspect'] == aspect]
        mention_count = ra_aspect['review_id'].nunique()

        mention_rate = (mention_count / total_eligible_reviews) if total_eligible_reviews > 0 else 0.0

        pos = (ra_aspect['sentiment'] == 'Positive').sum()
        neg = (ra_aspect['sentiment'] == 'Negative').sum()
        neu = (ra_aspect['sentiment'] == 'Neutral').sum()
        mix = (ra_aspect['sentiment'] == 'Mixed').sum()

        pos_pct = (pos / mention_count * 100.0) if mention_count > 0 else 0.0
        neg_pct = (neg / mention_count * 100.0) if mention_count > 0 else 0.0
        neu_pct = (neu / mention_count * 100.0) if mention_count > 0 else 0.0
        mix_pct = (mix / mention_count * 100.0) if mention_count > 0 else 0.0

        aspect_rows.append({
            'Aspect': aspect,
            'Eligible_Reviews': total_eligible_reviews,
            'Mention_Count': mention_count,
            'Mention_Rate_Pct': round(mention_rate * 100.0, 1),
            'Positive_Count': int(pos),
            'Positive_Pct': round(pos_pct, 1),
            'Negative_Count': int(neg),
            'Negative_Pct': round(neg_pct, 1),
            'Neutral_Count': int(neu),
            'Neutral_Pct': round(neu_pct, 1),
            'Mixed_Count': int(mix),
            'Mixed_Pct': round(mix_pct, 1)
        })

    return pd.DataFrame(aspect_rows)


def calculate_review_level_metrics(
    df_review_aspects: pd.DataFrame,
    df_all_reviews: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Calculates overall review-level sentiment based on a documented, defensible rule:
    A review is:
    - 'Negative' if it contains ANY Negative aspect sentiment.
    - 'Mixed' if it contains both Positive and Mixed/Neutral aspects without Negative.
    - 'Positive' if all mentioned aspects are Positive.
    - 'Neutral' if all mentioned aspects are Neutral.
    """
    if df_review_aspects.empty:
        return pd.DataFrame(columns=['review_id', 'Restaurant', 'Overall_Sentiment', 'Aspects_Mentioned_Count'])

    review_rows = []
    if df_all_reviews is not None and 'review_id' in df_all_reviews.columns and not df_all_reviews.empty:
        all_rids = df_all_reviews['review_id'].unique()
    else:
        all_rids = df_review_aspects['review_id'].unique()

    for rid in all_rids:
        sub = df_review_aspects[df_review_aspects['review_id'] == rid]
        if sub.empty:
            continue
        rest = sub.iloc[0]['Restaurant']
        sents = set(sub['sentiment'].tolist())

        if 'Negative' in sents:
            overall = 'Negative'
        elif 'Mixed' in sents:
            overall = 'Mixed'
        elif sents == {'Positive'}:
            overall = 'Positive'
        elif sents == {'Neutral'}:
            overall = 'Neutral'
        elif 'Positive' in sents and 'Neutral' in sents:
            overall = 'Positive'
        else:
            overall = 'Neutral'

        review_rows.append({
            'review_id': rid,
            'Restaurant': rest,
            'Overall_Sentiment': overall,
            'Aspects_Mentioned_Count': len(sub)
        })

    if not review_rows:
        return pd.DataFrame(columns=['review_id', 'Restaurant', 'Overall_Sentiment', 'Aspects_Mentioned_Count'])

    return pd.DataFrame(review_rows)


def calculate_time_trends(
    df_review_aspects: pd.DataFrame,
    restaurant: Optional[str] = None,
    aspect: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculates monthly review volume and sentiment trends.
    Uses actual review dates ('YearMonth'). Excludes missing/NaT dates.
    """
    df_valid = df_review_aspects.dropna(subset=['YearMonth']).copy()
    # Filter out NaT strings
    df_valid = df_valid[df_valid['YearMonth'].str.match(r'^\d{4}-\d{2}$', na=False)]

    if restaurant and restaurant != 'All Restaurants':
        df_valid = df_valid[df_valid['Restaurant'] == restaurant]
    if aspect and aspect != 'All Aspects':
        df_valid = df_valid[df_valid['aspect'] == aspect]

    if df_valid.empty:
        return pd.DataFrame()

    trend_rows = []
    for ym, grp in df_valid.groupby('YearMonth'):
        n_reviews = grp['review_id'].nunique()
        pos = (grp['sentiment'] == 'Positive').sum()
        neg = (grp['sentiment'] == 'Negative').sum()
        neu = (grp['sentiment'] == 'Neutral').sum()
        mix = (grp['sentiment'] == 'Mixed').sum()

        trend_rows.append({
            'YearMonth': ym,
            'Sample_Size': n_reviews,
            'Total_Mentions': n_reviews,
            'Positive_Count': int(pos),
            'Positive_Pct': round(pos / n_reviews * 100.0, 1) if n_reviews > 0 else 0.0,
            'Negative_Count': int(neg),
            'Negative_Pct': round(neg / n_reviews * 100.0, 1) if n_reviews > 0 else 0.0,
            'Neutral_Count': int(neu),
            'Neutral_Pct': round(neu / n_reviews * 100.0, 1) if n_reviews > 0 else 0.0,
            'Mixed_Count': int(mix),
            'Mixed_Pct': round(mix / n_reviews * 100.0, 1) if n_reviews > 0 else 0.0,
            'Small_Sample_Warning': n_reviews < 10,
            'Sample_Size_Warning': n_reviews < 10
        })

    df_trend = pd.DataFrame(trend_rows).sort_values('YearMonth').reset_index(drop=True)
    return df_trend


def calculate_peer_comparison(
    df_review_aspects: pd.DataFrame,
    target_restaurant: str,
    peer_restaurants: List[str],
    aspect: str
) -> pd.DataFrame:
    """
    Compares the target restaurant against peer restaurants for a selected aspect.
    Uses the exact same eligibility, definitions, and review-aspect reporting units.
    """
    comp_list = [target_restaurant] + [r for r in peer_restaurants if r != target_restaurant]
    rows = []

    for rest in comp_list:
        sub = df_review_aspects[(df_review_aspects['Restaurant'] == rest) & (df_review_aspects['aspect'] == aspect)]
        n = len(sub)
        if n == 0:
            rows.append({
                'Restaurant': rest,
                'Role': 'Target' if rest == target_restaurant else 'Peer',
                'Reviews_Mentioning_Aspect': 0,
                'Mention_Count': 0,
                'Positive_Pct': 0.0,
                'Negative_Pct': 0.0,
                'Neutral_Pct': 0.0,
                'Mixed_Pct': 0.0,
                'Positive_Count': 0,
                'Negative_Count': 0,
                'Neutral_Count': 0,
                'Mixed_Count': 0
            })
            continue

        pos = (sub['sentiment'] == 'Positive').sum()
        neg = (sub['sentiment'] == 'Negative').sum()
        neu = (sub['sentiment'] == 'Neutral').sum()
        mix = (sub['sentiment'] == 'Mixed').sum()

        rows.append({
            'Restaurant': rest,
            'Role': 'Target' if rest == target_restaurant else 'Peer',
            'Reviews_Mentioning_Aspect': n,
            'Mention_Count': n,
            'Positive_Pct': round(pos / n * 100.0, 1),
            'Negative_Pct': round(neg / n * 100.0, 1),
            'Neutral_Pct': round(neu / n * 100.0, 1),
            'Mixed_Pct': round(mix / n * 100.0, 1),
            'Positive_Count': int(pos),
            'Negative_Count': int(neg),
            'Neutral_Count': int(neu),
            'Mixed_Count': int(mix)
        })

    return pd.DataFrame(rows)


# =========================================================================
# PHASE 4: OPTIONAL EMPIRICAL BAYES SMOOTHING
# =========================================================================

class AspectBetaPrior(dict):
    """Holds estimated Beta prior parameters, supporting both dict and tuple access."""
    def __init__(self, prior_mean: float, alpha: float, beta: float):
        super().__init__(
            prior_mean=prior_mean,
            alpha=alpha,
            beta=beta,
            M=alpha + beta
        )
    def __getitem__(self, key):
        if isinstance(key, int):
            return [self['prior_mean'], self['alpha'], self['beta'], self['M']][key]
        return super().__getitem__(key)


def estimate_aspect_beta_priors(df_review_aspects: pd.DataFrame, target_sentiment: str = 'Negative') -> Dict[str, AspectBetaPrior]:
    """
    Estimates aspect-specific Beta priors (alpha_0, beta_0) across all restaurants
    using Method of Moments on restaurant-level rates.

    Returns:
        Dict[aspect, AspectBetaPrior]
    """
    priors = {}

    for aspect in VALID_ASPECTS:
        sub = df_review_aspects[df_review_aspects['aspect'] == aspect]
        rest_rates = []
        for rest, grp in sub.groupby('Restaurant'):
            n = len(grp)
            if n >= 5:  # Filter for restaurants with meaningful volume to estimate dispersion
                rate = (grp['sentiment'] == target_sentiment).sum() / n
                rest_rates.append(rate)

        if len(rest_rates) >= 3:
            p_bar = float(np.mean(rest_rates))
            v_bar = float(np.var(rest_rates, ddof=1))

            # Method of moments for Beta distribution
            if 0 < v_bar < (p_bar * (1 - p_bar)):
                common = (p_bar * (1 - p_bar) / v_bar) - 1.0
                alpha_0 = p_bar * common
                beta_0 = (1.0 - p_bar) * common
            else:
                # Robust default pseudo-counts if variance is saturated
                alpha_0 = p_bar * 10.0
                beta_0 = (1.0 - p_bar) * 10.0
        else:
            p_bar = 0.20
            alpha_0 = 2.0
            beta_0 = 8.0

        priors[aspect] = AspectBetaPrior(p_bar, alpha_0, beta_0)

    return priors


def apply_empirical_bayes_smoothing(
    df_aspect_metrics: pd.DataFrame,
    priors_or_review_aspects: Any,
    target_sentiment: str = 'Negative'
) -> pd.DataFrame:
    """
    Applies Beta-Binomial Empirical Bayes smoothing to target sentiment rate.
    Keeps raw count and rate side-by-side with smoothed rate.
    Accepts either pre-computed priors dict or df_review_aspects.
    """
    if isinstance(priors_or_review_aspects, dict):
        priors = priors_or_review_aspects
    else:
        priors = estimate_aspect_beta_priors(priors_or_review_aspects, target_sentiment=target_sentiment)

    df_out = df_aspect_metrics.copy()

    smoothed_pcts = []
    prior_means = []
    prior_weights = []

    count_col = f"{target_sentiment}_Count"

    for _, r in df_out.iterrows():
        aspect = r['Aspect']
        p_info = priors.get(aspect, (0.2, 2.0, 8.0))
        if isinstance(p_info, (tuple, list)):
            p_bar, a0, b0 = p_info[0], p_info[1], p_info[2]
        elif isinstance(p_info, dict):
            p_bar = p_info.get('prior_mean', 0.2)
            a0 = p_info.get('alpha', 2.0)
            b0 = p_info.get('beta', 8.0)
        else:
            p_bar, a0, b0 = 0.2, 2.0, 8.0

        n = r.get('Mention_Count', 0)
        k = r.get(count_col, 0)

        if n > 0:
            smoothed = (k + a0) / (n + a0 + b0)
        else:
            smoothed = p_bar

        smoothed_pcts.append(round(smoothed * 100.0, 1))
        prior_means.append(round(p_bar * 100.0, 1))
        prior_weights.append(round(a0 + b0, 1))

    df_out[f"{target_sentiment}_Smoothed_Pct"] = smoothed_pcts
    df_out[f"{target_sentiment}_Pct_Smoothed"] = smoothed_pcts
    df_out[f"{target_sentiment}_Prior_Mean_Pct"] = prior_means
    df_out[f"{target_sentiment}_Prior_Mean"] = [p / 100.0 for p in prior_means]
    df_out['EB_Prior_Strength_N'] = prior_weights
    df_out['Prior_Weight_M'] = prior_weights

    return df_out


# =========================================================================
# PHASE 5: FACTUAL DESCRIPTIVE INSIGHTS
# =========================================================================

def generate_descriptive_insights(
    df_aspect_metrics: pd.DataFrame,
    df_time_trends: Optional[pd.DataFrame] = None,
    df_peer_comp: Optional[pd.DataFrame] = None,
    restaurant_name: str = "This restaurant"
) -> List[Dict[str, Any]]:
    """
    Generates concise, factual observations from computed metrics.
    Strictly avoids:
    - Inferred causes
    - SOPs or recommendations
    - Complaint clustering
    """
    if isinstance(df_time_trends, str):
        restaurant_name = df_time_trends
        df_time_trends = None

    insights = []

    if df_aspect_metrics.empty:
        return insights

    # 1. Highest aspect mention rate
    df_sorted_mention = df_aspect_metrics.sort_values('Mention_Count', ascending=False)
    top_aspect = df_sorted_mention.iloc[0]
    if top_aspect['Mention_Count'] > 0:
        insights.append({
            'Category': 'Aspect Volume',
            'Observation': f"{top_aspect['Aspect']} has the highest mention rate at {top_aspect['Mention_Rate_Pct']}% ({top_aspect['Mention_Count']} out of {top_aspect['Eligible_Reviews']} eligible reviews).",
            'Metric': f"Mention Rate: {top_aspect['Mention_Rate_Pct']}%",
            'Sample_Size': top_aspect['Mention_Count']
        })

    # 2. Lowest aspect mention rate
    low_aspect = df_sorted_mention.iloc[-1]
    if low_aspect['Mention_Count'] >= 0 and low_aspect['Aspect'] != top_aspect['Aspect']:
        insights.append({
            'Category': 'Aspect Volume',
            'Observation': f"{low_aspect['Aspect']} has the lowest mention rate at {low_aspect['Mention_Rate_Pct']}% ({low_aspect['Mention_Count']} reviews).",
            'Metric': f"Mention Rate: {low_aspect['Mention_Rate_Pct']}%",
            'Sample_Size': low_aspect['Mention_Count']
        })

    # 3. Highest negative sentiment share
    mentioned_only = df_aspect_metrics[df_aspect_metrics['Mention_Count'] >= 3]
    if not mentioned_only.empty:
        top_neg = mentioned_only.sort_values('Negative_Pct', ascending=False).iloc[0]
        if top_neg['Negative_Pct'] > 0:
            insights.append({
                'Category': 'Sentiment Distribution',
                'Observation': f"{top_neg['Aspect']} exhibits the highest negative sentiment share at {top_neg['Negative_Pct']}% ({top_neg['Negative_Count']} negative out of {top_neg['Mention_Count']} reviews mentioning this aspect).",
                'Metric': f"Negative Share: {top_neg['Negative_Pct']}%",
                'Sample_Size': top_neg['Mention_Count']
            })

    # 4. Highest positive sentiment share
    if not mentioned_only.empty:
        top_pos = mentioned_only.sort_values('Positive_Pct', ascending=False).iloc[0]
        insights.append({
            'Category': 'Sentiment Distribution',
            'Observation': f"{top_pos['Aspect']} exhibits the highest positive sentiment share at {top_pos['Positive_Pct']}% ({top_pos['Positive_Count']} positive out of {top_pos['Mention_Count']} reviews mentioning this aspect).",
            'Metric': f"Positive Share: {top_pos['Positive_Pct']}%",
            'Sample_Size': top_pos['Mention_Count']
        })

    # 5. Time trend insight
    if df_time_trends is not None and len(df_time_trends) >= 2:
        recent = df_time_trends.iloc[-1]
        prior = df_time_trends.iloc[-2]
        delta_neg = round(recent['Negative_Pct'] - prior['Negative_Pct'], 1)
        direction = "increased" if delta_neg > 0 else "decreased" if delta_neg < 0 else "remained unchanged"

        warning = " (Note: small sample size)" if recent['Small_Sample_Warning'] else ""
        insights.append({
            'Category': 'Time Trend',
            'Observation': f"Negative sentiment share {direction} by {abs(delta_neg)}% from {prior['YearMonth']} ({prior['Negative_Pct']}%, N={prior['Sample_Size']}) to {recent['YearMonth']} ({recent['Negative_Pct']}%, N={recent['Sample_Size']}){warning}.",
            'Metric': f"Trend Shift: {delta_neg:+0.1f}%",
            'Sample_Size': recent['Sample_Size']
        })

    # 6. Peer comparison insight
    if df_peer_comp is not None and len(df_peer_comp) >= 2:
        target_row = df_peer_comp[df_peer_comp['Role'] == 'Target']
        peers = df_peer_comp[df_peer_comp['Role'] == 'Peer']

        if not target_row.empty and not peers.empty:
            t_neg = target_row.iloc[0]['Negative_Pct']
            t_n = target_row.iloc[0]['Reviews_Mentioning_Aspect']
            peer_avg_neg = round(peers['Negative_Pct'].mean(), 1)
            diff = round(t_neg - peer_avg_neg, 1)
            rel_str = "higher than" if diff > 0 else "lower than" if diff < 0 else "equal to"

            insights.append({
                'Category': 'Peer Comparison',
                'Observation': f"For the evaluated aspect, {restaurant_name}'s negative share ({t_neg}%, N={t_n}) is {abs(diff)}% {rel_str} the selected peer group average ({peer_avg_neg}%).",
                'Metric': f"Peer Difference: {diff:+0.1f}%",
                'Sample_Size': t_n
            })

    return insights
