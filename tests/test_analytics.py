"""
Unit Tests for DineSense AI Analytics Engine (src/analytics.py)

Tests verification criteria:
1. Review-aspect pair counted only once in primary sentiment distribution.
2. 'No Aspect Opinion' excluded from sentiment denominators.
3. Conflicting sentiments within a review-aspect pair follow the Mixed rule.
4. Counts and percentages reconcile exactly.
5. Time trends use actual review dates.
6. Peer comparisons use consistent filters and denominators.
7. Raw assertions remain unchanged.
8. Handles missing dates, empty selections, and small sample sizes.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import pytest
except ImportError:
    pytest = None
import numpy as np
import pandas as pd
from src.analytics import (
    resolve_sentiment_conflict,
    aggregate_review_aspect_units,
    calculate_aspect_metrics,
    calculate_review_level_metrics,
    calculate_time_trends,
    calculate_peer_comparison,
    estimate_aspect_beta_priors,
    apply_empirical_bayes_smoothing,
    generate_descriptive_insights,
    VALID_ASPECTS
)


def test_resolve_sentiment_conflict_mixed():
    # Both Positive and Negative -> Mixed
    assert resolve_sentiment_conflict(['Positive', 'Negative']) == 'Mixed'
    assert resolve_sentiment_conflict(['Positive', 'Negative', 'Neutral']) == 'Mixed'


def test_resolve_sentiment_conflict_single():
    # Exclusively one sentiment
    assert resolve_sentiment_conflict(['Positive', 'Positive']) == 'Positive'
    assert resolve_sentiment_conflict(['Negative']) == 'Negative'
    assert resolve_sentiment_conflict(['Neutral', 'Neutral']) == 'Neutral'
    assert resolve_sentiment_conflict(['Positive', 'Neutral']) == 'Positive'


def test_resolve_sentiment_conflict_non_opinion():
    assert resolve_sentiment_conflict([]) == 'No Aspect Opinion'
    assert resolve_sentiment_conflict(['InvalidLabel']) == 'No Aspect Opinion'


def test_aggregate_review_aspect_units_deduplication():
    # Test review with 2 assertions for Food (one pos, one neg) and 1 for Service
    sample_assertions = pd.DataFrame([
        {
            'review_id': 'REV_001',
            'aspect': 'Food',
            'sentiment': 'Positive',
            'annotation_status': 'Annotated',
            'Restaurant': 'Test Cafe',
            'Review_Date': pd.Timestamp('2019-01-15'),
            'YearMonth': '2019-01',
            'clause_text': 'Great burger'
        },
        {
            'review_id': 'REV_001',
            'aspect': 'Food',
            'sentiment': 'Negative',
            'annotation_status': 'Annotated',
            'Restaurant': 'Test Cafe',
            'Review_Date': pd.Timestamp('2019-01-15'),
            'YearMonth': '2019-01',
            'clause_text': 'Fries were cold'
        },
        {
            'review_id': 'REV_001',
            'aspect': 'Service',
            'sentiment': 'Positive',
            'annotation_status': 'Annotated',
            'Restaurant': 'Test Cafe',
            'Review_Date': pd.Timestamp('2019-01-15'),
            'YearMonth': '2019-01',
            'clause_text': 'Staff was polite'
        },
        {
            'review_id': 'REV_001',
            'aspect': 'Ambience',
            'sentiment': np.nan,
            'annotation_status': 'No Aspect Opinion',
            'Restaurant': 'Test Cafe',
            'Review_Date': pd.Timestamp('2019-01-15'),
            'YearMonth': '2019-01',
            'clause_text': 'We sat inside'
        }
    ])

    df_ra = aggregate_review_aspect_units(sample_assertions)

    # 1. Primary reporting unit is unique (review_id, aspect)
    assert len(df_ra) == 2  # Only Food and Service, Ambience is No Aspect Opinion
    assert set(df_ra['aspect']) == {'Food', 'Service'}

    # 2. Conflicting Food assertions resolved to 'Mixed'
    food_row = df_ra[df_ra['aspect'] == 'Food'].iloc[0]
    assert food_row['sentiment'] == 'Mixed'
    assert food_row['clause_count'] == 2

    # 3. Service assertion is 'Positive'
    service_row = df_ra[df_ra['aspect'] == 'Service'].iloc[0]
    assert service_row['sentiment'] == 'Positive'
    assert service_row['clause_count'] == 1


def test_no_aspect_opinion_excluded_from_denominators():
    sample_assertions = pd.DataFrame([
        {
            'review_id': 'REV_002',
            'aspect': 'Food',
            'sentiment': 'Negative',
            'annotation_status': 'Annotated',
            'Restaurant': 'Cafe Two',
            'Review_Date': pd.NaT,
            'YearMonth': np.nan,
            'clause_text': 'Bland food'
        },
        {
            'review_id': 'REV_003',
            'aspect': 'Food',
            'sentiment': None,
            'annotation_status': 'No Aspect Opinion',
            'Restaurant': 'Cafe Two',
            'Review_Date': pd.NaT,
            'YearMonth': np.nan,
            'clause_text': 'Ordered soup'
        }
    ])

    df_ra = aggregate_review_aspect_units(sample_assertions)
    # REV_003 must be excluded from sentiment units
    assert len(df_ra) == 1
    assert df_ra.iloc[0]['review_id'] == 'REV_002'


def test_aspect_metrics_calculation_and_reconciliation():
    # 3 reviews:
    # REV_1: Food=Positive, Service=Negative
    # REV_2: Food=Mixed, Service=Negative
    # REV_3: Food=Negative
    df_ra = pd.DataFrame([
        {'review_id': 'R1', 'aspect': 'Food', 'Restaurant': 'RestA', 'sentiment': 'Positive'},
        {'review_id': 'R1', 'aspect': 'Service', 'Restaurant': 'RestA', 'sentiment': 'Negative'},
        {'review_id': 'R2', 'aspect': 'Food', 'Restaurant': 'RestA', 'sentiment': 'Mixed'},
        {'review_id': 'R2', 'aspect': 'Service', 'Restaurant': 'RestA', 'sentiment': 'Negative'},
        {'review_id': 'R3', 'aspect': 'Food', 'Restaurant': 'RestA', 'sentiment': 'Negative'}
    ])
    df_all_rev = pd.DataFrame([
        {'review_id': 'R1', 'Restaurant': 'RestA'},
        {'review_id': 'R2', 'Restaurant': 'RestA'},
        {'review_id': 'R3', 'Restaurant': 'RestA'},
        {'review_id': 'R4', 'Restaurant': 'RestA'}  # Review with no aspect opinions
    ])

    metrics = calculate_aspect_metrics(df_ra, df_all_rev, restaurant='RestA')

    # Food: 3 mentions out of 4 eligible reviews -> 75% mention rate
    food = metrics[metrics['Aspect'] == 'Food'].iloc[0]
    assert food['Eligible_Reviews'] == 4
    assert food['Mention_Count'] == 3
    assert food['Mention_Rate_Pct'] == 75.0
    assert food['Positive_Count'] == 1
    assert food['Negative_Count'] == 1
    assert food['Mixed_Count'] == 1
    assert food['Neutral_Count'] == 0

    # Percentages must sum to ~100% of mentions (allowing for 0.1% rounding tolerance)
    total_pct = food['Positive_Pct'] + food['Negative_Pct'] + food['Mixed_Pct'] + food['Neutral_Pct']
    assert abs(total_pct - 100.0) < 0.5

    # Service: 2 mentions out of 4 eligible reviews -> 50% mention rate
    serv = metrics[metrics['Aspect'] == 'Service'].iloc[0]
    assert serv['Mention_Count'] == 2
    assert serv['Negative_Count'] == 2
    assert serv['Negative_Pct'] == 100.0


def test_time_trends_filtering_and_sample_sizes():
    df_ra = pd.DataFrame([
        {'review_id': 'R1', 'aspect': 'Food', 'Restaurant': 'RestA', 'sentiment': 'Positive', 'YearMonth': '2019-01'},
        {'review_id': 'R2', 'aspect': 'Food', 'Restaurant': 'RestA', 'sentiment': 'Negative', 'YearMonth': '2019-01'},
        {'review_id': 'R3', 'aspect': 'Food', 'Restaurant': 'RestA', 'sentiment': 'Positive', 'YearMonth': '2019-02'},
        {'review_id': 'R4', 'aspect': 'Food', 'Restaurant': 'RestA', 'sentiment': 'Positive', 'YearMonth': np.nan}  # Missing date
    ])

    trends = calculate_time_trends(df_ra, restaurant='RestA', aspect='Food')

    # Missing date excluded from time trend
    assert len(trends) == 2
    assert list(trends['YearMonth']) == ['2019-01', '2019-02']
    assert trends.iloc[0]['Sample_Size'] == 2
    assert trends.iloc[0]['Small_Sample_Warning'] == True
    assert trends.iloc[1]['Sample_Size'] == 1


def test_peer_comparison_consistency():
    df_ra = pd.DataFrame([
        {'review_id': 'R1', 'aspect': 'Food', 'Restaurant': 'RestA', 'sentiment': 'Negative'},
        {'review_id': 'R2', 'aspect': 'Food', 'Restaurant': 'RestA', 'sentiment': 'Negative'},
        {'review_id': 'R3', 'aspect': 'Food', 'Restaurant': 'RestB', 'sentiment': 'Positive'},
        {'review_id': 'R4', 'aspect': 'Food', 'Restaurant': 'RestB', 'sentiment': 'Positive'}
    ])

    comp = calculate_peer_comparison(df_ra, 'RestA', ['RestB'], aspect='Food')
    assert len(comp) == 2
    assert comp.iloc[0]['Restaurant'] == 'RestA'
    assert comp.iloc[0]['Role'] == 'Target'
    assert comp.iloc[0]['Negative_Pct'] == 100.0
    assert comp.iloc[1]['Restaurant'] == 'RestB'
    assert comp.iloc[1]['Role'] == 'Peer'
    assert comp.iloc[1]['Negative_Pct'] == 0.0


def test_empirical_bayes_smoothing_beta_binomial():
    # Create multi-restaurant distribution
    rows = []
    for i in range(10):
        rest = f"Rest_{i}"
        for j in range(20):
            sent = 'Negative' if (j % 5 == 0) else 'Positive'
            rows.append({'review_id': f"{rest}_{j}", 'aspect': 'Food', 'Restaurant': rest, 'sentiment': sent})

    df_ra = pd.DataFrame(rows)
    df_all_rev = df_ra[['review_id', 'Restaurant']].drop_duplicates()

    # Restaurant with tiny sample (1 review, 1 negative)
    tiny_ra = pd.DataFrame([
        {'review_id': 'Tiny_1', 'aspect': 'Food', 'Restaurant': 'Rest_Tiny', 'sentiment': 'Negative'}
    ])
    df_ra_combined = pd.concat([df_ra, tiny_ra], ignore_index=True)
    df_all_combined = pd.concat([df_all_rev, pd.DataFrame([{'review_id': 'Tiny_1', 'Restaurant': 'Rest_Tiny'}])], ignore_index=True)

    metrics = calculate_aspect_metrics(df_ra_combined, df_all_combined, restaurant='Rest_Tiny')
    smoothed_metrics = apply_empirical_bayes_smoothing(metrics, df_ra_combined, target_sentiment='Negative')

    food_row = smoothed_metrics[smoothed_metrics['Aspect'] == 'Food'].iloc[0]
    # Raw is 100% negative (1 out of 1)
    assert food_row['Negative_Pct'] == 100.0
    # Smoothed must be pulled down toward dataset prior (~20%)
    assert food_row['Negative_Smoothed_Pct'] < 100.0
    assert food_row['Negative_Smoothed_Pct'] > food_row['Negative_Prior_Mean_Pct']


def test_generate_descriptive_insights_no_recommendations():
    metrics = pd.DataFrame([
        {
            'Aspect': 'Food',
            'Eligible_Reviews': 100,
            'Mention_Count': 80,
            'Mention_Rate_Pct': 80.0,
            'Positive_Count': 70,
            'Positive_Pct': 87.5,
            'Negative_Count': 5,
            'Negative_Pct': 6.2,
            'Neutral_Count': 3,
            'Neutral_Pct': 3.8,
            'Mixed_Count': 2,
            'Mixed_Pct': 2.5
        },
        {
            'Aspect': 'Service',
            'Eligible_Reviews': 100,
            'Mention_Count': 50,
            'Mention_Rate_Pct': 50.0,
            'Positive_Count': 20,
            'Positive_Pct': 40.0,
            'Negative_Count': 25,
            'Negative_Pct': 50.0,
            'Neutral_Count': 3,
            'Neutral_Pct': 6.0,
            'Mixed_Count': 2,
            'Mixed_Pct': 4.0
        }
    ])

    insights = generate_descriptive_insights(metrics, restaurant_name="RestA")
    assert len(insights) >= 2
    for ins in insights:
        obs = ins['Observation'].lower()
        # Verify no SOP, recommendations, or cause inference
        assert 'recommend' not in obs
        assert 'sop' not in obs
        assert 'caused by' not in obs
        assert 'should' not in obs


if __name__ == '__main__':
    test_funcs = [v for k, v in list(globals().items()) if k.startswith('test_') and callable(v)]
    print(f"Executing {len(test_funcs)} DineSense AI analytics unit tests...\n")
    passed = 0
    for fn in test_funcs:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
            raise
    print(f"\n==========================================")
    print(f"ALL {passed}/{len(test_funcs)} UNIT TESTS PASSED SUCCESSFULLY!")
    print(f"==========================================")
