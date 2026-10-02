import unittest
import pandas as pd
import numpy as np
from src.data_loader import rating_to_sentiment, sentiment_to_label
from src.aspect_engine import (
    split_into_clauses,
    extract_aspects_from_review,
    predict_clause_sentiment,
    match_aspect_hybrid
)
from src.health_index import compute_aspect_health, compute_restaurant_scorecard
from src.report_llm import (
    build_restaurant_payload,
    verify_report_numbers,
    generate_grounded_report,
    retrieve_operational_sop
)
from src.root_cause import extract_opinion_pairs


class TestPipeline(unittest.TestCase):

    def test_sentiment_mapping(self):
        self.assertEqual(rating_to_sentiment(1.0), 'Negative')
        self.assertEqual(rating_to_sentiment(2.0), 'Negative')
        self.assertEqual(rating_to_sentiment(2.5), 'Neutral')
        self.assertEqual(rating_to_sentiment(3.0), 'Neutral')
        self.assertEqual(rating_to_sentiment(3.5), 'Neutral')
        self.assertEqual(rating_to_sentiment(4.0), 'Positive')
        self.assertEqual(rating_to_sentiment(5.0), 'Positive')
        self.assertEqual(sentiment_to_label('Negative'), 0)
        self.assertEqual(sentiment_to_label('Neutral'), 1)
        self.assertEqual(sentiment_to_label('Positive'), 2)

    def test_clause_splitting(self):
        text = "The food was delicious, but the service was terrible. The ambience was great though."
        clauses = split_into_clauses(text)
        self.assertGreaterEqual(len(clauses), 2)
        self.assertTrue(any("food" in c.lower() for c in clauses))
        self.assertTrue(any("service" in c.lower() for c in clauses))

    def test_rst_concessive_clause_splitting(self):
        # RST concessive discourse marker separation
        text = "Despite the cold soup, we thoroughly enjoyed the live jazz music."
        clauses = split_into_clauses(text)
        self.assertGreaterEqual(len(clauses), 2)
        self.assertTrue(any("soup" in c.lower() or "cold" in c.lower() for c in clauses))
        self.assertTrue(any("jazz" in c.lower() or "music" in c.lower() for c in clauses))
        # Ensure leading concessive marker is stripped cleanly
        for c in clauses:
            self.assertFalse(c.lower().startswith("despite "))

    def test_aspect_extraction(self):
        review = "Great mutton biryani and crispy naan, but the waiter took 40 minutes to bring the bill."
        findings = extract_aspects_from_review(review)
        aspects = [f['aspect'] for f in findings]
        self.assertIn('Food', aspects)
        self.assertIn('Service', aspects)

    def test_semantic_embedding_aspect_matching(self):
        # Implicit metaphor test: "cost an arm and a leg" -> Price / Value
        aspect_price = match_aspect_hybrid("It cost an arm and a leg")
        self.assertIn('Price / Value', aspect_price)

        # Direct service mention -> Service
        aspect_service = match_aspect_hybrid("The waiter was attentive and polite")
        self.assertIn('Service', aspect_service)

    def test_five_category_explicit_keywords(self):
        # 1. Food
        self.assertEqual(match_aspect_hybrid("The biryani was excellent"), {'Food'})
        # 2. Service
        self.assertEqual(match_aspect_hybrid("The staff were friendly"), {'Service'})
        # 3. Price / Value
        self.assertEqual(match_aspect_hybrid("The pricing was very expensive"), {'Price / Value'})
        # 4. Ambience
        self.assertEqual(match_aspect_hybrid("The decor and lighting were beautiful"), {'Ambience'})
        # 5. General Experience
        self.assertEqual(match_aspect_hybrid("Must visit!"), {'General Experience'})
        self.assertEqual(match_aspect_hybrid("I would definitely come back"), {'General Experience'})

    def test_general_experience_statements(self):
        self.assertIn('General Experience', match_aspect_hybrid("Must visit!"))
        self.assertIn('General Experience', match_aspect_hybrid("I would definitely come back"))
        self.assertIn('General Experience', match_aspect_hybrid("Overall, a wonderful dining experience"))
        self.assertIn('General Experience', match_aspect_hybrid("I highly recommend this restaurant"))
        self.assertIn('General Experience', match_aspect_hybrid("Worst experience of my life"))
        self.assertIn('General Experience', match_aspect_hybrid("We loved this place"))

    def test_no_match_clauses(self):
        # Factual, procedural, and contextual clauses must preserve no-match behaviour
        self.assertEqual(match_aspect_hybrid("Visited yesterday at 8 PM"), set())
        self.assertEqual(match_aspect_hybrid("We arrived at 9 PM yesterday"), set())
        self.assertEqual(match_aspect_hybrid("6 of us went there to dine in"), set())
        self.assertEqual(match_aspect_hybrid("Do follow us on Instagram"), set())
        # Generic sentiment words alone must NOT be classified as General Experience
        self.assertEqual(match_aspect_hybrid("It was good"), set())
        self.assertEqual(match_aspect_hybrid("Just fine"), set())
        self.assertEqual(match_aspect_hybrid("Terrible"), set())
        self.assertEqual(match_aspect_hybrid("Nice and fine"), set())

    def test_ambiguous_term_slow(self):
        # Service latency: slow service / staff
        self.assertEqual(match_aspect_hybrid("The service was slow"), {'Service'})
        self.assertEqual(match_aspect_hybrid("The staff was very slow"), {'Service'})
        self.assertEqual(match_aspect_hybrid("It was so slow"), {'Service'})
        # Ambiguous slow: music/ambience should NOT trigger Service
        self.assertEqual(match_aspect_hybrid("The music was slow and peaceful"), {'Ambience'})
        # Ambiguous slow: food cooking should NOT trigger Service
        self.assertEqual(match_aspect_hybrid("We ordered slow cooked mutton"), {'Food'})
        # Multi-aspect with slow cooked food and slow service
        res = match_aspect_hybrid("Slow cooked pork, but the waiter was slow")
        self.assertIn('Food', res)
        self.assertIn('Service', res)

    def test_multi_aspect_clauses(self):
        # Co-occurring aspects within a single clause
        res_food_amb = match_aspect_hybrid("Great food and lovely ambience")
        self.assertEqual(res_food_amb, {'Food', 'Ambience'})

        # Review with multiple aspects across clauses
        review = "The food was delicious, but the place is a must visit."
        findings = extract_aspects_from_review(review)
        detected_aspects = {f['aspect'] for f in findings}
        self.assertIn('Food', detected_aspects)
        self.assertIn('General Experience', detected_aspects)

    def test_minilm_semantic_fallback_five_categories(self):
        # Tier 2 MiniLM fallback for implicit concepts
        self.assertEqual(match_aspect_hybrid("The place was deafening"), {'Ambience'})
        self.assertEqual(match_aspect_hybrid("It cost an arm and a leg"), {'Price / Value'})
        self.assertEqual(match_aspect_hybrid("A truly memorable evening at this restaurant"), {'General Experience'})
        self.assertEqual(match_aspect_hybrid("We had an absolute blast dining here"), {'General Experience'})
        # Verify MiniLM does NOT indiscriminately assign General Experience
        self.assertEqual(match_aspect_hybrid("It was okay"), set())
        self.assertEqual(match_aspect_hybrid("6 of us went there to dine in"), set())

    def test_health_scoring_edge_cases(self):
        # Empty count edge case shrinks to prior (global prior 0.35 -> ~67.5)
        score_empty = compute_aspect_health(0, 0, 0, global_prior=0.35, k=5.0)
        self.assertTrue(60 <= score_empty <= 75)
        
        # 100% positive reviews
        score_all_pos = compute_aspect_health(100, 0, 0, global_prior=0.35, k=5.0)
        self.assertGreaterEqual(score_all_pos, 90.0)
        
        # 100% negative reviews
        score_all_neg = compute_aspect_health(0, 100, 0, global_prior=0.35, k=5.0)
        self.assertLessEqual(score_all_neg, 10.0)

    def test_time_decay_health_scoring(self):
        # Create a mock dataframe of reviews across different times
        sample_df = pd.DataFrame({
            'Restaurant': ['Test Cafe'] * 4,
            'Review': [
                'Terrible cold food and slow staff',
                'Worst food ever tasted',
                'Amazing delicious food and great service',
                'Outstanding experience, loved the food'
            ],
            'Rating': [1.0, 1.0, 5.0, 5.0],
            'Time': [
                '2022-01-01 10:00:00', # Old negative reviews
                '2022-01-02 10:00:00',
                '2023-01-01 10:00:00', # Recent positive reviews
                '2023-01-02 10:00:00'
            ]
        })

        # Scorecard without time decay
        scorecard_static = compute_restaurant_scorecard('Test Cafe', sample_df)
        self.assertFalse(scorecard_static['time_decay_applied'])

        # Scorecard with exponential time decay (half-life 90 days)
        scorecard_decay = compute_restaurant_scorecard('Test Cafe', sample_df, half_life_days=90.0)
        self.assertTrue(scorecard_decay['time_decay_applied'])
        
        # Because recent reviews are 5.0 positive, time decay should weight them much higher
        # and result in a significantly higher health index than static equal weighting
        self.assertGreater(
            scorecard_decay['aspects']['Food']['score'],
            scorecard_static['aspects']['Food']['score']
        )

    def test_opinion_extraction(self):
        text = "The waiter was extremely rude and the food was cold."
        pairs = extract_opinion_pairs(text)
        self.assertGreaterEqual(len(pairs), 1)
        self.assertTrue(any("waiter" in p or "rude" in p or "cold" in p for p in pairs))

    def test_report_fact_verification(self):
        payload = {
            'restaurant': 'Test Grill',
            'overall_health_index': 82.5,
            'average_stars': 4.2,
            'total_reviews': 100,
            'aspect_performance': {
                'Food': {'score': 88.0, 'status': 'Excellent', 'total_mentions': 45, 'negative_mentions': 2, 'positive_mentions': 40}
            },
            'top_complaint_clusters': []
        }
        
        # Good report citing true numbers
        good_report = "Test Grill operates with an Overall Health Index of 82.5 based on 100 reviews. Food score is 88.0."
        valid, unmatched = verify_report_numbers(good_report, payload)
        self.assertTrue(valid)
        self.assertEqual(len(unmatched), 0)
        
        # Hallucinated report
        hallucinated_report = "Test Grill had 999 complaints and an index of 12.34."
        valid_bad, unmatched_bad = verify_report_numbers(hallucinated_report, payload)
        self.assertFalse(valid_bad)
        self.assertTrue('999' in unmatched_bad or '12.34' in unmatched_bad)

    def test_hybrid_rag_report_generation(self):
        # Test SOP retrieval
        food_sop = retrieve_operational_sop('Food', 'cold soup and lukewarm curry')
        self.assertIn('Temperature', food_sop['sub_category'])
        self.assertIn('heat lamps', food_sop['action_item'].lower())

        price_sop = retrieve_operational_sop('Price', 'overpriced and too expensive')
        self.assertIn('decoy pricing', price_sop['action_item'].lower())

        # Test full grounded report synthesis
        scorecard = {
            'restaurant': 'Bistro 42',
            'overall_health_index': 62.0,
            'average_stars': 3.5,
            'total_reviews': 80,
            'aspects': {
                'Food': {'score': 54.0, 'status': 'Critical Alert', 'total_mentions': 40, 'neg_mentions': 15, 'pos_mentions': 20},
                'Service': {'score': 72.0, 'status': 'Satisfactory', 'total_mentions': 30, 'neg_mentions': 5, 'pos_mentions': 22}
            }
        }
        complaints = [{'root_cause': 'cold food delivered', 'frequency': 12, 'evidence_quotes': ['Food was freezing cold']}]
        payload = build_restaurant_payload(scorecard, complaints)
        
        report_result = generate_grounded_report(payload)
        report_md = report_result['report_markdown']
        
        self.assertIn('Executive Strategic Action Report', report_md)
        self.assertIn('Bistro 42', report_md)
        self.assertIn('Prescriptive 30-Day Operational Action Roadmap', report_md)
        self.assertTrue(report_result['verification_passed'])
        self.assertEqual(len(report_result['unmatched_numbers']), 0)


if __name__ == '__main__':
    unittest.main()
