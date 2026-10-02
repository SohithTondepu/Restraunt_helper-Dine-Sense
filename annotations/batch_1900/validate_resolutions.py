import json
import pandas as pd
import numpy as np

# Load the 60 cases
with open(r'D:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews\annotations\batch_1900\needs_review_info.json', 'r', encoding='utf-8') as f:
    items = json.load(f)

# Define the exact resolutions for all 60 cases based on review context
# For each assertion_id:
# aspect, target_span, opinion_span, sentiment, theme, rationale
resolutions = {
    # Service (6 cases)
    'REV_00242_C09_A01': {
        'aspect': 'Service',
        'target_span': '',
        'opinion_span': 'disappointing',
        'sentiment': 'Negative',
        'theme': 'Staff hospitality',
        'rationale': 'Context refers to staff failing to coordinate cake arrangement for celebration'
    },
    'REV_01089_C04_A01': {
        'aspect': 'Service',
        'target_span': 'packaging',
        'opinion_span': 'disappointed',
        'sentiment': 'Negative',
        'theme': 'Delivery Speed',
        'rationale': 'Explicitly expresses negative sentiment about food delivery packaging'
    },
    'REV_01554_C02_A01': {
        'aspect': 'Service',
        'target_span': '',
        'opinion_span': 'Disappointed',
        'sentiment': 'Negative',
        'theme': 'Order accuracy',
        'rationale': 'Delivery order item mismatch and service failure'
    },
    'REV_01732_C03_A01': {
        'aspect': 'Service',
        'target_span': '',
        'opinion_span': 'Truly disappointed',
        'sentiment': 'Negative',
        'theme': 'Order accuracy',
        'rationale': 'Order was missing essential sides/accompaniments upon delivery'
    },
    'REV_02534_C02_A01': {
        'aspect': 'Service',
        'target_span': '',
        'opinion_span': 'Extremely disappointed',
        'sentiment': 'Negative',
        'theme': 'Staff hospitality',
        'rationale': 'Reviewer turned away at the door due to abrupt stag entry policy'
    },
    'REV_02773_C07_A01': {
        'aspect': 'Service',
        'target_span': '',
        'opinion_span': 'Very disappointing',
        'sentiment': 'Negative',
        'theme': 'Delivery Speed',
        'rationale': 'Delivery order was never delivered despite waiting hours'
    },

    # Ambience (1 case)
    'REV_01864_C07_A01': {
        'aspect': 'Ambience',
        'target_span': '',
        'opinion_span': 'can be better',
        'sentiment': 'Negative',
        'theme': 'Music / Noise level',
        'rationale': 'Refers directly to background music and atmosphere evaluated as needing improvement'
    },

    # Food (26 cases)
    'REV_00314_C04_A01': {
        'aspect': 'Food',
        'target_span': '',
        'opinion_span': 'Nothing special',
        'sentiment': 'Neutral',
        'theme': 'Taste / Flavor',
        'rationale': 'Refers to the ordered dish taste evaluated as mediocre / nothing special'
    },
    'REV_01104_C03_A01': {
        'aspect': 'Food',
        'target_span': '',
        'opinion_span': 'really disappointing',
        'sentiment': 'Negative',
        'theme': 'Menu variety',
        'rationale': 'Refers to missing menu items and lack of availability of food choices'
    },
    'REV_01212_C04_A01': {
        'aspect': 'Food',
        'target_span': '',
        'opinion_span': 'can be better',
        'sentiment': 'Negative',
        'theme': 'Food quality',
        'rationale': 'Refers to Exotic Vegetable Pasta which reviewer felt needed improvement'
    },
    'REV_02795_C03_A01': {
        'aspect': 'Food',
        'target_span': 'garnishing',
        'opinion_span': 'mixed feelings',
        'sentiment': 'Mixed',
        'theme': 'Taste / Flavor',
        'rationale': 'Refers to dish presentation and flavor combination of pomegranate garnishing'
    },
    'REV_03166_C08_A01': {
        'aspect': 'Food',
        'target_span': 'bakery section',
        'opinion_span': "wasn't disappointed",
        'sentiment': 'Positive',
        'theme': 'Food quality',
        'rationale': 'Positive appraisal of baked goods and dessert quality'
    },
    'REV_03231_C32_A01': {
        'aspect': 'Food',
        'target_span': 'Chilli Paneer',
        'opinion_span': 'disappointing',
        'sentiment': 'Negative',
        'theme': 'Taste / Flavor',
        'rationale': 'Explicit negative rating of Chilli Paneer taste'
    },
    'REV_03490_C02_A01': {
        'aspect': 'Food',
        'target_span': '',
        'opinion_span': 'disappointed',
        'sentiment': 'Negative',
        'theme': 'Portion size',
        'rationale': 'Portion sizes and food quantities reduced compared to earlier visits'
    },
    'REV_04237_C03_A01': {
        'aspect': 'Food',
        'target_span': '',
        'opinion_span': 'disappointed',
        'sentiment': 'Negative',
        'theme': 'Portion size',
        'rationale': 'Waffle toppings and portion fell far below expectations'
    },
    'REV_04568_C06_A01': {
        'aspect': 'Food',
        'target_span': 'ragi sankati',
        'opinion_span': 'disappointed',
        'sentiment': 'Negative',
        'theme': 'Taste / Flavor',
        'rationale': 'Specific dish ragi sankati evaluated as disappointing'
    },
    'REV_04623_C02_A01': {
        'aspect': 'Food',
        'target_span': 'it',
        'opinion_span': 'very disappointed',
        'sentiment': 'Negative',
        'theme': 'Food quality',
        'rationale': 'Food quality failure involving non-veg pieces found in veg chilli potato order'
    },
    'REV_06201_C02_A01': {
        'aspect': 'Food',
        'target_span': '',
        'opinion_span': 'Disappointed',
        'sentiment': 'Negative',
        'theme': 'Food quality',
        'rationale': 'Undercooked rice and noodle dish texture evaluated as disappointing'
    },
    'REV_06221_C03_A01': {
        'aspect': 'Food',
        'target_span': '',
        'opinion_span': 'got disappointed',
        'sentiment': 'Negative',
        'theme': 'Food quality',
        'rationale': 'Chicken bone found in prawns biryani dish'
    },
    'REV_06434_C06_A01': {
        'aspect': 'Food',
        'target_span': 'paneer tikka',
        'opinion_span': 'could be better',
        'sentiment': 'Negative',
        'theme': 'Food quality',
        'rationale': 'Specific dishes stuffed paneer tikka and paneer butter masala could be better'
    },
    'REV_07103_C03_A01': {
        'aspect': 'Food',
        'target_span': 'temperature',
        'opinion_span': 'Disappointed',
        'sentiment': 'Negative',
        'theme': 'Food temperature',
        'rationale': 'Served coffee was lukewarm rather than hot'
    },
    'REV_07134_C08_A01': {
        'aspect': 'Food',
        'target_span': 'Snacks',
        'opinion_span': 'could be better',
        'sentiment': 'Negative',
        'theme': 'Food quality',
        'rationale': 'Bar snacks evaluated as needing improvement'
    },
    'REV_07409_C03_A01': {
        'aspect': 'Food',
        'target_span': 'special omelette',
        'opinion_span': 'disappointed',
        'sentiment': 'Negative',
        'theme': 'Food quality',
        'rationale': 'Special omelette dish was disappointing compared to price and standard scramble'
    },
    'REV_07765_C04_A01': {
        'aspect': 'Food',
        'target_span': '',
        'opinion_span': 'Okayish',
        'sentiment': 'Neutral',
        'theme': 'Taste / Flavor',
        'rationale': 'Refers to food taste of the sampled starter evaluated as average'
    },
    'REV_08235_C12_A01': {
        'aspect': 'Food',
        'target_span': '',
        'opinion_span': 'Highly disappointing',
        'sentiment': 'Negative',
        'theme': 'Freshness',
        'rationale': 'Food smelled stale and day-old'
    },
    'REV_08330_C03_A01': {
        'aspect': 'Food',
        'target_span': 'other two items',
        'opinion_span': 'disappointed',
        'sentiment': 'Negative',
        'theme': 'Food quality',
        'rationale': 'Comparison of menu items where the two non-chocolate items failed expectations'
    },
    'REV_08330_C07_A01': {
        'aspect': 'Food',
        'target_span': 'mint Mojito',
        'opinion_span': 'less disappointing',
        'sentiment': 'Negative',
        'theme': 'Taste / Flavor',
        'rationale': 'Evaluation of beverage taste (mint mojito)'
    },
    'REV_08702_C17_A01': {
        'aspect': 'Food',
        'target_span': 'kebab',
        'opinion_span': 'super disappointing',
        'sentiment': 'Negative',
        'theme': 'Taste / Flavor',
        'rationale': 'Kebab serving was dry and tasteless'
    },
    'REV_08702_C27_A01': {
        'aspect': 'Food',
        'target_span': '',
        'opinion_span': 'Nothing special',
        'sentiment': 'Neutral',
        'theme': 'Taste / Flavor',
        'rationale': 'Main course curries evaluated as average / nothing special'
    },
    'REV_08748_C05_A01': {
        'aspect': 'Food',
        'target_span': '',
        'opinion_span': 'not up to the mark',
        'sentiment': 'Negative',
        'theme': 'Menu variety',
        'rationale': 'Buffet dessert spread and taste was not up to the mark'
    },
    'REV_08840_C10_A01': {
        'aspect': 'Food',
        'target_span': 'haleem',
        'opinion_span': 'disappointed',
        'sentiment': 'Negative',
        'theme': 'Taste / Flavor',
        'rationale': 'Haleem taste and preparation disappointed the reviewer'
    },
    'REV_08942_C03_A01': {
        'aspect': 'Food',
        'target_span': 'brunch',
        'opinion_span': 'Truly disappointed',
        'sentiment': 'Negative',
        'theme': 'Food quality',
        'rationale': 'Sunday brunch buffet quality fell below standard'
    },
    'REV_08984_C06_A01': {
        'aspect': 'Food',
        'target_span': '',
        'opinion_span': 'disappointing',
        'sentiment': 'Negative',
        'theme': 'Food quality',
        'rationale': 'Ordered specialty dish did not meet expectations'
    },
    'REV_09538_C05_A01': {
        'aspect': 'Food',
        'target_span': '',
        'opinion_span': 'could be better',
        'sentiment': 'Neutral',
        'theme': 'Menu variety',
        'rationale': 'Bread and accompaniment options could be better'
    },

    # General Experience (27 cases)
    'REV_01142_C05_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'Never disappointed us',
        'sentiment': 'Positive',
        'theme': 'Overall experience',
        'rationale': 'Holistic endorsement of establishment across repeated visits'
    },
    'REV_01396_C02_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'was disappointed',
        'sentiment': 'Negative',
        'theme': 'Overall experience',
        'rationale': 'Overall impression of the lunch visit to the restaurant'
    },
    'REV_01935_C05_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'Okayish',
        'sentiment': 'Neutral',
        'theme': 'Overall experience',
        'rationale': 'Overall concluding summary of the visit'
    },
    'REV_01997_C03_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'HIGHLY DISAPPOINTED',
        'sentiment': 'Negative',
        'theme': 'Overall experience',
        'rationale': 'Overall visit verdict expressing high disappointment'
    },
    'REV_02101_C03_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'Extremely disappointed',
        'sentiment': 'Negative',
        'theme': 'Overall experience',
        'rationale': 'Holistic dining experience summary'
    },
    'REV_02172_C01_A01': {
        'aspect': 'General Experience',
        'target_span': "Jonathan's Kitchen",
        'opinion_span': 'very disappointed',
        'sentiment': 'Negative',
        'theme': 'Overall experience',
        'rationale': 'Overall impression of visiting the establishment'
    },
    'REV_02413_C05_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'will not prefer to return again',
        'sentiment': 'Negative',
        'theme': 'Return intent',
        'rationale': 'Explicit negative return intent statement for the entire restaurant'
    },
    'REV_02978_C11_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'was disappointed',
        'sentiment': 'Negative',
        'theme': 'Overall experience',
        'rationale': 'Collective holistic assessment of team corporate outing'
    },
    'REV_03133_C07_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'expectations were disappointed',
        'sentiment': 'Negative',
        'theme': 'Overall experience',
        'rationale': 'Overall diner expectations unmet for the restaurant'
    },
    'REV_03190_C03_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'really disappointed',
        'sentiment': 'Negative',
        'theme': 'Overall experience',
        'rationale': 'Entire party disappointed with overall visit'
    },
    'REV_03212_C21_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'Highly disappointed',
        'sentiment': 'Negative',
        'theme': 'Overall experience',
        'rationale': 'Overall takeaway and closing verdict on the restaurant visit'
    },
    'REV_03648_C01_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'never was disappointed',
        'sentiment': 'Positive',
        'theme': 'Overall experience',
        'rationale': 'Track record of satisfaction across past visits to the restaurant'
    },
    'REV_03960_C02_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'disappointed',
        'sentiment': 'Negative',
        'theme': 'Overall experience',
        'rationale': 'General summary verdict on the visit'
    },
    'REV_05161_C01_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'Disappointed',
        'sentiment': 'Negative',
        'theme': 'Overall experience',
        'rationale': 'Review title / opening standalone summary verdict'
    },
    'REV_06289_C06_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'very disappointing',
        'sentiment': 'Negative',
        'theme': 'Overall experience',
        'rationale': 'Holistic group event evaluation'
    },
    'REV_06383_C08_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'Totally disappointed',
        'sentiment': 'Negative',
        'theme': 'Overall experience',
        'rationale': 'Closing overall evaluation of the restaurant visit'
    },
    'REV_06783_C02_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'total disappointed',
        'sentiment': 'Negative',
        'theme': 'Overall experience',
        'rationale': 'Overall impression of the previous night dinner experience'
    },
    'REV_06783_C09_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'not be disappointed',
        'sentiment': 'Positive',
        'theme': 'Return intent',
        'rationale': 'Positive expectation and intent to dine again with friends'
    },
    'REV_06848_C07_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': "it's ok",
        'sentiment': 'Neutral',
        'theme': 'Overall experience',
        'rationale': 'Conditional holistic verdict summarizing the dining experience'
    },
    'REV_06897_C01_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'Disappointed',
        'sentiment': 'Negative',
        'theme': 'Overall experience',
        'rationale': 'Hashtag summary of overall visit experience'
    },
    'REV_07765_C07_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'can be better',
        'sentiment': 'Neutral',
        'theme': 'Overall experience',
        'rationale': 'Overall restaurant evaluation suggesting potential for improvement'
    },
    'REV_08148_C01_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'Very very very very disappointed',
        'sentiment': 'Negative',
        'theme': 'Overall experience',
        'rationale': 'Overall review headline / emotional verdict'
    },
    'REV_08217_C02_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'Disappointed',
        'sentiment': 'Negative',
        'theme': 'Overall experience',
        'rationale': 'Overall impression of visit'
    },
    'REV_08330_C01_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'disappointed',
        'sentiment': 'Negative',
        'theme': 'Overall experience',
        'rationale': 'Opening summary sentence evaluating the entire dinner visit'
    },
    'REV_09336_C01_A01': {
        'aspect': 'General Experience',
        'target_span': 'Zega',
        'opinion_span': 'not disappointed',
        'sentiment': 'Positive',
        'theme': 'Overall experience',
        'rationale': 'Satisfying overall visit to the restaurant'
    },
    'REV_09463_C10_A01': {
        'aspect': 'General Experience',
        'target_span': '',
        'opinion_span': 'not be disappointed',
        'sentiment': 'Positive',
        'theme': 'Overall experience',
        'rationale': 'General recommendation assuring future diners will not be disappointed'
    }
}

print(f"Resolutions defined: {len(resolutions)}")

# Validation test against clause_text and review_text
errors = []
for it in items:
    aid = it['assertion_id']
    if aid not in resolutions:
        errors.append(f"Missing resolution for {aid}")
        continue
    
    res = resolutions[aid]
    clause_text = it['clause_text']
    review_text = it['review_text']
    c_start = it['clause_start_char']
    c_end = it['clause_end_char']

    # 1. Assert clause offset
    if review_text[c_start:c_end] != clause_text:
        errors.append(f"Clause mismatch for {aid}: {repr(review_text[c_start:c_end])} vs {repr(clause_text)}")

    # 2. Check opinion_span
    op = res['opinion_span']
    if op not in clause_text:
        errors.append(f"Opinion span {repr(op)} not in clause {repr(clause_text)} for {aid}")
    else:
        op_start = review_text.find(op, c_start)
        op_end = op_start + len(op)
        if review_text[op_start:op_end] != op:
            errors.append(f"Opinion offset mismatch for {aid}")

    # 3. Check target_span
    tg = res['target_span']
    if tg != '':
        if tg not in clause_text:
            errors.append(f"Target span {repr(tg)} not in clause {repr(clause_text)} for {aid}")
        else:
            tg_start = review_text.find(tg, c_start)
            tg_end = tg_start + len(tg)
            if review_text[tg_start:tg_end] != tg:
                errors.append(f"Target offset mismatch for {aid}")

if errors:
    print(f"Found {len(errors)} validation errors:")
    for e in errors:
        print("  -", e)
else:
    print("ALL 60 RESOLUTIONS VALIDATED 100% PERFECTLY AGAINST REVIEW TEXT!")
