import pandas as pd

manifest = pd.read_csv(r"annotations\pilot_sample_manifest.csv")

# Candidate 25 review IDs covering all requirements:
candidate_ids = [
    'REV_00073', # Beyond Flavours: attendant, chill with friends, multiple aspects
    'REV_00129', # Paradise: biryani good, service good, multi-clause
    'REV_00313', # Shah Ghouse: short negative "biryani was oily"
    'REV_00457', # Over The Moon: "Food and ambience is nice", drinks
    'REV_00564', # The Fisherman's Wharf: "A bit expensive but food too good" (implicit target pricing + contrast)
    'REV_00795', # Shah Ghouse Shawarma: short praise "excellent service"
    'REV_00822', # Hyper Local: long review, IPL screening, 2+2 offers, crowd
    'REV_01074', # Sardarji's Chaats: "Chat is also good not that great though" (contrast/negation)
    'REV_01284', # Absolute Sizzlers: "nothing close to sizzler", "only looks" (strong negation/contrast)
    'REV_01427', # AB's: "crispy corns", service, multiple aspects
    'REV_01651', # NorFest: long review, elaboration on dishes and pricing
    'REV_01744', # Hotel Zara Hi-Fi: short praise "too tasty it is" (implicit target)
    'REV_01800', # 10 Downing Street: USER TARGET "not good very bad", salad dressing, mixed service/ambience
    'REV_02529', # Kobe Sizzlers: negation, contrast
    'REV_02912', # Hunger Maggi Point: USER TARGET "one fine night", "cost effective", delivery, soggy
    'REV_03108', # Paradise: "cold food", wait time
    'REV_04412', # Drunken Monkey: "crispy", drinks, descriptive
    'REV_04777', # Eat Burp: "oily", "very bad"
    'REV_05070', # Platform 65: USER TARGET "crispy", "fine-dine", "not that great", long review
    'REV_05863', # Kritunga: "not good very bad", staff attitude
    'REV_06090', # Pista House: "delicious", "oily", mixed
    'REV_06534', # Yum Yum Tree: USER TARGET "MUST TRY PLACE", Mandi, ratings
    'REV_08253', # The Roastery: "cold", delay, contrast
    'REV_08313', # Cream Stone: "crispy waffle", desserts
    'REV_08836', # Cascade - Radisson: USER TARGET "The place is amazing", "must try in that area"
]

print("Total candidates:", len(candidate_ids))
sample_df = manifest[manifest['review_id'].isin(candidate_ids)].copy()
print("Found in manifest:", len(sample_df))
print("\nRating Distribution in 25-Review Test Batch:")
print(sample_df['rating_bin'].value_counts())
print("\nLength Distribution in 25-Review Test Batch:")
print(sample_df['length_bin'].value_counts())
print("\nEstablishments Count:", sample_df['establishment_id'].nunique())
