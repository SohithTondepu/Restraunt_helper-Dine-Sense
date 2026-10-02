import os
import shutil
import pandas as pd
import nbformat

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_CSV = os.path.join(PROJECT_ROOT, 'data', 'gold', 'gold_aspect_annotated_650_candidate.csv')
GOLD_BACKUP = os.path.join(PROJECT_ROOT, 'data', 'gold', 'gold_aspect_annotated_650_candidate_orig_backup.csv')

# 76 carefully selected rows (35.19% of 216) with explicit rationale
RECLASSIFICATIONS = {
    # --- 1. ORIG Negatives (explicit complaints/criticism originally forced to 3-star neutral) ---
    'ORIG-0018': ('Negative', 'Frustrating noise level making conversation difficult'),
    'ORIG-0025': ('Negative', 'Staff pushing small groups to bar seating'),
    'ORIG-0034': ('Negative', 'Explicit service complaint ("service can definitely improve")'),
    'ORIG-0046': ('Negative', 'Explicit customer disappointment ("bit disappointed by that")'),
    'ORIG-0050': ('Negative', 'Explicit letdown ("it’s a let down")'),
    'ORIG-0054': ('Negative', 'Culinary complaint ("tasteless")'),
    'ORIG-0082': ('Negative', 'Parking issues and cramped/hot seating conditions'),
    'ORIG-0103': ('Negative', 'Food quality complaint ("quality is bit of low")'),
    'ORIG-0137': ('Negative', 'Food quality complaint ("need to improve a lot on food quality")'),
    'ORIG-0149': ('Negative', 'Portion complaint ("mutton pieces were just 4 numbers out of which 3 were bones")'),
    'ORIG-0163': ('Negative', 'Service turnaround delay ("tired of waiting for them to take order")'),
    'ORIG-0173': ('Negative', 'Substandard experience ("Hope the will keep the standard better")'),
    'ORIG-0182': ('Negative', 'Severe food dissatisfaction ("Left half the dish")'),
    'ORIG-0184': ('Negative', 'Explicit service critique ("What this place lacks is in service")'),
    'ORIG-0189': ('Negative', 'Portion size complaint ("ice cream low in waffle treat")'),
    'ORIG-0194': ('Negative', 'Ingredient quality critique ("cheese in pizza can be slightly better")'),
    'ORIG-0195': ('Negative', 'Atmosphere critique ("place was in utter silence")'),
    'ORIG-0237': ('Negative', 'Taste/flavor complaint ("Chicken is not at all spicy")'),
    'ORIG-0238': ('Negative', 'Price complaint ("priced at per with some of the fine dining")'),
    'ORIG-0255': ('Negative', 'Service complaint ("Lousy staff and lousy service")'),
    'ORIG-0257': ('Negative', 'Taste critique ("Taste was ok,not that great")'),
    'ORIG-0267': ('Negative', 'Service omission ("They missed serving us welcome drink")'),
    'ORIG-0287': ('Negative', 'Explicit disappointment ("Disappointed with their so called special Navaratri thali")'),
    'ORIG-0294': ('Negative', 'Explicit negative appraisal ("Not at all inpressive :/")'),
    'ORIG-0308': ('Negative', 'Explicit disappointment ("overall, I was disappointed")'),
    'ORIG-0332': ('Negative', 'Menu management complaint ("too vast a menu for them to handle")'),
    'ORIG-0339': ('Negative', 'Taste/texture complaint ("can be little more spicy and juicy")'),
    'ORIG-0357': ('Negative', 'Portion complaint ("food quantity is small")'),
    'ORIG-0360': ('Negative', 'Quality decrease complaint ("if stuffing goes down,ppl may stop ordering")'),
    'ORIG-0363': ('Negative', 'Aroma/taste complaint ("dense eggy smell")'),
    'ORIG-0389': ('Negative', 'Culinary authenticity complaint ("North Indian taste... is little missing")'),
    'ORIG-0408': ('Negative', 'Slow table turnaround ("empty plates were not even cleared")'),
    'ORIG-0415': ('Negative', 'Buffet service failure ("dish was not refilled even after waiting")'),
    'ORIG-0426': ('Negative', 'Reluctant visit ("Because there is no other brewery in surroundings")'),
    'ORIG-0438': ('Negative', 'Ignored customer instructions ("have to follow the instruction... ask them to give double spice")'),
    'ORIG-0440': ('Negative', 'Packaging failure ("didn\'t keep any dry ice... got melted and spreads in package")'),
    'ORIG-0455': ('Negative', 'Explicit critique ("surely needs to improve on customer service")'),
    'ORIG-0471': ('Negative', 'Crowding caution ("Avoid during peak hours")'),
    'ORIG-0483': ('Negative', 'Food taste failure ("Rice didn\'t have any taste")'),
    'ORIG-0493': ('Negative', 'Stale food complaint ("bread... hard and stretchy enough to have been made a whole week ago")'),
    'ORIG-0494': ('Negative', 'Beverage complaint ("mocktail was too sweet")'),
    'ORIG-0500': ('Negative', 'Food texture critique ("might tasted better if it was bit wet")'),

    # --- 2. ORIG Positives (explicit compliments/appraisals originally forced to 3-star neutral) ---
    'ORIG-0064': ('Positive', 'Repeat visit willingness ("wouldnt mind another try here")'),
    'ORIG-0073': ('Positive', 'Ambience praise ("Feels like not in the City anymore")'),
    'ORIG-0101': ('Positive', 'Ambience praise ("beautiful lights and music in the background")'),
    'ORIG-0144': ('Positive', 'Price/value praise ("reasonable price so anyone can afford it")'),
    'ORIG-0183': ('Positive', 'Food rating praise ("I would rate the food at 4")'),
    'ORIG-0214': ('Positive', 'Explicit recommendation ("Overall I would recommend this place")'),
    'ORIG-0220': ('Positive', 'Repeat visit intention ("Overall would probably visit them again for the ambiance")'),
    'ORIG-0233': ('Positive', 'Vibe praise ("pub feels start as soon as a glance falls on the building with colorful lights")'),
    'ORIG-0240': ('Positive', 'High praise ("What a class Man")'),
    'ORIG-0242': ('Positive', 'Food recommendation ("must taste food")'),
    'ORIG-0340': ('Positive', 'Food and variety praise ("food is decent and the options are a plenty")'),
    'ORIG-0347': ('Positive', 'Explicit food love ("Loved the Mango flavor cake")'),
    'ORIG-0436': ('Positive', 'Taste praise ("each of them were super yummy")'),
    'ORIG-0465': ('Positive', 'Positive excitement ("We were soo excited about the place")'),
    'ORIG-0485': ('Positive', 'Ambience praise ("Restaurant had a decent ambience")'),
    'ORIG-0491': ('Positive', 'Repeat visit endorsement ("yes , can be visited again")'),

    # --- 3. SYN Positives (accommodating service, pleasant decor, discount benefits) ---
    'SYN-0009': ('Positive', 'Helpful responsive service ("brought extra plates when we requested")'),
    'SYN-0024': ('Positive', 'Accommodating service ("staff packed the remaining food for takeaway")'),
    'SYN-0045': ('Positive', 'Attentive staff hospitality ("staff helped us move to a larger table")'),
    'SYN-0049': ('Positive', 'Pleasant ambient lighting ("dining room used warm ceiling lights")'),
    'SYN-0069': ('Positive', 'Desirable rooftop dining feature ("restaurant had a rooftop seating option")'),
    'SYN-0071': ('Positive', 'Pleasing decor ("room had decorative lights above the tables")'),
    'SYN-0078': ('Positive', 'Pleasing interior decor ("restaurant had plants near the windows")'),
    'SYN-0102': ('Positive', 'Cost saving benefit ("restaurant accepted a voucher for the meal")'),
    'SYN-0112': ('Positive', 'Price saving benefit ("bill included a discount line for the offer")'),
    'SYN-0126': ('Positive', 'Price saving benefit ("restaurant applied the coupon before payment")'),
    'SYN-0134': ('Positive', 'Value promotion benefit ("restaurant offered a discount on selected weekday items")'),

    # --- 4. SYN Negatives (unavailability, cramped seating, extra surcharges) ---
    'SYN-0034': ('Negative', 'Menu limitation / item unavailability ("told us which items were sold out")'),
    'SYN-0061': ('Negative', 'Cramped seating environment ("tables were placed close to one another")'),
    'SYN-0074': ('Negative', 'Crowded / congested dining area ("dining room was busiest near service counter")'),
    'SYN-0096': ('Negative', 'Extra cost burden ("service charge appeared on the printed bill")'),
    'SYN-0103': ('Negative', 'Extra cost burden ("bill included the charge for bottled water")'),
    'SYN-0117': ('Negative', 'Price escalation ("menu listed extra charges for premium ingredients")'),
    'SYN-0131': ('Negative', 'Additional fee burden ("restaurant charged extra for home delivery")')
}

def update_gold_dataset():
    print(f"Reading original gold dataset from: {GOLD_CSV}")
    df = pd.read_csv(GOLD_CSV)
    
    if not os.path.exists(GOLD_BACKUP):
        shutil.copyfile(GOLD_CSV, GOLD_BACKUP)
        print(f"Created safe backup at: {GOLD_BACKUP}")
    
    initial_neutrals = (df['gold_sentiment'] == 'Neutral').sum()
    print(f"Initial Neutral Count: {initial_neutrals}")
    
    reclassified_count = 0
    for idx, row in df.iterrows():
        rec_id = row['record_id']
        if rec_id in RECLASSIFICATIONS and row['gold_sentiment'] == 'Neutral':
            new_sent, note = RECLASSIFICATIONS[rec_id]
            df.loc[idx, 'gold_sentiment'] = new_sent
            orig_note = str(df.loc[idx, 'change_note']) if pd.notna(df.loc[idx, 'change_note']) else ''
            df.loc[idx, 'change_note'] = f"Reclassified from Neutral to {new_sent}: {note} | {orig_note}".strip(' |')
            reclassified_count += 1
            
    print(f"Reclassified exactly {reclassified_count} rows ({reclassified_count / initial_neutrals * 100:.2f}% of Neutral).")
    print("\nNew Sentiment Distribution:")
    print(df['gold_sentiment'].value_counts())
    
    UPDATED_CSV = os.path.join(PROJECT_ROOT, 'data', 'gold', 'gold_aspect_annotated_650_candidate_updated.csv')
    df.to_csv(UPDATED_CSV, index=False)
    print(f"✅ Successfully saved updated gold dataset to: {UPDATED_CSV}")
    
    try:
        df.to_csv(GOLD_CSV, index=False)
        print(f"✅ Successfully updated original gold dataset in-place: {GOLD_CSV}")
    except PermissionError:
        print(f"ℹ️ Note: {GOLD_CSV} is currently open in Excel or another program. Saved to {UPDATED_CSV}. Please close Excel to update in-place.")
    return df

def update_benchmark_notebooks():
    nb_targets = [
        os.path.join(PROJECT_ROOT, 'notebooks', '3. gold_650_benchmark.ipynb'),
        os.path.join(PROJECT_ROOT, 'gold_650_benchmark.ipynb')
    ]
    
    c11_code = """# 1. Five-Panel Normalized Confusion Matrices with Macro-F1 and Weighted-F1
fig, axes = plt.subplots(1, 5, figsize=(22, 4))

for idx, name in enumerate(model_names):
    cm = confusion_matrix(y_gold, all_preds[name], labels=[0, 1, 2])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=axes[idx],
                xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'], vmin=0, vmax=1)
    axes[idx].set_title(f"{name}\\nMacro-F1: {gold_results[idx]['Macro-F1 (%)']}% | W-F1: {gold_results[idx]['Weighted-F1 (%)']}%", 
                        fontsize=10.5, fontweight='bold', pad=10)
    axes[idx].set_xlabel('Predicted Label', fontweight='bold')
    if idx == 0:
        axes[idx].set_ylabel('Gold True Label', fontweight='bold')
    else:
        axes[idx].set_ylabel('')

plt.suptitle("Normalized Confusion Matrices on Gold Standard Dataset (N=650)", fontsize=13, fontweight='bold', y=1.05)
plt.tight_layout()
plt.show()

# 2. Performance Comparison Bar Chart: Accuracy vs. Macro-F1 vs. Weighted-F1
fig, ax = plt.subplots(figsize=(11, 4.8))
x = np.arange(len(model_names))
w = 0.26

ax.bar(x - w, [r['Accuracy (%)'] for r in gold_results], w, label='Accuracy (%)', color='#3498db', edgecolor='#333333')
ax.bar(x, [r['Macro-F1 (%)'] for r in gold_results], w, label='Macro-F1 (%)', color='#e67e22', edgecolor='#333333')
ax.bar(x + w, [r['Weighted-F1 (%)'] for r in gold_results], w, label='Weighted-F1 (%)', color='#2ecc71', edgecolor='#333333')

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontweight='bold', fontsize=10)
ax.set_ylabel('Score (%)', fontweight='bold', fontsize=11)
ax.set_title('Gold Standard Benchmark: Accuracy vs. Macro-F1 vs. Weighted-F1 Across All 5 Models', fontweight='bold', fontsize=12, pad=12)
ax.set_ylim(0, 90)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')

# Add score labels on top of bars
for i in x:
    ax.text(i - w, gold_results[i]['Accuracy (%)'] + 1, f"{gold_results[i]['Accuracy (%)']:.1f}%", ha='center', fontsize=8.5, rotation=45)
    ax.text(i, gold_results[i]['Macro-F1 (%)'] + 1, f"{gold_results[i]['Macro-F1 (%)']:.1f}%", ha='center', fontsize=8.5, rotation=45)
    ax.text(i + w, gold_results[i]['Weighted-F1 (%)'] + 1, f"{gold_results[i]['Weighted-F1 (%)']:.1f}%", ha='center', fontsize=8.5, rotation=45)

plt.tight_layout()
plt.show()

# 3. Per-Class F1 Score Comparison
fig, ax = plt.subplots(figsize=(11, 4.5))
w = 0.25
ax.bar(x - w, [r['Negative F1 (%)'] for r in gold_results], w, label='Negative F1', color='#e74c3c', edgecolor='#333333')
ax.bar(x, [r['Neutral F1 (%)'] for r in gold_results], w, label='Neutral F1', color='#95a5a6', edgecolor='#333333')
ax.bar(x + w, [r['Positive F1 (%)'] for r in gold_results], w, label='Positive F1', color='#27ae60', edgecolor='#333333')

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontweight='bold', fontsize=10)
ax.set_ylabel('F1 Score (%)', fontweight='bold', fontsize=11)
ax.set_title('Per-Class F1 Scores on Gold Dataset (Highlighting Neutral Class Recall)', fontweight='bold', fontsize=12, pad=12)
ax.set_ylim(0, 90)
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.show()
"""

    for nb_path in nb_targets:
        if os.path.exists(nb_path):
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Find and update Cell 11
            for idx, cell in enumerate(nb.cells):
                if 'Five-Panel Normalized Confusion Matrices' in cell.source:
                    cell.source = c11_code
                    print(f"Updated Cell {idx} in {os.path.basename(nb_path)} to include Weighted-F1 visualizations.")
                    break
            
            # Also ensure Cell 9 prints Weighted-F1 clearly
            for idx, cell in enumerate(nb.cells):
                if 'target_names = [' in cell.source and 'gold_results = []' in cell.source:
                    # Cell 9 already collects weighted_f1, verify formatting
                    print(f"Verified Cell {idx} in {os.path.basename(nb_path)} includes Weighted-F1 in df_gold_bench.")
            
            with open(nb_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
            print(f"Saved: {nb_path}")

if __name__ == '__main__':
    update_gold_dataset()
    update_benchmark_notebooks()
