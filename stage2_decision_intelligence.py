import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from src.analytics import load_dataset
from src.decision_engine import (
    compute_restaurant_aspect_health,
    generate_ai_recommendations,
    extract_aspect_drivers
)
from src.llm_engine import generate_llm_executive_strategy

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Stage 2: Restaurant Decision Intelligence Platform",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .health-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .health-score {
        font-size: 2.4rem;
        font-weight: 800;
    }
    .alert-card {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .warning-card {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .growth-card {
        background-color: #F0FDF4;
        border-left: 5px solid #22C55E;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .llm-box {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Control Center
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/000000/restaurant.png", width=70)
st.sidebar.title("Decision Intelligence")
st.sidebar.markdown("**Executive Operations Hub**")

# Load Dataset
@st.cache_data
def get_dataset():
    return load_dataset()

df_dataset = get_dataset()

st.sidebar.markdown("---")
restaurant_list = sorted(df_dataset['Restaurant'].dropna().unique().tolist())
selected_restaurant = st.sidebar.selectbox(
    "🏨 Select Restaurant Brand:",
    options=restaurant_list,
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 LLM Synthesis Settings")
user_api_key = st.sidebar.text_input("Gemini API Key (Optional):", type="password", help="Enter API key for live Gemini LLM executive strategy synthesis.")

# ---------------------------------------------------------
# Main Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">🍽️ Restaurant Decision Intelligence & Executive Platform</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Converting BERT Aspect Sentiments into Prescriptive Operational Actions for <b>{selected_restaurant}</b></div>', unsafe_allow_html=True)

# Compute Health Scores & Drivers
health_data = compute_restaurant_aspect_health(selected_restaurant, df_dataset)
drivers_data = extract_aspect_drivers(selected_restaurant, df_dataset)
ai_recs = generate_ai_recommendations(health_data, selected_restaurant)

# Aspect Health Metric Cards
h_cols = st.columns(5)
aspect_list = ['Food', 'Service', 'Price', 'Ambience / Location', 'Cleanliness']

for idx, asp in enumerate(aspect_list):
    info = health_data.get(asp, {'score': 75.0, 'status': 'Satisfactory', 'color': '#D97706'})
    score = info['score']
    status = info['status']
    color = info['color']
    
    with h_cols[idx]:
        st.markdown(f"""
        <div class="health-card">
            <div style="font-size:0.85rem; color:#64748B; font-weight:600; text-transform:uppercase;">{asp}</div>
            <div class="health-score" style="color:{color};">{score}</div>
            <div style="font-size:0.8rem; font-weight:600; color:{color};">{status}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 AI Managerial Recommendations",
    "🧠 Grounded LLM Strategy Synthesizer",
    "📊 Aspect Scorecard Breakdown",
    "⚔️ Competitive Benchmark",
    "🗣️ Praise vs. Complaint Drivers"
])

# =========================================================
# TAB 1: AI MANAGERIAL RECOMMENDATIONS
# =========================================================
with tab1:
    st.subheader(f"Prescriptive Action Plan for {selected_restaurant}")
    st.markdown("Automated operational instructions generated from fine-tuned NLP aspect sentiment triggers.")
    
    for rec in ai_recs:
        card_class = "alert-card" if rec['severity'] == 'high' else ("warning-card" if rec['severity'] == 'medium' else "growth-card")
        icon = "🚨" if rec['severity'] == 'high' else ("⚠️" if rec['severity'] == 'medium' else "💡")
        
        st.markdown(f"""
        <div class="{card_class}">
            <h4 style="margin:0; color:#0F172A;">{icon} {rec['type']} — {rec['aspect']}</h4>
            <p style="margin:0.4rem 0; color:#334155;"><b>Identified Issue:</b> {rec['issue']}</p>
            <p style="margin:0.4rem 0; color:#0F172A; font-weight:600;"><b>Recommended Managerial Action:</b> {rec['action']}</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 2: GROUNDED LLM STRATEGY SYNTHESIZER
# =========================================================
with tab2:
    st.subheader("Grounded LLM Executive Strategy Synthesis")
    st.markdown("Combines deterministic aspect metrics into a grounded JSON payload to generate executive strategy.")
    
    if st.button("✨ Generate LLM Executive Strategy", type="primary"):
        with st.spinner("Synthesizing grounded strategy from aspect health scores..."):
            llm_res = generate_llm_executive_strategy(selected_restaurant, health_data, drivers_data, api_key=user_api_key)
            
        st.caption(f"Strategy Engine Source: **{llm_res['source']}**")
        st.markdown(f"""
        <div class="llm-box">
            {llm_res['content']}
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🔍 Inspect Grounded JSON Payload Passed to LLM"):
            st.json(llm_res['payload_used'])

# =========================================================
# TAB 3: ASPECT SCORECARD BREAKDOWN
# =========================================================
with tab3:
    st.subheader("Aspect-Level Sentiment Metrics")
    
    scorecard_rows = []
    for asp, d in health_data.items():
        scorecard_rows.append({
            'Aspect Category': asp,
            'Health Index (0-100)': d['score'],
            'Status': d['status'],
            'Positive Mentions': d['pos_count'],
            'Neutral Mentions': d['neu_count'],
            'Negative Complaints': d['neg_count'],
            'Total Aspect Mentions': d['total_mentions']
        })
        
    df_sc = pd.DataFrame(scorecard_rows)
    st.dataframe(df_sc, hide_index=True, use_container_width=True)
    
    # Aspect Health Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Health Index (0-100)', y='Aspect Category', data=df_sc, palette='Greens_r', ax=ax)
    ax.set_title(f"Aspect Health Index — {selected_restaurant}")
    ax.set_xlim(0, 100)
    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.annotate(f'{width:.1f}', (width - 5, p.get_y() + p.get_height() / 2.),
                        ha='center', va='center', color='white', fontweight='bold')
    st.pyplot(fig)

# =========================================================
# TAB 4: COMPETITIVE BENCHMARK
# =========================================================
with tab4:
    st.subheader("Head-to-Head Competitive Benchmark")
    st.markdown("Compare aspect performance against another restaurant in the dataset.")
    
    comp_restaurant = st.selectbox(
        "Select Competitor Restaurant to Compare:",
        options=[r for r in restaurant_list if r != selected_restaurant],
        index=1 if len(restaurant_list) > 1 else 0
    )
    
    health_comp = compute_restaurant_aspect_health(comp_restaurant, df_dataset)
    
    comp_data = []
    for asp in aspect_list:
        s1 = health_data.get(asp, {}).get('score', 75.0)
        s2 = health_comp.get(asp, {}).get('score', 75.0)
        comp_data.append({
            'Aspect': asp,
            selected_restaurant: s1,
            comp_restaurant: s2,
            'Difference': round(s1 - s2, 1)
        })
        
    df_comp = pd.DataFrame(comp_data)
    
    col_comp_table, col_comp_chart = st.columns([2, 3])
    
    with col_comp_table:
        st.markdown("#### Health Score Comparison Table")
        st.dataframe(df_comp, hide_index=True, use_container_width=True)
        
    with col_comp_chart:
        st.markdown("#### Side-by-Side Aspect Comparison Chart")
        df_melt = pd.melt(df_comp, id_vars=['Aspect'], value_vars=[selected_restaurant, comp_restaurant], var_name='Restaurant', value_name='Health Score')
        
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        sns.barplot(x='Aspect', y='Health Score', hue='Restaurant', data=df_melt, palette='Set2', ax=ax2)
        ax2.set_title(f"{selected_restaurant} vs. {comp_restaurant}")
        ax2.set_ylim(0, 100)
        plt.xticks(rotation=15)
        st.pyplot(fig2)

# =========================================================
# TAB 5: PRAISE VS COMPLAINT DRIVERS
# =========================================================
with tab5:
    st.subheader(f"Praise & Complaint Keyword Drivers for {selected_restaurant}")
    
    col_praise, col_complaint = st.columns(2)
    
    with col_praise:
        st.markdown("#### 🟢 Top Praise Drivers (4+ Star Reviews)")
        for kw in drivers_data['praise']:
            st.success(f"✨ Praise Keyword: `{kw}`")
            
    with col_complaint:
        st.markdown("#### 🔴 Top Complaint Drivers (1-2 Star Reviews)")
        for kw in drivers_data['complaints']:
            st.error(f"⚠️ Complaint Keyword: `{kw}`")
