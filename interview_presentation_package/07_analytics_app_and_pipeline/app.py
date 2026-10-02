"""
DineSense AI - Aspect-Based Sentiment & Restaurant Analytics Dashboard
========================================================================

Architecture & Data Principles:
- Primary Reporting Unit: Unique (review_id, aspect) pair.
- Conflict Resolution: Mixed sentiment derived when both Positive and Negative
  assertions occur within the same review for an aspect.
- No Aspect Opinion: Strictly treated as an exclusion filter; never included in denominators.
- Empirical Bayes Smoothing: Beta-Binomial shrinkage using dataset-derived aspect priors.
- Descriptive Only: Strictly factual observations grounded in computed metrics.
  No clustering, causal inferences, SOPs, or unexplained composite health scores.
"""

import os
import sys

# Ensure both script directory and project root are in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..')) if 'interview_presentation_package' in SCRIPT_DIR else SCRIPT_DIR
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    from src.analytics import (
        load_and_prepare_data,
        resolve_sentiment_conflict,
        aggregate_review_aspect_units,
        calculate_aspect_metrics,
        calculate_review_level_metrics,
        calculate_time_trends,
        calculate_peer_comparison,
        generate_descriptive_insights,
        VALID_ASPECTS,
        VALID_SENTIMENTS
    )
except ImportError:
    from analytics import (
        load_and_prepare_data,
        resolve_sentiment_conflict,
        aggregate_review_aspect_units,
        calculate_aspect_metrics,
        calculate_review_level_metrics,
        calculate_time_trends,
        calculate_peer_comparison,
        generate_descriptive_insights,
        VALID_ASPECTS,
        VALID_SENTIMENTS
    )

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DineSense AI | Restaurant Aspect Analytics",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.0rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1E293B;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .insight-card {
        background-color: #FFFFFF;
        border-left: 4px solid #3B82F6;
        border: 1px solid #E2E8F0;
        border-left-width: 4px;
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.75rem;
    }
    .badge-info {
        background-color: #EFF6FF;
        color: #1E40AF;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-warn {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Color Scheme for Sentiments
COLOR_MAP = {
    'Positive': '#10B981',   # Emerald Green
    'Negative': '#EF4444',   # Rose Red
    'Neutral':  '#94A3B8',   # Slate Gray
    'Mixed':    '#F59E0B'    # Amber
}

# -----------------------------------------------------------------------------
# Cached Data Loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_analytics_data():
    df_assertions, df_review_aspects = load_and_prepare_data()
    return df_assertions, df_review_aspects

with st.spinner("Loading assertion data and compiling review-aspect aggregations..."):
    df_assertions, df_review_aspects = get_cached_analytics_data()

# -----------------------------------------------------------------------------
# Sidebar Navigation & Filter Controls
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/restaurant-.png", width=64)
st.sidebar.title("DineSense AI")
st.sidebar.caption("Aspect-Based Restaurant Analytics")

st.sidebar.markdown("---")
st.sidebar.subheader("Filters & Scope")

# Restaurant Selector
restaurant_list = sorted([r for r in df_review_aspects['Restaurant'].unique() if pd.notna(r) and r != 'gold_benchmark'])
selected_restaurant = st.sidebar.selectbox(
    "Select Restaurant",
    options=["All Restaurants"] + restaurant_list,
    index=0
)

# Aspect Selector
selected_aspect = st.sidebar.selectbox(
    "Select Target Aspect",
    options=VALID_ASPECTS,
    index=0
)

# Date Filter
valid_dates = df_review_aspects['Review_Date'].dropna()
has_dates = len(valid_dates) > 0
if has_dates:
    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()
    st.sidebar.markdown("**Date Filtering:**")
    enable_date_filter = st.sidebar.checkbox(
        "Filter by Date Range",
        value=False,
        help="Check to restrict reviews strictly to a custom date window."
    )
    if enable_date_filter:
        raw_date_input = st.sidebar.date_input(
            "Review Date Window",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Select start and end dates."
        )
    else:
        raw_date_input = None
else:
    enable_date_filter = False
    raw_date_input = None

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Methodology Notes:**
- **Unit of Analysis:** Unique `(review, aspect)` pairs.
- **Mixed Resolution:** Assigned when a review contains both Positive and Negative clauses for an aspect.
- **Strict Denominators:** Reviews with *No Aspect Opinion* are excluded from sentiment rates.
""")

# -----------------------------------------------------------------------------
# Apply Active Filters to Datasets
# -----------------------------------------------------------------------------
filtered_ra = df_review_aspects.copy()
filtered_assertions = df_assertions.copy()

if selected_restaurant != "All Restaurants":
    filtered_ra = filtered_ra[filtered_ra['Restaurant'] == selected_restaurant]
    filtered_assertions = filtered_assertions[filtered_assertions['Restaurant'] == selected_restaurant]

# Apply Date Filter if enabled
active_date_label = "All Recorded Dates"
if enable_date_filter and raw_date_input is not None:
    if isinstance(raw_date_input, (tuple, list)):
        if len(raw_date_input) == 2:
            d_start, d_end = sorted([raw_date_input[0], raw_date_input[1]])
            active_date_label = f"{d_start} to {d_end}"
        elif len(raw_date_input) == 1:
            d_start = d_end = raw_date_input[0]
            active_date_label = f"{d_start} (Single Day)"
            st.sidebar.caption("👉 *Single day selected. Select second date to expand range.*")
        else:
            d_start, d_end = None, None
    else:
        d_start = d_end = raw_date_input
        active_date_label = f"{d_start}"

    if d_start is not None and d_end is not None:
        start_ts = pd.to_datetime(d_start)
        end_ts = pd.to_datetime(d_end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

        date_mask_ra = filtered_ra['Review_Date'].notna() & (filtered_ra['Review_Date'] >= start_ts) & (filtered_ra['Review_Date'] <= end_ts)
        filtered_ra = filtered_ra[date_mask_ra]

        date_mask_as = filtered_assertions['Review_Date'].notna() & (filtered_assertions['Review_Date'] >= start_ts) & (filtered_assertions['Review_Date'] <= end_ts)
        filtered_assertions = filtered_assertions[date_mask_as]

# Compute All Reviews reference for mention rates
df_all_reviews_subset = filtered_assertions[['review_id', 'Restaurant']].drop_duplicates()

# Calculate Aspect Metrics
aspect_metrics_df = calculate_aspect_metrics(
    filtered_ra,
    df_all_reviews=df_all_reviews_subset,
    restaurant=selected_restaurant if selected_restaurant != "All Restaurants" else None
)

# Review-level Metrics
review_level_df = calculate_review_level_metrics(filtered_ra)

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">DineSense AI Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-title">Scope: <b>{selected_restaurant}</b> | Target Aspect: <b>{selected_aspect}</b> | '
    f'Date Window: <b>{active_date_label}</b> | '
    f'Active Filters: <i>{len(filtered_ra):,} Aspect Mentions across {len(df_all_reviews_subset):,} Eligible Reviews</i></div>',
    unsafe_allow_html=True
)

if filtered_ra.empty:
    st.warning(
        f"⚠️ **No review opinions found** for **{selected_restaurant}** matching the active filter criteria ({active_date_label}). "
        "Please select a broader date range or choose another restaurant."
    )

# Top Navigation Tabs
tab_overview, tab_aspect, tab_trends, tab_peers, tab_insights, tab_explorer = st.tabs([
    "1. Overview",
    "2. Aspect Analytics",
    "3. Time Trends",
    "4. Peer Comparison",
    "5. Descriptive Insights",
    "6. Review Explorer"
])

# =============================================================================
# TAB 1: OVERVIEW
# =============================================================================
with tab_overview:
    st.subheader("Executive Overview & High-Level Aggregates")
    st.caption("Factual high-level distribution of reviews, aspect mention coverage, and overall review-level sentiment.")

    # High level KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-value">{len(df_all_reviews_subset):,}</div>'
            f'<div class="metric-label">Eligible Reviews</div>'
            f'</div>', unsafe_allow_html=True
        )
    with col2:
        num_rest = filtered_ra['Restaurant'].nunique()
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-value">{num_rest:,}</div>'
            f'<div class="metric-label">Restaurants Evaluated</div>'
            f'</div>', unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-value">{len(filtered_ra):,}</div>'
            f'<div class="metric-label">Aspect Opinions Evaluated</div>'
            f'</div>', unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-value">{len(filtered_assertions):,}</div>'
            f'<div class="metric-label">Preserved Clause Assertions</div>'
            f'</div>', unsafe_allow_html=True
        )

    st.markdown("---")

    # Row 2: Aspect Mention Rates & Overall Review Sentiment
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("##### Aspect Mention Rates (% of Eligible Reviews)")
        st.caption("Proportion of eligible reviews that express an opinion on each operational aspect.")

        mention_chart_df = aspect_metrics_df[['Aspect', 'Mention_Rate_Pct', 'Mention_Count', 'Eligible_Reviews']].copy()
        fig_mention = px.bar(
            mention_chart_df,
            x='Aspect',
            y='Mention_Rate_Pct',
            text='Mention_Rate_Pct',
            custom_data=['Mention_Count', 'Eligible_Reviews'],
            color='Aspect',
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_mention.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>Mention Rate: %{y:.1f}%<br>Count: %{customdata[0]} / %{customdata[1]} reviews<extra></extra>"
        )
        fig_mention.update_layout(
            yaxis_title="Mention Rate (%)",
            xaxis_title="",
            showlegend=False,
            height=320,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_mention, use_container_width=True)

    with col_right:
        st.markdown("##### Overall Review-Level Sentiment")
        st.caption("Derived across all aspects mentioned per review (Positive / Negative / Neutral / Mixed).")

        rev_counts = review_level_df['Overall_Sentiment'].value_counts().reset_index()
        rev_counts.columns = ['Sentiment', 'Count']
        total_rev_op = rev_counts['Count'].sum()
        rev_counts['Pct'] = (rev_counts['Count'] / total_rev_op * 100).round(1)

        fig_rev = px.pie(
            rev_counts,
            names='Sentiment',
            values='Count',
            color='Sentiment',
            color_discrete_map=COLOR_MAP,
            hole=0.45
        )
        fig_rev.update_traces(
            textinfo='percent+label',
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>"
        )
        fig_rev.update_layout(
            height=320,
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    st.markdown("##### Aspect Metrics Summary Table")
    display_summary_cols = ['Aspect', 'Eligible_Reviews', 'Mention_Count', 'Mention_Rate_Pct',
                            'Positive_Count', 'Positive_Pct', 'Negative_Count', 'Negative_Pct',
                            'Neutral_Count', 'Neutral_Pct', 'Mixed_Count', 'Mixed_Pct']
    st.dataframe(
        aspect_metrics_df[display_summary_cols].style.format({
            'Mention_Rate_Pct': '{:.1f}%',
            'Positive_Pct': '{:.1f}%',
            'Negative_Pct': '{:.1f}%',
            'Neutral_Pct': '{:.1f}%',
            'Mixed_Pct': '{:.1f}%',
        }),
        use_container_width=True,
        hide_index=True
    )


# =============================================================================
# TAB 2: ASPECT ANALYTICS
# =============================================================================
with tab_aspect:
    st.subheader(f"In-Depth Aspect Breakdown: {selected_aspect}")
    st.caption("Examine sentiment distributions, raw counts, percentages, and optional Empirical Bayes smoothing.")

    aspect_row = aspect_metrics_df[aspect_metrics_df['Aspect'] == selected_aspect]

    if not aspect_row.empty:
        ar = aspect_row.iloc[0]
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1:
            st.metric("Mention Rate", f"{ar['Mention_Rate_Pct']:.1f}%", f"{ar['Mention_Count']} / {ar['Eligible_Reviews']} revs")
        with col_m2:
            st.metric("Positive", f"{ar['Positive_Pct']:.1f}%", f"{ar['Positive_Count']} mentions")
        with col_m3:
            st.metric("Negative", f"{ar['Negative_Pct']:.1f}%", f"{ar['Negative_Count']} mentions")
        with col_m4:
            st.metric("Neutral", f"{ar['Neutral_Pct']:.1f}%", f"{ar['Neutral_Count']} mentions")
        with col_m5:
            st.metric("Mixed", f"{ar['Mixed_Pct']:.1f}%", f"{ar['Mixed_Count']} mentions")

        st.markdown("---")

        # Sentiment Distribution Breakdown Chart
        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            st.markdown(f"##### Sentiment Breakdown for {selected_aspect}")
            sentiment_bar_df = pd.DataFrame([
                {'Sentiment': 'Positive', 'Count': ar['Positive_Count'], 'Percentage': ar['Positive_Pct']},
                {'Sentiment': 'Negative', 'Count': ar['Negative_Count'], 'Percentage': ar['Negative_Pct']},
                {'Sentiment': 'Neutral',  'Count': ar['Neutral_Count'],  'Percentage': ar['Neutral_Pct']},
                {'Sentiment': 'Mixed',    'Count': ar['Mixed_Count'],    'Percentage': ar['Mixed_Pct']}
            ])

            fig_asp_bar = px.bar(
                sentiment_bar_df,
                x='Sentiment',
                y='Count',
                text='Percentage',
                color='Sentiment',
                color_discrete_map=COLOR_MAP
            )
            fig_asp_bar.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside',
                hovertemplate="<b>%{x}</b><br>Count: %{y}<br>Share: %{text:.1f}% of mentions<extra></extra>"
            )
            fig_asp_bar.update_layout(
                yaxis_title="Unique Review Mentions",
                xaxis_title="",
                showlegend=False,
                height=320,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_asp_bar, use_container_width=True)

        with col_c2:
            st.markdown("##### Sentiment Polar Balance")
            net_sentiment = ar['Positive_Pct'] - ar['Negative_Pct']
            st.write(
                f"**Aspect:** `{selected_aspect}`\n\n"
                f"- **Eligible Reviews Analyzed:** {int(ar['Eligible_Reviews']):,}\n"
                f"- **Unique Reviews Mentioning:** {int(ar['Mention_Count']):,} ({ar['Mention_Rate_Pct']:.1f}%)\n"
                f"- **Positive Sentiment:** {ar['Positive_Pct']:.1f}% ({int(ar['Positive_Count']):,} mentions)\n"
                f"- **Negative Sentiment:** {ar['Negative_Pct']:.1f}% ({int(ar['Negative_Count']):,} mentions)\n"
                f"- **Neutral Sentiment:** {ar['Neutral_Pct']:.1f}% ({int(ar['Neutral_Count']):,} mentions)\n"
                f"- **Mixed Sentiment:** {ar['Mixed_Pct']:.1f}% ({int(ar['Mixed_Count']):,} mentions)\n\n"
                f"**Net Aspect Polarity (% Pos - % Neg):** `{net_sentiment:+.1f}%`"
            )
            if ar['Mention_Count'] < 10:
                st.warning(f"⚠️ Small sample size (N={ar['Mention_Count']} mentions < 10). Interpret rates with caution.")
            else:
                st.success(f"✓ Reliable sample size (N={ar['Mention_Count']} mentions).")

        st.markdown("---")
        st.markdown("##### Cross-Aspect Sentiment Comparison")
        st.caption("Compare sentiment proportions across all 5 operational aspects side-by-side.")

        # Melt aspect metrics for 100% stacked bar chart
        stacked_records = []
        for _, row in aspect_metrics_df.iterrows():
            for sent in ['Positive', 'Negative', 'Neutral', 'Mixed']:
                stacked_records.append({
                    'Aspect': row['Aspect'],
                    'Sentiment': sent,
                    'Count': row[f'{sent}_Count'],
                    'Percentage': row[f'{sent}_Pct'],
                    'Denominator': row['Mention_Count']
                })
        df_stacked = pd.DataFrame(stacked_records)

        fig_stack = px.bar(
            df_stacked,
            x='Percentage',
            y='Aspect',
            color='Sentiment',
            orientation='h',
            color_discrete_map=COLOR_MAP,
            custom_data=['Count', 'Denominator']
        )
        fig_stack.update_traces(
            hovertemplate="<b>%{y}</b> - %{fullData.name}<br>Share: %{x:.1f}%<br>Count: %{customdata[0]} / %{customdata[1]} mentions<extra></extra>"
        )
        fig_stack.update_layout(
            barmode='stack',
            xaxis_title="Sentiment Share (% of Mentions)",
            yaxis_title="",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_stack, use_container_width=True)
    else:
        st.warning(f"No records found for aspect {selected_aspect}.")


# =============================================================================
# TAB 3: TIME TRENDS
# =============================================================================
with tab_trends:
    st.subheader(f"Monthly Sentiment Trends: {selected_aspect}")
    st.caption("Trends are evaluated on actual review dates from the verified corpus. Visible sample sizes (N) prevent misleading conclusions from low-volume months.")

    time_df = calculate_time_trends(
        filtered_ra,
        aspect=selected_aspect,
        restaurant=selected_restaurant if selected_restaurant != "All Restaurants" else None
    )

    if not time_df.empty:
        warn_col = 'Small_Sample_Warning' if 'Small_Sample_Warning' in time_df.columns else 'Sample_Size_Warning'
        vol_col = 'Total_Mentions' if 'Total_Mentions' in time_df.columns else 'Sample_Size'

        small_sample_months = time_df[time_df[warn_col]]
        if not small_sample_months.empty:
            st.warning(
                f"⚠️ Note on Sample Sizes: {len(small_sample_months)} monthly period(s) have fewer than 10 reviews. "
                "Fluctuations in these periods should be interpreted with caution."
            )

        col_t1, col_t2 = st.columns([3, 2])

        with col_t1:
            st.markdown("##### Monthly Sentiment Share (% of Aspect Mentions)")
            fig_trends = go.Figure()
            for sent in ['Positive', 'Negative', 'Neutral', 'Mixed']:
                fig_trends.add_trace(go.Scatter(
                    x=time_df['YearMonth'],
                    y=time_df[f'{sent}_Pct'],
                    mode='lines+markers',
                    name=sent,
                    line=dict(color=COLOR_MAP[sent], width=2),
                    hovertemplate=f"<b>%{{x}}</b><br>{sent}: %{{y:.1f}}%<extra></extra>"
                ))
            fig_trends.update_layout(
                yaxis_title="Share of Mentions (%)",
                xaxis_title="Month",
                height=340,
                hovermode='x unified',
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_trends, use_container_width=True)

        with col_t2:
            st.markdown("##### Monthly Review Volume (Sample Size N)")
            fig_vol = px.bar(
                time_df,
                x='YearMonth',
                y=vol_col,
                text=vol_col,
                color=warn_col,
                color_discrete_map={True: '#FCA5A5', False: '#93C5FD'},
                labels={warn_col: 'N < 10'}
            )
            fig_vol.update_traces(
                textposition='outside',
                hovertemplate="<b>%{x}</b><br>Sample Size: %{y} mentions<extra></extra>"
            )
            fig_vol.update_layout(
                yaxis_title="Number of Mentions",
                xaxis_title="Month",
                showlegend=False,
                height=340,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        st.markdown("##### Monthly Metrics Table")
        disp_cols = [c for c in ['YearMonth', vol_col, 'Positive_Count', 'Positive_Pct',
                                 'Negative_Count', 'Negative_Pct', 'Neutral_Count', 'Neutral_Pct',
                                 'Mixed_Count', 'Mixed_Pct', warn_col] if c in time_df.columns]
        st.dataframe(
            time_df[disp_cols].style.format({
                'Positive_Pct': '{:.1f}%',
                'Negative_Pct': '{:.1f}%',
                'Neutral_Pct': '{:.1f}%',
                'Mixed_Pct': '{:.1f}%',
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No dated review records are available for the current selection to compute time trends.")


# =============================================================================
# TAB 4: PEER COMPARISON
# =============================================================================
with tab_peers:
    st.subheader(f"Peer Restaurant Comparison for Aspect: {selected_aspect}")
    st.caption("Compares restaurants using identical aspect definitions, eligibility rules, and underlying denominators. No arbitrary composite rankings are introduced.")

    all_rest_names = sorted([r for r in df_review_aspects['Restaurant'].unique() if pd.notna(r) and r != 'gold_benchmark'])

    col_p1, col_p2 = st.columns([1, 2])
    with col_p1:
        target_rest = st.selectbox(
            "Target Restaurant",
            options=all_rest_names,
            index=0 if selected_restaurant == "All Restaurants" else all_rest_names.index(selected_restaurant) if selected_restaurant in all_rest_names else 0
        )
        default_peers = [r for r in all_rest_names if r != target_rest][:5]
        peer_rests = st.multiselect(
            "Select Peer Group",
            options=[r for r in all_rest_names if r != target_rest],
            default=default_peers
        )

    with col_p2:
        if peer_rests:
            peer_comp_df = calculate_peer_comparison(
                df_review_aspects,
                target_restaurant=target_rest,
                peer_restaurants=peer_rests,
                aspect=selected_aspect
            )

            fig_peer = go.Figure()
            fig_peer.add_trace(go.Bar(
                name='Positive %',
                x=peer_comp_df['Restaurant'],
                y=peer_comp_df['Positive_Pct'],
                marker_color=COLOR_MAP['Positive'],
                text=peer_comp_df['Positive_Pct'].apply(lambda v: f"{v:.1f}%"),
                textposition='auto',
                customdata=peer_comp_df['Mention_Count'],
                hovertemplate="Positive: %{y:.1f}%<br>Total Mentions: %{customdata}<extra></extra>"
            ))
            fig_peer.add_trace(go.Bar(
                name='Negative %',
                x=peer_comp_df['Restaurant'],
                y=peer_comp_df['Negative_Pct'],
                marker_color=COLOR_MAP['Negative'],
                text=peer_comp_df['Negative_Pct'].apply(lambda v: f"{v:.1f}%"),
                textposition='auto',
                customdata=peer_comp_df['Mention_Count'],
                hovertemplate="Negative: %{y:.1f}%<br>Total Mentions: %{customdata}<extra></extra>"
            ))
            fig_peer.update_layout(
                barmode='group',
                yaxis_title="Sentiment Share (%)",
                xaxis_title="",
                height=340,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_peer, use_container_width=True)
        else:
            st.info("Please select at least one peer restaurant to compare.")

    if peer_rests and not peer_comp_df.empty:
        st.markdown("##### Detailed Comparison Metrics")
        st.dataframe(
            peer_comp_df[['Restaurant', 'Role', 'Mention_Count', 'Positive_Count', 'Positive_Pct',
                          'Negative_Count', 'Negative_Pct', 'Neutral_Count', 'Neutral_Pct',
                          'Mixed_Count', 'Mixed_Pct']].style.format({
                'Positive_Pct': '{:.1f}%',
                'Negative_Pct': '{:.1f}%',
                'Neutral_Pct': '{:.1f}%',
                'Mixed_Pct': '{:.1f}%',
            }),
            use_container_width=True,
            hide_index=True
        )


# =============================================================================
# TAB 5: DESCRIPTIVE INSIGHTS
# =============================================================================
with tab_insights:
    st.subheader("Data-Grounded Descriptive Insights")
    st.caption("Factual observations generated strictly from computed metrics. No causal claims, SOP recommendations, or cluster inferences are included.")

    insights_list = generate_descriptive_insights(
        aspect_metrics_df,
        restaurant_name=selected_restaurant
    )

    if insights_list:
        for ins in insights_list:
            aspect_tag = ins.get('Aspect', ins.get('Category', 'General'))
            metric_support = ins.get('Metric_Support', ins.get('Metric', ''))
            small_sample = ins.get('Small_Sample_Warning', ins.get('Sample_Size', 10) < 10)

            warn_badge = '<span class="badge-warn">⚠️ Small Sample Limitation</span>' if small_sample else ''

            st.markdown(
                f'<div class="insight-card">'
                f'<b>{ins["Observation"]}</b> {warn_badge}<br>'
                f'<span style="font-size: 0.85rem; color: #64748B;">Supporting Metric: {metric_support} (Aspect: {aspect_tag})</span>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("Insufficient data to generate descriptive observations for the selected scope.")


# =============================================================================
# TAB 6: REVIEW EXPLORER
# =============================================================================
with tab_explorer:
    st.subheader("Traceability & Review Explorer")
    st.caption("Inspect individual reviews or underlying clause-level assertions behind aggregated metrics.")

    subtab_reviews, subtab_clauses = st.tabs(["Review-Aspect Units (Primary Unit)", "Underlying Clause Assertions (Raw Model Output)"])

    with subtab_reviews:
        st.markdown("##### Review-Aspect Pair View")
        st.caption("Shows deduplicated records where conflicting clauses were reconciled into Positive, Negative, Neutral, or Mixed.")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_aspect_ra = st.multiselect("Filter Aspect", options=VALID_ASPECTS, default=VALID_ASPECTS, key="ra_asp")
        with col_f2:
            filter_sent_ra = st.multiselect("Filter Sentiment", options=['Positive', 'Negative', 'Neutral', 'Mixed'],
                                            default=['Positive', 'Negative', 'Neutral', 'Mixed'], key="ra_sent")

        display_ra = filtered_ra[
            filtered_ra['aspect'].isin(filter_aspect_ra) &
            filtered_ra['sentiment'].isin(filter_sent_ra)
        ]

        st.write(f"Displaying **{len(display_ra):,}** review-aspect records:")
        cols_to_show_ra = ['review_id', 'Restaurant', 'aspect', 'sentiment', 'YearMonth', 'Reviewer', 'clause_text']
        available_cols_ra = [c for c in cols_to_show_ra if c in display_ra.columns]
        st.dataframe(display_ra[available_cols_ra], use_container_width=True, hide_index=True)

    with subtab_clauses:
        st.markdown("##### Preserved Assertion-Level Clauses")
        st.caption("The raw model inference outputs at the individual clause assertion level.")

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            filter_aspect_as = st.multiselect("Filter Aspect", options=VALID_ASPECTS, default=VALID_ASPECTS, key="as_asp")
        with col_c2:
            filter_sent_as = st.multiselect("Filter Sentiment", options=VALID_SENTIMENTS, default=VALID_SENTIMENTS, key="as_sent")

        display_as = filtered_assertions[
            filtered_assertions['aspect'].isin(filter_aspect_as) &
            filtered_assertions['sentiment'].isin(filter_sent_as)
        ]

        st.write(f"Displaying **{len(display_as):,}** clause assertion records:")
        cols_to_show_as = ['review_id', 'Restaurant', 'clause_text', 'aspect', 'sentiment', 'confidence', 'YearMonth']
        available_cols_as = [c for c in cols_to_show_as if c in display_as.columns]
        st.dataframe(display_as[available_cols_as], use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("DineSense AI • Enterprise Restaurant Review Analytics • Built with Streamlit & Plotly")
