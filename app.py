import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="SMRT Transmission Analyst Pro", layout="wide", page_icon="🚌")

# "Analyst Pro" Styling
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 3rem;}
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 15px;
        border-radius: 6px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .priority-header {
        color: #d63031;
        font-weight: bold;
        padding-bottom: 10px;
        border-bottom: 2px solid #d63031;
        margin-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e3f2fd;
        border-bottom: 3px solid #0984e3;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. ANALYST CALCULATION ENGINE
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_analyze_data():
    try:
        df = pd.read_csv("pred_trans_full_final.csv")
        
        # --- SMART COLUMN CLEANER (Fixes the Depot Issue) ---
        # 1. Strip whitespace from all columns
        df.columns = df.columns.str.strip()
        
        # 2. Rename ambiguous columns to standard names
        # This finds any column containing "Depot" (case insensitive) and renames it to "Depot"
        for col in df.columns:
            if "depot" in col.lower():
                df.rename(columns={col: "Depot"}, inplace=True)
            if "bus" in col.lower() and "no" in col.lower():
                df.rename(columns={col: "Bus No"}, inplace=True)

    except FileNotFoundError:
        st.error("⚠️ Critical: 'pred_trans_full_final.csv' not found.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0

    # --- HARDCODED BUS MODEL ---
    df['Bus Model'] = 'Volvo B9TL'

    # --- A. DATA PARSING ---
    date_cols = ['Registration Date', 'Replacement Date', 'Pred Replacement Date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')

    today = pd.Timestamp.now()
    
    # --- B. FEATURE ENGINEERING ---
    df['Bus_Age_Days'] = (today - df['Registration Date']).dt.days
    df['Bus_Age_Years'] = df['Bus_Age_Days'] / 365.25
    df = df[df['Bus_Age_Days'] > 0].copy() 
    df['Daily_Usage_km'] = df['Odometer_latest'] / df['Bus_Age_Days']
    
    # Replacement Metrics
    df['Repl_Count'] = np.where(df['Replacement Date'].notnull(), 1, 0)
    df['Age_at_Repl'] = np.where(
        df['Replacement Date'].notnull(),
        (df['Replacement Date'] - df['Registration Date']).dt.days / 365.25,
        np.nan
    )
    
    # Current Usage Logic
    def get_comp_usage(row):
        if pd.notnull(row['Replacement Date']):
            days = (today - row['Replacement Date']).days
            return days * row['Daily_Usage_km'] if days >= 0 else row['Odometer_latest']
        return row['Odometer_latest']
    
    df['Current_Comp_Usage'] = df.apply(get_comp_usage, axis=1)

    # --- C. STATISTICAL BASELINES ---
    failures_km = df[df['Replacement Date'].notnull()]['Mileage Extracted']
    target_km = failures_km.median() if not failures_km.empty else 478000
    
    failures_age = df['Age_at_Repl'].dropna()
    target_age = failures_age.median() if not failures_age.empty else 7.5

    # --- D. HEALTH SCORING & METRICS ---
    def calculate_metrics(row):
        # 1. Base Runway
        remaining_km = target_km - row['Current_Comp_Usage']
        
        # 2. Health Score (Lower is Worse)
        age_score = max(0, 1 - (row['Bus_Age_Years'] / 17))
        usage_score = max(0, 1 - (row['Daily_Usage_km'] / 350))
        wear_score = max(0, 1 - (row['Current_Comp_Usage'] / (target_km * 1.2)))
        
        health_score = (0.3 * age_score + 0.3 * usage_score + 0.4 * wear_score) * 100
        
        return pd.Series([remaining_km, health_score])

    df[['Remaining_Km', 'Health_Score']] = df.apply(calculate_metrics, axis=1)

    # --- E. CAPACITY-CONSTRAINED SCHEDULING ---
    urgent_mask = (df['Remaining_Km'] < 20000)
    df_urgent = df[urgent_mask].sort_values('Health_Score', ascending=True).copy()
    df_safe = df[~urgent_mask].copy()
    
    days_buffer = 9 # Approx 3-4 buses per month
    new_dates = []
    new_days_rem = []
    
    for i in range(len(df_urgent)):
        scheduled_days = (i + 1) * days_buffer
        scheduled_date = today + timedelta(days=scheduled_days)
        new_dates.append(scheduled_date)
        new_days_rem.append(scheduled_days)
        
    df_urgent['New_Pred_Date'] = new_dates
    df_urgent['Days_Rem'] = new_days_rem
    
    # Safe Buses Logic
    df_safe['Days_Rem'] = df_safe['Remaining_Km'] / df_safe['Daily_Usage_km']
    df_safe['New_Pred_Date'] = today + pd.to_timedelta(df_safe['Days_Rem'], unit='D')
    
    df = pd.concat([df_urgent, df_safe])
    
    # --- F. STATUS FLAGS ---
    def get_status(days):
        if days <= 30: return "Critical (1 Mo)"
        if days <= 90: return "Priority (3 Mo)"
        if days <= 180: return "Monitor (6 Mo)"
        return "Healthy"
    
    df['Action_Status'] = df['Days_Rem'].apply(get_status)
    df['Pred_Month'] = df['New_Pred_Date'].dt.strftime('%Y-%m')
    df['Pred_Month_Str'] = df['New_Pred_Date'].dt.strftime('%b %Y')

    # --- G. SURVIVAL DATA ---
    def prepare_survival_df(events, censored):
        data = [{'x': e, 'event': 1} for e in events] + [{'x': c, 'event': 0} for c in censored]
        sdf = pd.DataFrame(data).sort_values('x')
        sdf['at_risk'] = range(len(sdf), 0, -1)
        p = 1.0; probs = []
        for i, row in sdf.iterrows():
            if row['event'] == 1: p = p * (1 - 1/row['at_risk'])
            probs.append(p)
        sdf['prob'] = probs
        return sdf

    km_surv = prepare_survival_df(failures_km.tolist(), df['Current_Comp_Usage'].tolist())
    age_surv = prepare_survival_df(failures_age.tolist(), df[df['Replacement Date'].isnull()]['Bus_Age_Years'].tolist())

    return df, km_surv, age_surv, target_km

df, km_surv, age_surv, target_km = load_and_analyze_data()

# Colors
status_colors = {
    'Healthy': '#2ecc71',         'Monitor (6 Mo)': '#f1c40f', 
    'Priority (3 Mo)': '#e67e22', 'Critical (1 Mo)': '#e74c3c'
}

# -----------------------------------------------------------------------------
# 3. DASHBOARD UI
# -----------------------------------------------------------------------------
# Updated Title with Version Number to verify deployment
st.title("🚍 Transmission Maintenance Analyst Dashboard (v3.1)")
st.caption(f"Report Date: {datetime.now().strftime('%d %b %Y')} | Bus Model: Volvo B9TL")

if not df.empty:
    
    # --- METRICS ---
    c1, c2, c3, c4, c5 = st.columns(5)
    crit_count = len(df[df['Action_Status'] == 'Critical (1 Mo)'])
    prio_count = len(df[df['Action_Status'] == 'Priority (3 Mo)'])
    
    c1.metric("Fleet Size", len(df))
    c2.metric("Avg Fleet Age", f"{df['Bus_Age_Years'].mean():.1f} Yrs")
    c3.metric("Avg Failure Mileage", f"{target_km:,.0f} km")
    c4.metric("CRITICAL (<30 Days)", crit_count, delta="Schedule Immediate", delta_color="inverse")
    c5.metric("PRIORITY (<90 Days)", prio_count, delta="Queue Next", delta_color="off")

    st.markdown("---")

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Actionable Insights", "📉 Survival Analysis", "📋 Master Data (Analyst View)"])

    # =========================================================================
    # TAB 1: INSIGHTS
    # =========================================================================
    with tab1:
        c_left, c_right = st.columns([1.5, 1])
        
        with c_left:
            st.subheader("1. Fleet Health Matrix")
            
            # --- SUMMARY 1 (SCATTER) ---
            st.info("""
            **📊 Analyst Interpretation: Risk Matrix**
            * **X-Axis (Usage):** Current component mileage.
            * **Y-Axis (Age):** Bus age. Older buses (Top) with high mileage (Right) fail most frequently.
            * **Red Line:** Historical failure average. Buses past this line are operating on borrowed time.
            """)
            
            fig_scatter = px.scatter(
                df, x='Current_Comp_Usage', y='Bus_Age_Years',
                color='Action_Status', color_discrete_map=status_colors,
                hover_data=['Bus No', 'Pred_Month_Str', 'Health_Score'],
                title="Risk Analysis: Usage vs Age",
            )
            fig_scatter.add_vline(x=target_km, line_dash="dash", line_color="red", annotation_text="Avg Failure")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # --- SUMMARY 2 (DEPOT) ---
            st.subheader("2. Depot Workload Analysis")
            if 'Depot' in df.columns:
                st.info("""
                **🏭 Analyst Interpretation: Depot Load**
                * **Box Height:** Shows variation. A tall box means inconsistent usage (some buses sit, some run hard).
                * **Outliers (Dots):** Specific buses that are being over-utilized compared to their depot peers.
                """)
                
                fig_depot = px.box(
                    df, x='Depot', y='Current_Comp_Usage', 
                    color='Action_Status', color_discrete_map=status_colors,
                    title="Component Mileage Distribution by Depot",
                    points="all"
                )
                st.plotly_chart(fig_depot, use_container_width=True)
            else:
                st.warning("Depot column missing in data.")

        with c_right:
            st.markdown('<div class="priority-header">🔥 Priority Action List</div>', unsafe_allow_html=True)
            st.caption("Auto-scheduled based on capacity (Max 4/Month).")
            
            action_list = df[df['Days_Rem'] <= 90].sort_values('Days_Rem')
            
            if not action_list.empty:
                st.dataframe(
                    action_list[['Bus No', 'Bus Model', 'Pred_Month_Str', 'Current_Comp_Usage']]
                    .style.format({'Current_Comp_Usage': '{:,.0f}'})
                    .applymap(lambda x: 'background-color: #ffcccc' if 'Critical' in str(x) else '', subset=pd.IndexSlice[:, :]),
                    use_container_width=True,
                    height=500
                )
            else:
                st.success("✅ Schedule Clear.")

    # =========================================================================
    # TAB 2: SURVIVAL ANALYSIS
    # =========================================================================
    with tab2:
        st.subheader("Reliability Analysis (AI Models)")
        
        # --- SUMMARY 3 (SURVIVAL) ---
        st.success("""
        **🧠 Analyst Interpretation: Survival Curves**
        * **Orange Line (90% Limit):** This is your 'Safe Operating Limit'. 10% of units fail by this point.
        * **Steep Drop:** Indicates the 'danger zone' where failures accelerate rapidly.
        """)

        c_surv1, c_surv2 = st.columns(2)
        
        with c_surv1:
            st.markdown("##### 📉 Survival by Mileage")
            fig_km = px.line(km_surv, x='x', y='prob', title="Prob. of Survival vs. Km")
            fig_km.add_hline(y=0.9, line_dash="dot", line_color="orange", annotation_text="90% Limit")
            st.plotly_chart(fig_km, use_container_width=True)

        with c_surv2:
            st.markdown("##### 📉 Survival by Age")
            fig_age = px.line(age_surv, x='x', y='prob', title="Prob. of Survival vs. Years")
            fig_age.add_hline(y=0.9, line_dash="dot", line_color="orange", annotation_text="90% Limit")
            st.plotly_chart(fig_age, use_container_width=True)

    # =========================================================================
    # TAB 3: MASTER DATA
    # =========================================================================
    with tab3:
        st.subheader("Master Fleet Register")
        
        with st.expander("📖 Analyst Reference Guide: Calculations & Definitions", expanded=True):
            st.markdown(f"""
            ### 🧮 Health Score Formula (0-100)
            $$ \\text{{Score}} = (30\\% \\times \\text{{Age Factor}}) + (30\\% \\times \\text{{Usage Factor}}) + (40\\% \\times \\text{{Wear Factor}}) $$
            
            ### 📝 Metric Definitions
            | Field Name | Description |
            | :--- | :--- |
            | **Bus Model** | Vehicle Type (Volvo B9TL). |
            | **Current_Comp_Usage** | Mileage on current transmission. Resets if replaced. |
            | **Action_Status** | **Critical:** <30 days. **Priority:** <90 days. |
            | **Target Mileage** | Median historical failure: **{target_km:,.0f} km**. |
            """)

        c_exp1, c_exp2 = st.columns([4, 1])
        with c_exp1: st.write("Exportable dataset with engineered features.")
        with c_exp2:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export CSV", data=csv, file_name="SMRT_Analyst_Report.csv", mime="text/csv", type="primary")

        display_cols = [
            'Bus No', 'Bus Model', 'Action_Status', 'Pred_Month_Str', 'Health_Score', 'Depot',
            'Bus_Age_Years', 'Odometer_latest', 'Current_Comp_Usage',
            'Repl_Count', 'Registration Date', 'Replacement Date'
        ]
        
        format_dict = {
            'Bus_Age_Years': '{:.1f}', 'Odometer_latest': '{:,.0f}',
            'Current_Comp_Usage': '{:,.0f}', 'Health_Score': '{:.0f}',
            'Registration Date': lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else "-",
            'Replacement Date': lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else "-"
        }
        
        valid_cols = [c for c in display_cols if c in df.columns]

        st.dataframe(
            df[valid_cols].sort_values('Action_Status').style
            .format(format_dict)
            .background_gradient(subset=['Health_Score'], cmap='RdYlGn'),
            use_container_width=True, height=600
        )
else:
    st.info("Please upload 'pred_trans_full_final.csv'.")
