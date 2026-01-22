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

# CONFIRMED FILE MAPPING
FILE_B9TL = "pred_trans_full_final.csv"
FILE_A95 = "AI_Predictive_Maintenance_Dashboard_Full.csv"

# Styling
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
# 2. DATA LOADING & MERGING ENGINE
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_combine_data():
    dfs = []
    
    # --- LOAD VOLVO B9TL ---
    try:
        df_b = pd.read_csv(FILE_B9TL)
        df_b.columns = df_b.columns.str.strip() 
        df_b = df_b.loc[:, ~df_b.columns.duplicated()] # Drop Duplicates
        
        df_b['Bus Model'] = 'Volvo B9TL'
        
        # Map to Standard Variables
        b_map = {
            'Reg No.': 'Bus No', 
            'Bus Number': 'Bus No',
            'Registration Date': 'Registration Date',
            'Replacement Date': 'Replacement Date',
            'Pred Replacement Date': 'Pred Replacement Date',
            'Odometer': 'Odometer_latest',
            'Mileage Extracted': 'Mileage Extracted'
        }
        df_b.rename(columns=b_map, inplace=True)
        df_b = df_b.loc[:, ~df_b.columns.duplicated()]
        dfs.append(df_b)
    except FileNotFoundError:
        st.warning(f"⚠️ '{FILE_B9TL}' not found.")

    # --- LOAD MAN A95 ---
    try:
        df_a = pd.read_csv(FILE_A95)
        df_a.columns = df_a.columns.str.strip()
        df_a = df_a.loc[:, ~df_a.columns.duplicated()]
        
        df_a['Bus Model'] = 'MAN A95'
        
        # Map to Standard Variables
        a_map = {
            'Bus No': 'Bus No',
            'Registration Date': 'Registration Date', 'Reg Date': 'Registration Date',
            'Last Replacement Date': 'Replacement Date',
            'Predicted Replacement Date': 'Pred Replacement Date',
            'Current_Odometer_Est': 'Odometer_latest',
            'Mileage_Fail': 'Mileage Extracted'
        }
        df_a.rename(columns=a_map, inplace=True)
        df_a = df_a.loc[:, ~df_a.columns.duplicated()]
        dfs.append(df_a)
    except FileNotFoundError:
        st.warning(f"⚠️ '{FILE_A95}' not found.")

    if not dfs: return pd.DataFrame()
    
    # Combine
    df = pd.concat(dfs, ignore_index=True, sort=False)

    # --- SMART COLUMN CLEANER ---
    for col in df.columns:
        if "depot" in col.lower():
            df.rename(columns={col: "Depot"}, inplace=True)
        if "bus" in col.lower() and "no" in col.lower():
            df.rename(columns={col: "Bus No"}, inplace=True)

    df = df.loc[:, ~df.columns.duplicated()]
    return df

# -----------------------------------------------------------------------------
# 3. ANALYTIC ENGINE
# -----------------------------------------------------------------------------
def analyze_data(df, selected_model):
    
    # FILTER BY MODEL
    if selected_model != "All":
        df = df[df['Bus Model'] == selected_model].copy()

    if df.empty: return df, pd.DataFrame(), pd.DataFrame(), 0

    # --- A. DATA PARSING ---
    date_cols = ['Registration Date', 'Replacement Date', 'Pred Replacement Date']
    for col in date_cols:
        if col in df.columns:
            # Try parsing with dayfirst=True for DD/MM/YYYY support
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    today = pd.Timestamp.now()
    
    # --- B. FEATURE ENGINEERING ---
    if 'Registration Date' not in df.columns: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0

    df['Bus_Age_Days'] = (today - df['Registration Date']).dt.days
    df['Bus_Age_Years'] = df['Bus_Age_Days'] / 365.25
    
    # FILTER: Only keep valid buses (Age > 0)
    df = df[df['Bus_Age_Days'] > 0].copy()
    
    if df.empty: return df, pd.DataFrame(), pd.DataFrame(), 0
    
    # Odometer Fill
    if 'Odometer_latest' not in df.columns: df['Odometer_latest'] = df['Bus_Age_Days'] * 220
    df['Odometer_latest'] = pd.to_numeric(df['Odometer_latest'], errors='coerce').fillna(df['Bus_Age_Days'] * 220)
    
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
    if 'Mileage Extracted' in df.columns and df['Mileage Extracted'].count() > 5:
        failures_km = df[df['Replacement Date'].notnull()]['Mileage Extracted']
        target_km = failures_km.median()
    else:
        est_fail = (df[df['Replacement Date'].notnull()]['Age_at_Repl'] * 365.25 * df[df['Replacement Date'].notnull()]['Daily_Usage_km'])
        target_km = est_fail.median() if not est_fail.empty else 478000
    
    failures_age = df['Age_at_Repl'].dropna()

    # --- D. HEALTH SCORING & METRICS ---
    def calculate_metrics(row):
        remaining_km = target_km - row['Current_Comp_Usage']
        age_score = max(0, 1 - (row['Bus_Age_Years'] / 19))
        usage_score = max(0, 1 - (row['Daily_Usage_km'] / 350))
        wear_score = max(0, 1 - (row['Current_Comp_Usage'] / (target_km * 1.2)))
        health_score = (0.3 * age_score + 0.3 * usage_score + 0.4 * wear_score) * 100
        # Return as LIST to guarantee DataFrame creation
        return [remaining_km, health_score]

    # Robust Apply
    metrics_list = df.apply(calculate_metrics, axis=1).tolist()
    metrics_df = pd.DataFrame(metrics_list, index=df.index, columns=['Remaining_Km', 'Health_Score'])
    
    df['Remaining_Km'] = metrics_df['Remaining_Km']
    df['Health_Score'] = metrics_df['Health_Score']

    # --- E. CAPACITY-CONSTRAINED SCHEDULING ---
    urgent_mask = (df['Remaining_Km'] < 20000)
    df_urgent = df[urgent_mask].sort_values('Health_Score', ascending=True).copy()
    df_safe = df[~urgent_mask].copy()
    
    days_buffer = 9 
    new_dates = []
    new_days_rem = []
    
    for i in range(len(df_urgent)):
        scheduled_days = (i + 1) * days_buffer
        scheduled_date = today + timedelta(days=scheduled_days)
        new_dates.append(scheduled_date)
        new_days_rem.append(scheduled_days)
        
    df_urgent['New_Pred_Date'] = new_dates
    df_urgent['Days_Rem'] = new_days_rem
    
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
        if len(events) < 2: return pd.DataFrame(columns=['x', 'prob'])
        data = [{'x': e, 'event': 1} for e in events] + [{'x': c, 'event': 0} for c in censored]
        sdf = pd.DataFrame(data).sort_values('x')
        sdf['at_risk'] = range(len(sdf), 0, -1)
        p = 1.0; probs = []
        for i, row in sdf.iterrows():
            if row['event'] == 1: p = p * (1 - 1/row['at_risk'])
            probs.append(p)
        sdf['prob'] = probs
        return sdf
    
    if 'Mileage Extracted' in df.columns and df['Mileage Extracted'].count() > 5:
        fail_list = df[df['Replacement Date'].notnull()]['Mileage Extracted'].tolist()
    else:
        fail_list = (df[df['Replacement Date'].notnull()]['Age_at_Repl'] * 365.25 * df[df['Replacement Date'].notnull()]['Daily_Usage_km']).tolist()

    km_surv = prepare_survival_df(fail_list, df['Current_Comp_Usage'].tolist())
    age_surv = prepare_survival_df(failures_age.tolist(), df[df['Replacement Date'].isnull()]['Bus_Age_Years'].tolist())

    return df, km_surv, age_surv, target_km

# -----------------------------------------------------------------------------
# 4. SIDEBAR FILTER
# -----------------------------------------------------------------------------
raw_df = load_and_combine_data()

with st.sidebar:
    st.header("🔍 Fleet Selection")
    if not raw_df.empty:
        # Get raw model names
        raw_models = sorted(raw_df['Bus Model'].unique())
        
        # DISPLAY MAPPING (Renaming for Dropdown)
        display_map = {
            "Volvo B9TL": "Transmission - B9TL",
            "MAN A95": "Transmission - A95"
        }
        
        # Reverse map to find actual value based on display name
        reverse_map = {v: k for k, v in display_map.items()}
        
        # Create display list
        display_options = [display_map.get(m, m) for m in raw_models]
        
        # Default Selection
        default_idx = 0
        if "Transmission - B9TL" in display_options:
            default_idx = display_options.index("Transmission - B9TL")
            
        selected_display = st.selectbox("Select Bus Model", display_options, index=default_idx)
        
        # Convert back to internal name for processing
        selected_model = reverse_map.get(selected_display, selected_display)
        
    else:
        selected_model = "All"
        selected_display = "All"

# Process Selected Fleet
df, km_surv, age_surv, target_km = analyze_data(raw_df, selected_model)

# Colors
status_colors = {
    'Healthy': '#2ecc71',         'Monitor (6 Mo)': '#f1c40f', 
    'Priority (3 Mo)': '#e67e22', 'Critical (1 Mo)': '#e74c3c'
}

# -----------------------------------------------------------------------------
# 5. DASHBOARD UI
# -----------------------------------------------------------------------------
st.title(f"🚍 Transmission Maintenance Analyst Dashboard")
st.caption(f"Report Date: {datetime.now().strftime('%d %b %Y')} | Bus Model: {selected_display}")

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
            
            st.info(f"""
            **📊 Analyst Interpretation: Risk Matrix ({selected_display})**
            * **X-Axis (Usage):** Current component mileage.
            * **Y-Axis (Age):** Bus age. Older buses (Top) with high mileage (Right) fail most frequently.
            * **Red Line:** Historical failure average. Buses past this line are operating on borrowed time.
            """)
            
            # Ensure unique columns for plot
            plot_df = df.loc[:, ~df.columns.duplicated()]
            
            fig_scatter = px.scatter(
                plot_df, x='Current_Comp_Usage', y='Bus_Age_Years',
                color='Action_Status', color_discrete_map=status_colors,
                hover_data=['Bus No', 'Pred_Month_Str', 'Health_Score'],
                title="Risk Analysis: Usage vs Age",
            )
            fig_scatter.add_vline(x=target_km, line_dash="dash", line_color="red", annotation_text="Avg Failure")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # --- DEPOT ---
            st.subheader("2. Depot Workload Analysis")
            if 'Depot' in df.columns:
                st.info("""
                **🏭 Analyst Interpretation: Depot Load**
                * **Box Height:** Shows variation. A tall box means inconsistent usage (some buses sit, some run hard).
                * **Outliers (Dots):** Specific buses that are being over-utilized compared to their depot peers.
                """)
                
                fig_depot = px.box(
                    plot_df, x='Depot', y='Current_Comp_Usage', 
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
        st.subheader(f"Reliability Analysis ({selected_display})")
        
        st.success("""
        **🧠 Analyst Interpretation: Survival Curves**
        * **Orange Line (90% Limit):** This is your 'Safe Operating Limit'. 10% of units fail by this point.
        * **Steep Drop:** Indicates the 'danger zone' where failures accelerate rapidly.
        """)

        c_surv1, c_surv2 = st.columns(2)
        
        with c_surv1:
            st.markdown("##### 📉 Survival by Mileage")
            if not km_surv.empty:
                fig_km = px.line(km_surv, x='x', y='prob', title="Prob. of Survival vs. Km")
                fig_km.add_hline(y=0.9, line_dash="dot", line_color="orange", annotation_text="90% Limit")
                st.plotly_chart(fig_km, use_container_width=True)
            else:
                st.info("Insufficient data for Mileage curve.")

        with c_surv2:
            st.markdown("##### 📉 Survival by Age")
            if not age_surv.empty:
                fig_age = px.line(age_surv, x='x', y='prob', title="Prob. of Survival vs. Years")
                fig_age.add_hline(y=0.9, line_dash="dot", line_color="orange", annotation_text="90% Limit")
                st.plotly_chart(fig_age, use_container_width=True)
            else:
                st.info("Insufficient data for Age curve.")

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
            | **Bus Model** | {selected_display} |
            | **Current_Comp_Usage** | Mileage on current transmission. Resets if replaced. |
            | **Action_Status** | **Critical:** <30 days. **Priority:** <90 days. |
            | **Target Mileage** | Median historical failure: **{target_km:,.0f} km**. |
            """)

        c_exp1, c_exp2 = st.columns([4, 1])
        with c_exp1: st.write("Exportable dataset with engineered features.")
        with c_exp2:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export CSV", data=csv, file_name=f"SMRT_Analyst_Report_{selected_model}.csv", mime="text/csv", type="primary")

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
    st.info("Please ensure 'pred_trans_full_final.csv' and 'AI_Predictive_Maintenance_Dashboard_Full.csv' are in the directory.")
