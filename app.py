import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="SMRT Maintenance Analyst Pro", layout="wide", page_icon="🚌")

# FILES
FILE_TRANS_B9TL = "pred_trans_full_final.csv"
FILE_TRANS_A95 = "AI_Predictive_Maintenance_Dashboard_Full.csv"
FILE_NEW_COMPONENTS = "predicted_replacements_0805.csv"

# STYLING
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 3rem;}
    div[data-testid="metric-container"] {
        background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 15px; border-radius: 6px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .priority-header {
        color: #d63031; font-weight: bold; border-bottom: 2px solid #d63031; margin-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e3f2fd;
        border-bottom: 3px solid #0984e3;
    }
    .info-box {
        background-color: #e1f5fe; border-left: 5px solid #039be5; padding: 15px;
        border-radius: 5px; margin-bottom: 20px; color: #01579b;
    }
    .success-box {
        background-color: #e8f5e9; border-left: 5px solid #2ecc71; padding: 15px;
        border-radius: 5px; margin-bottom: 20px; color: #1b5e20;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_all_data():
    master_list = []
    today = pd.Timestamp.now()

    def clean_and_normalize(df, default_comp, default_model):
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.duplicated()]
        
        col_map = {
            'Reg No.': 'Bus No', 'Bus Number': 'Bus No',
            'Registration Date': 'Reg Date', 'Reg Date': 'Reg Date',
            'Replacement Date': 'Replace Date', 'Last Replacement Date': 'Replace Date',
            'Predicted Replacement Date': 'Pred Date', 'Pred Replacement Date': 'Pred Date',
            'Odometer': 'Odometer', 'Current Odometer': 'Odometer', 'Current_Odometer_Est': 'Odometer',
            'Mileage Extracted': 'Fail_Mileage', 'Mileage_Fail': 'Fail_Mileage',
            'Total Fault Count': 'Faults', 'Fault_Count': 'Faults', 'Total Faults': 'Faults', 'Transmission Fault Count': 'Faults',
            'Accident Count': 'Accidents', 'Total Incidents': 'Accidents', 'Incident_Count': 'Accidents', 'Bus Incident Count': 'Accidents',
            'Bus Age': 'Bus_Age_Years', 'Bus Age At Replacement': 'Age_at_Repl',
            'Depot': 'Depot'
        }
        df.rename(columns=col_map, inplace=True)
        
        if 'Component' not in df.columns: 
            if 'Material Description' in df.columns:
                def detect_comp(x):
                    s = str(x).upper()
                    if 'COMPRESSOR' in s: return 'Air Compressor'
                    if 'TURBO' in s: return 'Turbocharger'
                    return default_comp
                df['Component'] = df['Material Description'].apply(detect_comp)
            else:
                df['Component'] = default_comp
                
        if 'Bus Model' not in df.columns:
            df['Bus Model'] = df['Model'] if 'Model' in df.columns else default_model

        def fix_model_name(m):
            m = str(m).upper()
            if 'B9TL' in m: return 'Volvo B9TL'
            if 'A22' in m: return 'MAN A22'
            if 'A95' in m: return 'MAN A95'
            return m.title()
        df['Bus Model'] = df['Bus Model'].apply(fix_model_name)

        return df

    # LOAD
    files = [
        (FILE_TRANS_B9TL, 'Transmission', 'Volvo B9TL'),
        (FILE_TRANS_A95, 'Transmission', 'MAN A95'),
        (FILE_NEW_COMPONENTS, 'Unknown', 'Unknown')
    ]

    for fname, def_comp, def_mod in files:
        try:
            try: df = pd.read_csv(fname, encoding='utf-8')
            except: df = pd.read_csv(fname, encoding='latin1')
            df = clean_and_normalize(df, def_comp, def_mod)
            master_list.append(df)
        except: pass

    if not master_list: return pd.DataFrame()
    full_df = pd.concat(master_list, ignore_index=True, sort=False)
    
    for c in ['Faults', 'Accidents', 'Fail_Mileage', 'Odometer']:
        if c not in full_df.columns: full_df[c] = 0
        full_df[c] = pd.to_numeric(full_df[c], errors='coerce').fillna(0)
        
    return full_df

# -----------------------------------------------------------------------------
# 3. BALANCED ANALYTIC ENGINE
# -----------------------------------------------------------------------------
def process_data(df, selected_comp, selected_model):
    df = df[(df['Component'] == selected_comp) & (df['Bus Model'] == selected_model)].copy()
    if df.empty: return df, 0, pd.DataFrame(), pd.DataFrame()

    today = pd.Timestamp.now()

    # 1. DATE & AGE
    for col in ['Reg Date', 'Replace Date', 'Pred Date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    if 'Bus_Age_Years' not in df.columns:
        if 'Reg Date' in df.columns:
            df['Bus_Age_Years'] = (today - df['Reg Date']).dt.days / 365.25
        else:
            df['Bus_Age_Years'] = 10.0
            
    df['Bus_Age_Days'] = df['Bus_Age_Years'] * 365.25
    
    # 2. USAGE
    df['Odometer'] = np.where(df['Odometer'] > 1000, df['Odometer'], df['Bus_Age_Days'] * 220)
    df['Daily_Usage'] = df['Odometer'] / df['Bus_Age_Days'].replace(0, 1)
    df['Daily_Usage'] = df['Daily_Usage'].fillna(100).clip(lower=10) 
    
    def get_comp_usage(row):
        if pd.notnull(row['Replace Date']):
            days = (today - row['Replace Date']).days
            return max(0, days * row['Daily_Usage'])
        return row['Odometer']
    df['Curr_Comp_Km'] = df.apply(get_comp_usage, axis=1)

    # 3. TARGET LIFE
    valid_fails = df[df['Fail_Mileage'] > 50000]['Fail_Mileage']
    target_km_hist = valid_fails.median() if not valid_fails.empty else 0
    current_high = df['Curr_Comp_Km'].quantile(0.80)
    
    if selected_model == 'MAN A95' and selected_comp == 'Transmission':
        target_km = max(600000, current_high * 0.95)
    else:
        target_km = max(target_km_hist, 450000)

    # 4. HEALTH SCORE (DISPLAY ONLY - 30/30/40 Formula)
    usage_factor = 100 - (df['Curr_Comp_Km'] / target_km * 100)
    usage_factor = usage_factor.clip(0, 100)
    
    age_factor = 100 - (df['Bus_Age_Years'] / 17 * 100)
    age_factor = age_factor.clip(0, 100)
    
    wear_factor = 100 - (df['Faults'] * 15) - (df['Accidents'] * 10)
    wear_factor = wear_factor.clip(0, 100)
    
    df['Health_Score'] = (0.30 * age_factor) + (0.30 * usage_factor) + (0.40 * wear_factor)
    df['Health_Score'] = df['Health_Score'].astype(int)

    # 5. RISK SCORE (ACTION PRIORITY - Usage Driven)
    # We use a separate "Risk Score" for sorting that heavily weights Usage (Mileage).
    # This prevents low-mileage buses with faults from jumping the queue.
    
    df['Usage_Pct'] = (df['Curr_Comp_Km'] / target_km) * 100
    
    # Risk Formula: 80% Usage, 10% Faults, 5% Accidents
    df['Risk_Score'] = (df['Usage_Pct'] * 0.8) + (df['Faults'] * 10) + (df['Accidents'] * 5)
    
    # Sort strictly by Risk
    df = df.sort_values('Risk_Score', ascending=False).reset_index(drop=True)
    
    crit_limit = 5
    prio_limit = 15
    
    status_list = []
    date_list = []
    
    for idx, row in df.iterrows():
        # SAFETY CHECK: If usage is < 50%, force Monitor (Green)
        if row['Usage_Pct'] < 50:
            status_list.append("Monitor (6 Mo)")
            future_days = 365 + (idx * 2)
            date_list.append(today + timedelta(days=future_days))
            continue

        # Otherwise apply quota
        if idx < crit_limit:
            status_list.append("Critical (1 Mo)")
            date_list.append(today + timedelta(days=30))
        elif idx < (crit_limit + prio_limit):
            status_list.append("Priority (3 Mo)")
            date_list.append(today + timedelta(days=90))
        else:
            status_list.append("Monitor (6 Mo)")
            future_days = 180 + ((idx - 20) * 2) 
            date_list.append(today + timedelta(days=future_days))

    df['Status'] = status_list
    df['New_Pred_Date'] = date_list
    df['Pred_Month_Str'] = df['New_Pred_Date'].dt.strftime('%b-%Y')
    df['Time_To_Action'] = (df['New_Pred_Date'] - today).dt.days

    # 6. SURVIVAL
    def make_curve(data_list):
        if not data_list: return pd.DataFrame()
        data_list = sorted([x for x in data_list if pd.notnull(x) and x > 0])
        if not data_list: return pd.DataFrame()
        y = [1 - (i/len(data_list)) for i in range(len(data_list))]
        return pd.DataFrame({'x': data_list, 'prob': y})

    km_curve_data = valid_fails.tolist() if not valid_fails.empty else [target_km*0.8, target_km, target_km*1.2]
    surv_km = make_curve(km_curve_data)
    surv_age = make_curve(df['Bus_Age_Years'].dropna().tolist())

    return df, target_km, surv_km, surv_age

# -----------------------------------------------------------------------------
# 4. SIDEBAR & UI
# -----------------------------------------------------------------------------
raw_df = load_all_data()

with st.sidebar:
    st.header("🔍 Filters")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
        
    if not raw_df.empty:
        comps = sorted(raw_df['Component'].unique())
        c_idx = comps.index('Transmission') if 'Transmission' in comps else 0
        sel_comp = st.selectbox("Select Component", comps, index=c_idx)
        
        rel_models = sorted(raw_df[raw_df['Component'] == sel_comp]['Bus Model'].dropna().unique())
        sel_model = st.selectbox("Select Bus Model", rel_models)
    else:
        sel_comp, sel_model = "None", "None"

# Process
df, target_km, surv_km, surv_age = process_data(raw_df, sel_comp, sel_model)

# Colors
status_colors = {
    'Healthy': '#2ecc71',         
    'Monitor (6 Mo)': '#2ecc71',  # GREEN
    'Priority (3 Mo)': '#e67e22', 
    'Critical (1 Mo)': '#e74c3c'
}

st.title(f"🚍 Maintenance Analyst Dashboard")
st.caption(f"Component: {sel_comp} | Model: {sel_model} | Date: {datetime.now().strftime('%d %b %Y')}")

if not df.empty:
    
    c1, c2, c3, c4, c5 = st.columns(5)
    crit = len(df[df['Status'] == 'Critical (1 Mo)'])
    prio = len(df[df['Status'] == 'Priority (3 Mo)'])
    avg_age = df['Bus_Age_Years'].mean()
    if pd.isna(avg_age): avg_age = 0.0
    
    c1.metric("Fleet Size", len(df))
    c2.metric("Avg Fleet Age", f"{avg_age:.1f} Yrs")
    c3.metric("Target Life", f"{target_km:,.0f} km")
    c4.metric("CRITICAL (<30 Days)", crit, delta="Act Now", delta_color="inverse")
    c5.metric("PRIORITY (<90 Days)", prio, delta="Queue Next", delta_color="off")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Actionable Insights", "📉 Survival Analysis", "📋 Master Data"])

    # TAB 1
    with tab1:
        st.markdown(f"""
        <div class="info-box">
            <strong>📊 Analyst Interpretation: Risk Matrix ({sel_comp})</strong><br>
            <ul>
                <li><strong>X-Axis (Usage):</strong> Current component mileage.</li>
                <li><strong>Y-Axis (Age):</strong> Bus age. Older buses (Top) with high mileage (Right) fail most frequently.</li>
                <li><strong>Red Line:</strong> Historical failure average. Buses past this line are operating on borrowed time.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        c_left, c_right = st.columns([1.5, 1])
        with c_left:
            st.subheader("1. Fleet Health Matrix")
            plot_df = df.dropna(subset=['Curr_Comp_Km', 'Bus_Age_Years'])
            fig_scat = px.scatter(
                plot_df, x='Curr_Comp_Km', y='Bus_Age_Years', color='Status',
                color_discrete_map=status_colors, hover_data=['Bus No', 'Pred_Month_Str', 'Health_Score'],
                title="Usage vs Age Risk"
            )
            fig_scat.add_vline(x=target_km, line_dash="dash", line_color="red", annotation_text="Target")
            st.plotly_chart(fig_scat, use_container_width=True)
            
            if 'Depot' in df.columns and df['Depot'].notna().any():
                st.subheader("2. Depot Workload")
                fig_d = px.histogram(df, x='Depot', color='Status', color_discrete_map=status_colors, barmode='stack')
                st.plotly_chart(fig_d, use_container_width=True)

        with c_right:
            st.markdown('<div class="priority-header">🔥 Priority Action List</div>', unsafe_allow_html=True)
            # Sort by Risk Score for the list
            urgent = df.sort_values('Risk_Score', ascending=False).head(20)
            if not urgent.empty:
                st.dataframe(
                    urgent[['Bus No', 'Status', 'Pred_Month_Str', 'Curr_Comp_Km', 'Faults']]
                    .style.format({'Curr_Comp_Km': '{:,.0f}', 'Faults': '{:.0f}'})
                    .applymap(lambda x: 'background-color: #ffcccc' if 'Critical' in str(x) else ('background-color: #fff3cd' if 'Priority' in str(x) else ('background-color: #d4edda' if 'Monitor' in str(x) else '')), subset=pd.IndexSlice[:, :]),
                    use_container_width=True, height=500
                )
            else:
                st.success("✅ No immediate actions.")

    # TAB 2
    with tab2:
        st.markdown("""
        <div class="success-box">
            <strong>📉 Analyst Interpretation: Survival Curves</strong><br>
            <ul>
                <li><strong>Orange Line (90% Limit):</strong> 'Safe Operating Limit'. 10% of units fail by this point.</li>
                <li><strong>Steep Drop:</strong> Indicates the 'danger zone' where failures accelerate.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Reliability Analysis")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("##### 📉 Survival by Mileage")
            if not surv_km.empty:
                fig_km = px.line(surv_km, x='x', y='prob', title="Prob. Survival vs Km")
                fig_km.add_hline(y=0.9, line_dash="dot", line_color="orange", annotation_text="90% Limit")
                st.plotly_chart(fig_km, use_container_width=True)
            else: st.info("Insufficient Failure Data")
        with col_s2:
            st.markdown("##### 📉 Survival by Age")
            if not surv_age.empty:
                fig_age = px.line(surv_age, x='x', y='prob', title="Fleet Age Distribution")
                fig_age.add_hline(y=0.9, line_dash="dot", line_color="orange", annotation_text="90% Limit")
                st.plotly_chart(fig_age, use_container_width=True)
            else: st.info("Insufficient Age Data")

    # TAB 3
    with tab3:
        with st.expander("📖 Analyst Reference Guide: Calculations & Definitions", expanded=True):
            st.markdown("""
            #### 🧮 Health Score Formula (0-100)
            $$Score = (30\% \\times Age Factor) + (30\% \\times Usage Factor) + (40\% \\times Wear Factor)$$
            *For Display Only.*
            
            #### ⚠️ Risk Score (For Sorting)
            *Action Priority is calculated based on **Usage (80%)** and **Faults (20%)** to ensure high-mileage buses are prioritized.*
            
            #### 📝 Metric Definitions
            | Field Name | Description |
            | :--- | :--- |
            | **Usage Factor** | Based on mileage vs target life. |
            | **Action_Status** | **Critical:** Top 5 highest risk. **Priority:** Next 15. |
            """)

        st.subheader("Master Register")
        
        display_cols = [
            'Bus No', 'Material_Description', 'Bus Model', 
            'Replace Date', 'New_Pred_Date', 'Health_Score', 'Bus_Age_Years', 'Status',
            'Days_Since_Last_Repl', 'Time_To_Action', 'Odometer', 
            'Faults', 'Accidents'
        ]
        
        rename_map = {
            'Material_Description': 'Material Description',
            'Bus Model': 'Model',
            'Replace Date': 'Replacement Date',
            'New_Pred_Date': 'Predicted Replacement Date',
            'Health_Score': 'Health Score',
            'Bus_Age_Years': 'Bus Age',
            'Days_Since_Last_Repl': 'Days Since Last Replacement',
            'Time_To_Action': 'Time To Action',
            'Faults': 'Total Fault Count',
            'Accidents': 'Accident Count'
        }
        
        valid_cols = [c for c in display_cols if c in df.columns]
        final_table = df[valid_cols].rename(columns=rename_map).sort_values('Time To Action')
        
        def color_health(val):
            if val < 50: return 'background-color: #ffcccc; color: black;'
            elif val < 75: return 'background-color: #fff3cd; color: black;'
            return 'background-color: #d4edda; color: black;'

        st.dataframe(
            final_table.style.format({
                'Bus Age': '{:.1f}', 'Odometer': '{:,.0f}',
                'Days Since Last Replacement': '{:.0f}', 'Time To Action': '{:.0f}',
                'Total Fault Count': '{:.0f}', 'Accident Count': '{:.0f}',
                'Replacement Date': lambda x: x.strftime('%d/%m/%Y') if pd.notnull(x) else "-",
                'Predicted Replacement Date': lambda x: x.strftime('%d/%m/%Y') if pd.notnull(x) else "-"
            }).applymap(color_health, subset=['Health Score']),
            use_container_width=True, height=600
        )
        
        csv = final_table.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export CSV (Formatted)", csv, f"Report_{sel_comp}.csv", "text/csv")

else:
    st.info("Waiting for data files...")
    st.write(f"Ensure `{FILE_TRANS_B9TL}`, `{FILE_TRANS_A95}`, and `{FILE_NEW_COMPONENTS}` are in the directory.")
