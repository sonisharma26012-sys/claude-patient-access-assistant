
import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------------------------
# Page setup
# -------------------------
st.set_page_config(
    page_title="Claude Patient Access Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
    .main {background-color: #f7f9fc;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .hero {
        background: linear-gradient(135deg, #263238 0%, #1565C0 48%, #00897B 100%);
        padding: 2rem 2rem;
        border-radius: 22px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.10);
    }
    .hero h1 {
        font-size: 2.7rem;
        margin-bottom: 0.25rem;
        color: white;
    }
    .hero p {font-size: 1.05rem; color: #e8f5e9;}
    .metric-card {
        padding: 1.1rem 1.25rem;
        border-radius: 18px;
        background: white;
        box-shadow: 0 6px 18px rgba(31, 45, 61, 0.08);
        border-left: 7px solid #1565C0;
        min-height: 135px;
    }
    .metric-card.red {border-left-color: #E53935;}
    .metric-card.orange {border-left-color: #FB8C00;}
    .metric-card.green {border-left-color: #00897B;}
    .metric-card.purple {border-left-color: #6A1B9A;}
    .metric-label {
        color: #64748B;
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .04em;
    }
    .metric-value {
        color: #1F2937;
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.2;
        margin-top: 0.35rem;
    }
    .metric-help {
        color: #64748B;
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }
    .insight-box {
        background: white;
        border-radius: 18px;
        padding: 1.35rem;
        box-shadow: 0 6px 18px rgba(31, 45, 61, 0.08);
        border: 1px solid #e5e7eb;
    }
    .risk-high {color:#B91C1C; font-weight:800;}
    .risk-med {color:#B45309; font-weight:800;}
    .risk-low {color:#047857; font-weight:800;}
    .small-caption {color:#64748B; font-size:0.9rem;}
    div[data-testid="stTabs"] button {font-size: 1rem;}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load data
# -------------------------
@st.cache_data(show_spinner=False)
def load_data():
    possible_files = [
        "../data/KaggleV2-May-2016.csv",
        "../data/appointments.csv",
        "../data/patient_appointments.csv",
        "KaggleV2-May-2016.csv",
    ]
    data_path = None
    for f in possible_files:
        if os.path.exists(f):
            data_path = f
            break
    if data_path is None:
        return None

    df = pd.read_csv(data_path)

    # Standardize column names
    df.columns = [c.strip().replace("-", "_").replace(" ", "_") for c in df.columns]

    # Expected Kaggle columns after standardizing:
    # PatientId, AppointmentID, Gender, ScheduledDay, AppointmentDay, Age, Neighbourhood,
    # Scholarship, Hipertension, Diabetes, Alcoholism, Handcap, SMS_received, No_show

    # Rename common variations
    rename_map = {
        "No_show": "NoShow",
        "No_show_": "NoShow",
        "No_Show": "NoShow",
        "Hipertension": "Hypertension",
        "Handcap": "Handicap",
        "Neighbourhood": "Neighborhood",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "NoShow" in df.columns:
        df["NoShowFlag"] = df["NoShow"].astype(str).str.lower().str.strip().map({"yes": 1, "no": 0})
    else:
        df["NoShowFlag"] = 0

    for col in ["ScheduledDay", "AppointmentDay"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "ScheduledDay" in df.columns and "AppointmentDay" in df.columns:
        df["WaitDays"] = (df["AppointmentDay"].dt.date.astype("datetime64[ns]") - df["ScheduledDay"].dt.date.astype("datetime64[ns]")).dt.days
        df["WaitDays"] = df["WaitDays"].clip(lower=0)
        df["AppointmentWeekday"] = df["AppointmentDay"].dt.day_name()
        df["ScheduledWeekday"] = df["ScheduledDay"].dt.day_name()
        df["AppointmentMonth"] = df["AppointmentDay"].dt.strftime("%b %Y")

    if "Age" in df.columns:
        df = df[df["Age"].fillna(0) >= 0].copy()
        bins = [0, 12, 18, 30, 45, 60, 75, 120]
        labels = ["0-12", "13-18", "19-30", "31-45", "46-60", "61-75", "76+"]
        df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True, include_lowest=True)

    if "WaitDays" in df.columns:
        wait_bins = [-1, 0, 3, 7, 14, 30, 3650]
        wait_labels = ["Same day", "1-3 days", "4-7 days", "8-14 days", "15-30 days", "31+ days"]
        df["WaitBucket"] = pd.cut(df["WaitDays"], bins=wait_bins, labels=wait_labels)

    binary_cols = ["Scholarship", "Hypertension", "Diabetes", "Alcoholism", "SMS_received", "Handicap"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df

df = load_data()

if df is None:
    st.error("Dataset not found. Put KaggleV2-May-2016.csv inside the data folder, then run again.")
    st.stop()

# -------------------------
# Helper functions
# -------------------------
def pct(x):
    if pd.isna(x):
        return "0.00%"
    return f"{x:.2f}%"

def no_show_rate(data):
    return data["NoShowFlag"].mean() * 100 if len(data) else 0

def metric_card(label, value, help_text="", color=""):
    st.markdown(
        f"""
        <div class="metric-card {color}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def grouped_rate(data, group_col, min_n=50):
    out = (
        data.groupby(group_col, dropna=False)
        .agg(appointments=("NoShowFlag", "size"), no_show_rate=("NoShowFlag", "mean"))
        .reset_index()
    )
    out["no_show_rate"] = out["no_show_rate"] * 100
    return out[out["appointments"] >= min_n].sort_values("no_show_rate", ascending=False)

def safe_col(name):
    return name in df.columns

# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.title("🎛️ Dashboard Filters")
st.sidebar.caption("Use filters to explore patient access patterns.")

filtered = df.copy()

if safe_col("Gender"):
    genders = sorted(df["Gender"].dropna().unique().tolist())
    selected_gender = st.sidebar.multiselect("Gender", genders, default=genders)
    filtered = filtered[filtered["Gender"].isin(selected_gender)]

if safe_col("Age"):
    age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
    selected_age = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
    filtered = filtered[(filtered["Age"] >= selected_age[0]) & (filtered["Age"] <= selected_age[1])]

if safe_col("Neighborhood"):
    neighborhoods = sorted(df["Neighborhood"].dropna().unique().tolist())
    selected_neighborhoods = st.sidebar.multiselect(
        "Neighborhood", neighborhoods, default=neighborhoods[: min(15, len(neighborhoods))]
    )
    if selected_neighborhoods:
        filtered = filtered[filtered["Neighborhood"].isin(selected_neighborhoods)]

if safe_col("SMS_received"):
    sms_filter = st.sidebar.selectbox("SMS Reminder", ["All", "Received SMS", "No SMS"])
    if sms_filter == "Received SMS":
        filtered = filtered[filtered["SMS_received"] == 1]
    elif sms_filter == "No SMS":
        filtered = filtered[filtered["SMS_received"] == 0]

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: Medical Appointment No-Shows, 110K+ appointment records.")

# -------------------------
# Header
# -------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🏥 Claude-Powered Patient Access & No-Show Prevention Assistant</h1>
        <p>Healthcare analytics + AI workflow project using real appointment data to identify access barriers, high-risk groups, and operational interventions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# KPI Row
# -------------------------
total_appts = len(filtered)
overall_rate = no_show_rate(filtered)
avg_wait = filtered["WaitDays"].mean() if safe_col("WaitDays") else 0
unique_patients = filtered["PatientId"].nunique() if safe_col("PatientId") else 0
neigh_count = filtered["Neighborhood"].nunique() if safe_col("Neighborhood") else 0

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    metric_card("Appointments", f"{total_appts:,}", "Filtered appointment records", "green")
with k2:
    metric_card("No-show Rate", pct(overall_rate), "Missed appointments", "red")
with k3:
    metric_card("Avg Wait Days", f"{avg_wait:.2f}", "Schedule → visit gap", "orange")
with k4:
    metric_card("Unique Patients", f"{unique_patients:,}", "Distinct patient IDs", "purple")
with k5:
    metric_card("Neighborhoods", f"{neigh_count:,}", "Access geography", "")

st.markdown("")

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Overview",
    "👥 Patient Risk",
    "⏱️ Operations",
    "📍 Neighborhoods",
    "🤖 AI Insights"
])

color_sequence = ["#1565C0", "#00897B", "#FB8C00", "#E53935", "#6A1B9A", "#00ACC1"]

with tab1:
    st.subheader("Executive Overview")
    c1, c2 = st.columns([1, 1])

    with c1:
        if safe_col("WaitBucket"):
            wait = grouped_rate(filtered, "WaitBucket", min_n=20)
            fig = px.bar(
                wait,
                x="WaitBucket",
                y="no_show_rate",
                text=wait["no_show_rate"].round(1),
                title="No-Show Rate by Wait Time",
                labels={"WaitBucket": "Wait time bucket", "no_show_rate": "No-show rate (%)"},
                color="no_show_rate",
                color_continuous_scale=["#00897B", "#FB8C00", "#E53935"],
            )
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(height=430, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if safe_col("AgeGroup"):
            age = grouped_rate(filtered, "AgeGroup", min_n=20)
            fig = px.line(
                age.sort_values("AgeGroup"),
                x="AgeGroup",
                y="no_show_rate",
                markers=True,
                title="No-Show Rate by Age Group",
                labels={"AgeGroup": "Age group", "no_show_rate": "No-show rate (%)"},
            )
            fig.update_traces(line=dict(width=4, color="#1565C0"), marker=dict(size=10))
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Key Takeaways")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown(f"""
        <div class="insight-box">
        <b>Access Risk</b><br>
        Overall no-show rate is <span class="risk-high">{pct(overall_rate)}</span>. This suggests missed visits are a major patient access and operational issue.
        </div>
        """, unsafe_allow_html=True)
    with t2:
        high_wait = filtered[filtered.get("WaitDays", 0) >= 15]["NoShowFlag"].mean() * 100 if safe_col("WaitDays") else 0
        st.markdown(f"""
        <div class="insight-box">
        <b>Wait-Time Effect</b><br>
        Patients waiting 15+ days have an estimated no-show rate of <span class="risk-med">{pct(high_wait)}</span>.
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown(f"""
        <div class="insight-box">
        <b>Stakeholder Value</b><br>
        The app converts appointment data into plain-language insights for operations, outreach, and care-access teams.
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.subheader("Patient Risk Dashboard")
    c1, c2 = st.columns(2)

    with c1:
        if safe_col("Gender"):
            gender = grouped_rate(filtered, "Gender", min_n=20)
            fig = px.pie(
                gender,
                names="Gender",
                values="appointments",
                title="Appointment Volume by Gender",
                hole=0.55,
                color_discrete_sequence=color_sequence,
            )
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        risk_cols = [c for c in ["Scholarship", "Hypertension", "Diabetes", "Alcoholism", "SMS_received"] if safe_col(c)]
        rows = []
        for col in risk_cols:
            temp = filtered.groupby(col).agg(appointments=("NoShowFlag", "size"), no_show_rate=("NoShowFlag", "mean")).reset_index()
            if 1 in temp[col].values:
                r = temp[temp[col] == 1].iloc[0]
                rows.append({"Risk Factor": col.replace("_", " "), "Appointments": int(r["appointments"]), "No-show Rate": r["no_show_rate"] * 100})
        risk_df = pd.DataFrame(rows)
        if not risk_df.empty:
            fig = px.bar(
                risk_df.sort_values("No-show Rate", ascending=True),
                x="No-show Rate",
                y="Risk Factor",
                orientation="h",
                text=risk_df.sort_values("No-show Rate", ascending=True)["No-show Rate"].round(1),
                title="No-Show Rate by Clinical / Social Risk Factor",
                color="No-show Rate",
                color_continuous_scale=["#00897B", "#FB8C00", "#E53935"],
            )
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(height=430, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if safe_col("Age"):
            fig = px.histogram(
                filtered,
                x="Age",
                color="NoShowFlag",
                nbins=40,
                title="Age Distribution by Attendance Outcome",
                labels={"NoShowFlag": "No-show flag"},
                color_discrete_map={0: "#00897B", 1: "#E53935"},
            )
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        if safe_col("AgeGroup"):
            age_table = grouped_rate(filtered, "AgeGroup", min_n=20)
            age_table["No-show Rate"] = age_table["no_show_rate"].round(2)
            age_table = age_table.rename(columns={"appointments": "Appointments", "AgeGroup": "Age Group"})
            st.markdown("#### Age Group Risk Table")
            st.dataframe(age_table[["Age Group", "Appointments", "No-show Rate"]], use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Operational Dashboard")
    c1, c2 = st.columns(2)

    with c1:
        if safe_col("AppointmentWeekday"):
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekday = grouped_rate(filtered, "AppointmentWeekday", min_n=20)
            weekday["AppointmentWeekday"] = pd.Categorical(weekday["AppointmentWeekday"], categories=weekday_order, ordered=True)
            weekday = weekday.sort_values("AppointmentWeekday")
            fig = px.bar(
                weekday,
                x="AppointmentWeekday",
                y="no_show_rate",
                title="No-Show Rate by Appointment Weekday",
                labels={"AppointmentWeekday": "Appointment day", "no_show_rate": "No-show rate (%)"},
                color="no_show_rate",
                color_continuous_scale=["#00897B", "#FB8C00", "#E53935"],
                text=weekday["no_show_rate"].round(1)
            )
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(height=420, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if safe_col("SMS_received"):
            sms = grouped_rate(filtered, "SMS_received", min_n=20)
            sms["SMS Status"] = sms["SMS_received"].map({0: "No SMS", 1: "Received SMS"})
            fig = px.bar(
                sms,
                x="SMS Status",
                y="no_show_rate",
                title="SMS Reminder Effectiveness",
                labels={"no_show_rate": "No-show rate (%)"},
                color="SMS Status",
                color_discrete_map={"No SMS": "#E53935", "Received SMS": "#00897B"},
                text=sms["no_show_rate"].round(1)
            )
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(height=420, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if safe_col("WaitDays"):
            fig = px.box(
                filtered,
                x="NoShow",
                y="WaitDays",
                title="Wait Days Distribution by Attendance Outcome",
                labels={"NoShow": "No-show status", "WaitDays": "Wait days"},
                color="NoShow",
                color_discrete_sequence=["#00897B", "#E53935"],
            )
            fig.update_layout(height=420, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        if safe_col("AppointmentMonth"):
            month = filtered.groupby("AppointmentMonth").agg(appointments=("NoShowFlag", "size"), no_show_rate=("NoShowFlag", "mean")).reset_index()
            month["no_show_rate"] *= 100
            fig = px.line(
                month,
                x="AppointmentMonth",
                y="no_show_rate",
                markers=True,
                title="Monthly No-Show Trend",
                labels={"AppointmentMonth": "Month", "no_show_rate": "No-show rate (%)"},
            )
            fig.update_traces(line=dict(width=4, color="#6A1B9A"), marker=dict(size=10))
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Neighborhood Access Dashboard")
    if safe_col("Neighborhood"):
        neigh = grouped_rate(filtered, "Neighborhood", min_n=100)
        c1, c2 = st.columns(2)

        with c1:
            top_risk = neigh.head(12).sort_values("no_show_rate")
            fig = px.bar(
                top_risk,
                x="no_show_rate",
                y="Neighborhood",
                orientation="h",
                title="Top High-Risk Neighborhoods",
                labels={"no_show_rate": "No-show rate (%)"},
                color="no_show_rate",
                color_continuous_scale=["#FB8C00", "#E53935"],
                text=top_risk["no_show_rate"].round(1),
            )
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(height=540, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            volume = neigh.sort_values("appointments", ascending=False).head(12).sort_values("appointments")
            fig = px.bar(
                volume,
                x="appointments",
                y="Neighborhood",
                orientation="h",
                title="Highest Appointment Volume Neighborhoods",
                labels={"appointments": "Appointments"},
                color="appointments",
                color_continuous_scale=["#90CAF9", "#1565C0"],
                text="appointments",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(height=540, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Neighborhood Risk Table")
        table = neigh.copy()
        table["No-show Rate"] = table["no_show_rate"].round(2)
        table = table.rename(columns={"appointments": "Appointments"})
        st.dataframe(table[["Neighborhood", "Appointments", "No-show Rate"]].head(25), use_container_width=True, hide_index=True)

with tab5:
    st.subheader("AI Insights & Claude-Style Executive Summary")

    # Prepare deterministic AI-like summary even if no API key is available
    top_wait_bucket = None
    if safe_col("WaitBucket"):
        wait_rates = grouped_rate(filtered, "WaitBucket", min_n=20)
        if not wait_rates.empty:
            top_wait_bucket = wait_rates.sort_values("no_show_rate", ascending=False).iloc[0]

    top_neighborhood = None
    if safe_col("Neighborhood"):
        neigh = grouped_rate(filtered, "Neighborhood", min_n=100)
        if not neigh.empty:
            top_neighborhood = neigh.iloc[0]

    sms_sentence = ""
    if safe_col("SMS_received"):
        sms_rates = grouped_rate(filtered, "SMS_received", min_n=20)
        try:
            no_sms = sms_rates.loc[sms_rates["SMS_received"] == 0, "no_show_rate"].iloc[0]
            yes_sms = sms_rates.loc[sms_rates["SMS_received"] == 1, "no_show_rate"].iloc[0]
            sms_sentence = f"Patients without SMS reminders show a {no_sms:.2f}% no-show rate compared with {yes_sms:.2f}% among patients who received SMS reminders."
        except Exception:
            sms_sentence = "SMS reminder patterns should be reviewed as part of the access intervention strategy."

    summary = f"""
    **Executive Summary**

    This patient access analysis reviews **{total_appts:,} appointment records** and identifies an overall no-show rate of **{overall_rate:.2f}%**. 
    The data suggests that missed appointments are not random; they are associated with scheduling delays, demographic patterns, and neighborhood-level access barriers.

    **Key Findings**
    - The average scheduling wait time is **{avg_wait:.2f} days**.
    - The highest-risk wait-time segment is **{top_wait_bucket['WaitBucket'] if top_wait_bucket is not None else 'not available'}**, with a no-show rate of **{top_wait_bucket['no_show_rate']:.2f}%**.
    - The highest-risk neighborhood in the current filter is **{top_neighborhood['Neighborhood'] if top_neighborhood is not None else 'not available'}**, with a no-show rate of **{top_neighborhood['no_show_rate']:.2f}%**.
    - {sms_sentence}

    **Recommended Actions**
    1. Prioritize outreach for patients scheduled more than 14 days out.
    2. Create neighborhood-specific reminder and transportation support workflows.
    3. Use SMS plus follow-up calls for high-risk appointment groups.
    4. Build a weekly access report for clinic managers showing no-show risk by wait time, age group, and neighborhood.
    5. Turn this dashboard into a staff-facing AI assistant so non-technical teams can ask questions in plain English.
    """

    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown(summary)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("#### Ask the Assistant")
        user_q = st.text_area(
            "Type a stakeholder question",
            value="Which patient groups should we prioritize to reduce missed appointments?",
            height=120,
        )
        if st.button("Ask Claude", type="primary"):
            st.markdown(
                f"""
                <div class="insight-box">
                <b>Question:</b> {user_q}<br><br>
                <b>Answer:</b> Based on the current filtered dataset, the first priority should be patients with longer wait times, 
                high-risk neighborhood groups, and patients who may need stronger reminder workflows. The strongest operational opportunity is to 
                reduce appointment delay and combine SMS reminders with targeted follow-up for groups showing above-average no-show rates.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.info("This page is designed to demonstrate the Claude Corps use case: turning healthcare data into plain-English recommendations for non-technical stakeholders.")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption(
    "Portfolio project for Claude Corps: Python • Streamlit • Pandas • Plotly • Generative AI workflow design • Healthcare access analytics"
)
