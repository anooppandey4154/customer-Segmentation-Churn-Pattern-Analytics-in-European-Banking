# app.py - Copy and paste this directly into your GitHub file
import streamlit as st
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(
    page_title="European Banking Churn Analytics",
    page_icon="📊",
    layout="wide"
)

# Header Title
st.title("📊 Customer Churn Pattern Analytics in European Banking")
st.markdown("Interactive live dashboard mapping key segment risks, geographic indices, and high-value customer capital protection metrics.")

# 1. Load Data
@st.cache_data
def load_data():
    try:
        # Load from GitHub repository file
        df = pd.read_csv("churn_data.csv")
        return df
    except Exception as e:
        st.warning("Database file 'churn_data.csv' not found. Generating exact 10,000-sample European Banking cohort.")
        n_samples = 10000
        np.random.seed(42)
        geographies = np.random.choice(['France', 'Germany', 'Spain'], size=n_samples, p=[0.5, 0.25, 0.25])
        genders = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.54, 0.46])
        ages = np.clip(np.random.normal(38.9, 10.5, n_samples).astype(int), 18, 92)
        credit_scores = np.clip(np.random.normal(650, 96, n_samples).astype(int), 350, 850)
        tenures = np.random.randint(0, 11, size=n_samples)
        balances = np.where(np.random.rand(n_samples) > 0.36, np.random.normal(119800, 30000, n_samples), 0.0)
        num_products = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.50, 0.46, 0.03, 0.01])
        is_active_member = np.random.choice([0, 1], size=n_samples, p=[0.48, 0.52])
        estimated_salaries = np.random.uniform(15000, 200000, n_samples)

        log_odds = -2.0 + (geographies == 'Germany') * 1.0 + (ages >= 46) * 1.3 + (ages >= 60) * 0.2 + (num_products >= 3) * 2.5 - is_active_member * 0.9 - (credit_scores > 700) * 0.1
        probs = 1 / (1 + np.exp(-log_odds))
        exited = (np.random.rand(n_samples) < probs).astype(int)

        df = pd.DataFrame({
            'CustomerId': 15600000 + np.arange(n_samples),
            'Surname': [f"Client_{x}" for x in range(n_samples)],
            'Geography': geographies,
            'Gender': genders,
            'Age': ages,
            'CreditScore': credit_scores,
            'Tenure': tenures,
            'Balance': np.round(balances, 2),
            'NumOfProducts': num_products,
            'HasCrCard': np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),
            'IsActiveMember': is_active_member,
            'EstimatedSalary': np.round(estimated_salaries, 2),
            'Exited': exited
        })
        return df

df = load_data()

# 2. Left Sidebar Filters
st.sidebar.header("🎛️ Analytics Controls")
selected_geo = st.sidebar.selectbox("Geography Region", ["All"] + list(df["Geography"].unique()))
selected_gender = st.sidebar.radio("Gender Select", ["All", "Male", "Female"])

min_tenure = st.sidebar.slider("Min Tenure Years", 0, 10, 0)
min_balance = st.sidebar.number_input("Min Account Balance (€)", value=0, min_value=0)

# Filter trigger
filtered_df = df.copy()
if selected_geo != "All":
    filtered_df = filtered_df[filtered_df["Geography"] == selected_geo]
if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]
filtered_df = filtered_df[filtered_df["Tenure"] >= min_tenure]
filtered_df = filtered_df[filtered_df["Balance"] >= min_balance]

# 3. KPI Metrics Columns
col1, col2, col3, col4 = st.columns(4)

total_clients = len(filtered_df)
exited_clients = len(filtered_df[filtered_df["Exited"] == 1])
churn_rate = (exited_clients / total_clients * 100) if total_clients > 0 else 0

# High-value customer metrics (Balance > 100k)
hv_df = filtered_df[filtered_df["Balance"] > 100000]
hv_total = len(hv_df)
hv_exited = len(hv_df[hv_df["Exited"] == 1])
hv_rate = (hv_exited / hv_total * 100) if hv_total > 0 else 0

# Total capital at risk
revenue_risk = filtered_df[filtered_df["Exited"] == 1]["Balance"].sum()

with col1:
    st.metric("Overall Churn Rate", f"{churn_
