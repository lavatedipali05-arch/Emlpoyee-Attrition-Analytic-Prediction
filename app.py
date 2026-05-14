import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns # Added seaborn import
import os # Import the os module

# --- Custom CSS Styling ---
st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #0E1117;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161B22;
}

/* Buttons */
.stButton>button {
    background-color: #00BFFF;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #0099cc;
    color: white;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background-color: #161B22;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #00BFFF;
}

/* Input Boxes */
.stTextInput>div>div>input {
    background-color: #262730;
    color: white;
}

/* Selectbox */
.stSelectbox>div>div {
    background-color: #262730;
    color: white;
}

/* Titles */
h1, h2, h3 {
    color: #00BFFF;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: #161B22;
}

/* Chart background */
.plot-container {
    background-color: #161B22;
    border-radius: 10px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)


# Load pipeline + dataset
pipeline = joblib.load("pipeline.pkl")
# Use os.path.join to construct the path relative to the script's directory
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "employee Attrition CSV file 1.csv"))

st.set_page_config(page_title="HR Analytics Dashboard", layout="wide")

st.title("🚀 HR Analytics & Attrition Prediction Dashboard")

# ---- TABS ----
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 Prediction", "📈 Insights"])

# ------------------- 📊 DASHBOARD -------------------
with tab1:
    st.subheader("Attrition Overview")

    attrition_counts = df['Attrition'].value_counts()

    fig, ax = plt.subplots()
    ax.bar(attrition_counts.index, attrition_counts.values)
    ax.set_title("Attrition Distribution")

    st.pyplot(fig)

    st.subheader("Department Wise Attrition")
    dept_attrition = df.groupby("Department")["Attrition"].value_counts().unstack()

    st.bar_chart(dept_attrition)

    st.subheader("Job Satisfaction Distribution")
    job_satisfaction_counts = df['JobSatisfaction'].value_counts().sort_index()
    fig_job_sat, ax_job_sat = plt.subplots()
    sns.barplot(x=job_satisfaction_counts.index, y=job_satisfaction_counts.values, ax=ax_job_sat, palette='viridis')
    ax_job_sat.set_title("Job Satisfaction Distribution")
    ax_job_sat.set_xlabel("Job Satisfaction Level (1=Low, 4=High)")
    ax_job_sat.set_ylabel("Number of Employees")
    st.pyplot(fig_job_sat)

# ------------------- 🔮 PREDICTION -------------------
with tab2:
    st.subheader("Employee Prediction")

    sample_data = df.drop("Attrition", axis=1).iloc[0].to_dict()
    input_data = {}

    for col, val in sample_data.items():
        if isinstance(val, (int, float)):
            input_data[col] = st.number_input(col, value=float(val))
        else:
            input_data[col] = st.selectbox(col, df[col].unique())

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])

        prediction = pipeline.predict(input_df)[0]
        prob = pipeline.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ High Risk (Leave) — Probability: {prob:.2f}")
        else:
            st.success(f"✅ Low Risk (Stay) — Probability: {prob:.2f}")

# ------------------- 📈 INSIGHTS -------------------
with tab3:
    st.subheader("Feature Importance")

    try:
        model = pipeline.named_steps['model']
        preprocessor = pipeline.named_steps['preprocessor']
        importances = model.feature_importances_

        # Get feature names after preprocessing
        transformed_feature_names = preprocessor.get_feature_names_out()
        feature_importances_df = pd.DataFrame({
            'Feature': transformed_feature_names,
            'Importance': importances
        })

        # Sort by importance and get top N
        feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
        top_n = 20 # Display top 20 features for clarity

        fig2, ax2 = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importances_df.head(top_n), ax=ax2, palette='viridis')
        ax2.set_title(f"Top {top_n} Feature Importances (from Encoded Features)")
        ax2.set_xlabel("Importance")
        ax2.set_ylabel("Feature Name")
        st.pyplot(fig2)
        # Existing charts
st.subheader("Attrition Distribution")
st.bar_chart(df["Attrition"].value_counts())

# Heatmap section add here
st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(include=['int64', 'float64'])

corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    ax=ax
)

st.pyplot(fig)

# Next charts below

    except Exception as e:
        st.warning(f)
