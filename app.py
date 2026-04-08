import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import io

# --- Configuration & Styling ---
st.set_page_config(page_title="VoteIntel Dashboard", page_icon="🗳️", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: black; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv('final_voteintel_dataset.csv')
    # Preprocessing
    df['CRIMINAL_CASES'] = pd.to_numeric(df['CRIMINAL CASES'], errors='coerce').fillna(0)
    df['EDUCATION'] = df['EDUCATION'].str.replace('\n', '').fillna('Not Available')
    df['GENDER'] = df['GENDER'].str.strip().str.upper()
    return df

df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("🗳️ VoteIntel Hub")
st.sidebar.info("Analyze election trends and predict candidate success.")
page = st.sidebar.radio("Navigation", ["🏠 Home", "📊 Explore Data", "🔮 Prediction", "🔍 Bulk Scanner"])

# --- PAGE: HOME ---
if page == "🏠 Home":
    st.title("Election Insights Dashboard")
    st.write("Explore candidate statistics, education levels, assets, and election outcomes from the VoteIntel dataset.")
    
    # Key Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Candidates", f"{len(df):,}")
    m2.metric("Total States", f"{df['STATE'].nunique()}")
    m3.metric("Total Constituencies", f"{df['CONSTITUENCY'].nunique()}")
    m4.metric("Avg. Assets (₹)", f"{(df['ASSETS'].mean()/1e7):.1f} Cr")

    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Distribution of Candidates by State")
    state_counts = df['STATE'].value_counts().reset_index()
    fig_state = px.bar(state_counts, x='STATE', y='count', color='count', labels={'count': 'Candidates'})
    st.plotly_chart(fig_state, use_container_width=True)

# --- PAGE: EXPLORE ---
elif page == "📊 Explore Data":
    st.title("Interactive Data Exploration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Winners by Party")
        winners_df = df[df['WINNER'] == 1]['PARTY'].value_counts().head(10).reset_index()
        fig_party = px.pie(winners_df, values='count', names='PARTY', hole=0.4, title="Top 10 Winning Parties")
        st.plotly_chart(fig_party)

    with col2:
        st.subheader("Education vs Assets")
        fig_scatter = px.scatter(df, x="AGE", y="ASSETS", color="WINNER", 
                                 hover_data=['NAME', 'PARTY'], 
                                 title="Age vs Assets (Color: Winner Status)")
        st.plotly_chart(fig_scatter)

    st.subheader("Education Level Breakdown")
    edu_count = df['EDUCATION'].value_counts().reset_index()
    fig_edu = px.bar(edu_count, x='count', y='EDUCATION', orientation='h', color='count')
    st.plotly_chart(fig_edu, use_container_width=True)

# --- PAGE: PREDICTION ---
elif page == "🔮 Prediction":
    st.title("Candidate Win Predictor")
    st.write("Enter candidate details to predict the likelihood of winning based on historical patterns.")

    # Model Preparation (Simplified)
    features = ['AGE', 'ASSETS', 'LIABILITIES', 'CRIMINAL_CASES', 'GENDER', 'CATEGORY', 'EDUCATION']
    model_df = df[features + ['WINNER']].dropna()
    
    le_dict = {}
    for col in ['GENDER', 'CATEGORY', 'EDUCATION']:
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col].astype(str))
        le_dict[col] = le
    
    X = model_df[features]
    y = model_df['WINNER']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Input Form
    with st.form("pred_form"):
        c1, c2 = st.columns(2)
        age = c1.slider("Age", 25, 100, 45)
        assets = c2.number_input("Total Assets (₹)", min_value=0.0, value=10000000.0)
        liabilities = c1.number_input("Liabilities (₹)", min_value=0.0, value=0.0)
        criminal = c2.number_input("Criminal Cases", min_value=0, value=0)
        
        gender = c1.selectbox("Gender", df['GENDER'].unique())
        category = c2.selectbox("Category", df['CATEGORY'].unique())
        edu = st.selectbox("Education Level", df['EDUCATION'].unique())
        
        submit = st.form_submit_button("Predict Result")

    if submit:
        # Encode inputs
        try:
            g_enc = le_dict['GENDER'].transform([gender])[0]
            cat_enc = le_dict['CATEGORY'].transform([category])[0]
            edu_enc = le_dict['EDUCATION'].transform([edu])[0]
            
            input_data = [[age, assets, liabilities, criminal, g_enc, cat_enc, edu_enc]]
            prediction = rf.predict(input_data)[0]
            prob = rf.predict_proba(input_data)[0][1]

            if prediction == 1:
                st.success(f"Prediction: **WINNER** (Confidence: {prob:.2%})")
            else:
                st.error(f"Prediction: **UNLIKELY TO WIN** (Confidence: {1-prob:.2%})")
        except:
            st.warning("Could not process prediction. Check if all categories are present in the training data.")

# --- PAGE: BULK SCANNER ---
elif page == "🔍 Bulk Scanner":
    st.title("📊 VoteIntel Bulk Scanner")

# -----------------------------
# 🔹 Sample Data Create
# -----------------------------
sample_data = pd.DataFrame({
    "AGE": [45, 50],
    "ASSETS": [5000000, 2000000],
    "CRIMINAL CASES": [1, 0],
    "EDUCATION": ["Graduate", "Post Graduate"],
    "STATE": ["Gujarat", "Maharashtra"]
})

# -----------------------------
# 🔹 Download Buttons
# -----------------------------
st.subheader("⬇ Download Sample File")

# CSV
csv = sample_data.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "sample.csv", "text/csv")

# Excel
excel_file = "sample.xlsx"
sample_data.to_excel(excel_file, index=False)
with open(excel_file, "rb") as f:
    st.download_button("Download Excel", f, "sample.xlsx")

# JSON
json_data = sample_data.to_json(orient="records")
st.download_button("Download JSON", json_data, "sample.json", "application/json")


# -----------------------------
# 🔹 Upload File
# -----------------------------
st.subheader("📤 Upload CSV File")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# -----------------------------
# 🔹 Dummy Prediction Function
# -----------------------------
def predict(df):
    # 🔹 Convert column to numeric (IMPORTANT FIX)
    df["CRIMINAL CASES"] = pd.to_numeric(df["CRIMINAL CASES"], errors='coerce')

    # 🔹 Fill NaN with 0 (optional but recommended)
    df["CRIMINAL CASES"] = df["CRIMINAL CASES"].fillna(0)

    # 🔹 Prediction logic
    df["Prediction"] = df["CRIMINAL CASES"].apply(
        lambda x: "High Risk" if x > 0 else "Low Risk"
    )

    return df
# -----------------------------
# 🔹 Process File
# -----------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("📄 Uploaded Data")
    st.dataframe(df)

    result = predict(df)

    st.write("✅ Prediction Result")
    st.dataframe(result)

    # Download result
    result_csv = result.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download Result", result_csv, "result.csv", "text/csv")
