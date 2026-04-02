import streamlit as st
import pandas as pd

# Page config
st.set_page_config(page_title="Election Dashboard", layout="wide")

# Title
st.title("📊 Election Data Analysis Dashboard")

# Load Data
df = pd.read_csv("final_voteintel_dataset.csv")

# Sidebar filters
st.sidebar.header("Filters")

state = st.sidebar.selectbox("Select State", df["STATE"].unique())
party = st.sidebar.selectbox("Select Party", df["PARTY"].unique())

filtered_df = df[(df["STATE"] == state) & (df["PARTY"] == party)]

# Metrics
col1, col2 = st.columns(2)

col1.metric("Total Candidates", len(filtered_df))
col2.metric("Winners", filtered_df["WINNER"].sum())

# Charts
st.subheader("Winner Distribution")
st.bar_chart(filtered_df["WINNER"].value_counts())

st.subheader("Party Wise Winners")
winners = filtered_df[filtered_df["WINNER"] == 1]
st.bar_chart(winners["PARTY"].value_counts())

st.subheader("State Wise Votes")
state_votes = filtered_df.groupby("STATE")["TOTAL VOTES"].sum()
st.bar_chart(state_votes)

st.subheader("Net Worth Distribution")
st.bar_chart(filtered_df["NET_WORTH"])



