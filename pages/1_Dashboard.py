import streamlit as st
import plotly.express as px
import pandas as pd
from utils.loader import load_data

st.title("📊 Dashboard")

df = load_data()
if df is None:
    st.warning("No data found. Please upload a CSV via Admin page or put sample_tweets.csv in /data.")
    st.stop()

# Top‑level KPIs
total = len(df)
pos_pct = (df["sentiment"] == "positive").mean() * 100
neg_pct = (df["sentiment"] == "negative").mean() * 100
col1, col2, col3 = st.columns(3)
col1.metric("Total tweets", f"{total:,}")
col2.metric("% Positive", f"{pos_pct:.1f}%")
col3.metric("% Negative", f"{neg_pct:.1f}%")

st.divider()

# Trend over time
if "Datetime" in df.columns:
    ts = (
        df.set_index("Datetime")
        .resample("6H")
        .agg(pos_pct=("sentiment", lambda x: (x == "positive").mean() * 100))
        .reset_index()
    )
    fig = px.line(ts, x="Datetime", y="pos_pct", title="Sentiment trend (6‑hour bins)")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# BERTopic table (if exists)
try:
    topic_info = pd.read_csv("data/topic_info.csv")
    st.subheader("Top Negative Topics (BERTopic)")
    st.dataframe(topic_info[["Topic", "Name", "Count"]])
except FileNotFoundError:
    st.info("Run topic_negative.py to generate topic_info.csv")