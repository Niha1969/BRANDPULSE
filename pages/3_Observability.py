import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 Observability & Explainability")

try:
    roc_df = pd.read_csv("data/roc_points.csv")
    fig = px.area(roc_df, x="fpr", y="tpr", title="ROC Curve")
    st.plotly_chart(fig, use_container_width=True)
except FileNotFoundError:
    st.info("Add ROC data to data/roc_points.csv or generate it via tests.")

try:
    shap_df = pd.read_csv("data/shap_global.csv")
    fig = px.bar(shap_df.head(20), x="importance", y="token", orientation="h", title="Global SHAP token importance")
    st.plotly_chart(fig, use_container_width=True)
except FileNotFoundError:
    st.info("Run your SHAP script to create data/shap_global.csv")