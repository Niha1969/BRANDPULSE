import streamlit as st
from pathlib import Path

st.set_page_config(page_title="BrandPulse – Home", page_icon="📊", layout="wide")

st.title("BrandPulse – Real-time Brand-Sentiment Monitoring")

st.markdown("""
**Course**: CETM46  
**Author**: Nihal  

• Quickly explore KPIs on the **Dashboard** page.  
• Score new text or CSV files on **Inference**.  
• Inspect model performance in **Observability**.  
• Upload upgraded models under **Admin**.
""")

# Dev-journey accordion (optional blurb for the report)
with st.expander("Development journey ⏱️"):
    st.markdown("""
    * Sprint 0 – Data acquisition & cleaning  
    * Sprint 1 – Multipage Streamlit skeleton  
    * Sprint 2 – Inference + Observability  
    * Sprint 3 – CI/CD & docs
    """)

st.subheader("Sample Data")
from utils.loader import load_data
sample_df = load_data(sample=True)
if sample_df is not None:
    st.write(sample_df.head())
else:
    st.info("Add **data/sample_tweets.csv** to enable the quick-demo 🙌")
