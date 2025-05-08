import streamlit as st
import pandas as pd
from utils.loader import load_model
from run_model import score_texts   # you'll add a small helper in run_model

st.title("🔮 Inference")
model, tokenizer = load_model()
if model is None:
    st.error("Model not found. Place the HF model in /models or run run_model.py.")
    st.stop()

cols = st.tabs(["Single text", "Batch CSV"])

with cols[0]:
    text = st.text_area("Enter text to analyse", height=150)
    if st.button("Predict") and text:
        pred, proba = score_texts([text], model, tokenizer)[0]
        st.write(f"**Prediction**: {pred}  (neg, neu, pos) – Prob: {proba:.2f}")

with cols[1]:
    file = st.file_uploader("Upload CSV with a 'text' column", type=["csv"])
    if file and st.button("Batch predict"):
        df = pd.read_csv(file)
        preds = score_texts(df["text"].tolist(), model, tokenizer)
        df["sent_pred"] = [p for p, _ in preds]
        st.success("Done – download below")
        st.download_button("Download scored CSV", df.to_csv(index=False), "scored.csv")