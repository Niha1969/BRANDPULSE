import streamlit as st
from pathlib import Path

st.title("⚙️ Admin & Settings")

st.markdown("Upload a new Hugging Face model (.pt or directory zip) to hot‑swap the inference backend.")
file = st.file_uploader("Model file or zip", type=["pt", "zip"])
if file and st.button("Save model"):
    dest = Path("models") / file.name
    dest.parent.mkdir(exist_ok=True)
    with open(dest, "wb") as f:
        f.write(file.getbuffer())
    st.success(f"Saved to {dest}\nRestart Streamlit to use the new model.")