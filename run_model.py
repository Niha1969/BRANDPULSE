"""
run_model.py
============

• Load cleaned tweets from data/dell_tweets_clean.csv
• Score sentiment with Twitter‑RoBERTa
• Save data/sentiment_scored.csv  (adds sent_pred + probabilities)
• Compute SHAP global importances on a 500‑tweet sample
• Save data/shap_global.csv
"""

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, torch.nn.functional as F
import shap, numpy as np
from pathlib import Path

# --------------------------------------------------
#  Config
# --------------------------------------------------
DATA_PATH = Path("data/dell_tweets_clean.csv")
OUT_PATH  = Path("data/sentiment_scored.csv")
SHAP_OUT  = Path("data/shap_global.csv")
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

# --------------------------------------------------
#  Load data
# --------------------------------------------------
print("🔄  Loading cleaned CSV …")
df = pd.read_csv(DATA_PATH)
texts = df["clean_text"].astype(str).tolist()

# --------------------------------------------------
#  Load model
# --------------------------------------------------
print("🔄  Loading tokenizer & model …")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# --------------------------------------------------
#  Batch scoring
# --------------------------------------------------
def score(batch):
    enc = tok(batch, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits          # (B, 3)
    probs = F.softmax(logits, dim=-1).cpu()
    preds = probs.argmax(1).numpy()
    return preds, probs.numpy()

all_preds, all_probs = [], []
print("🚀  Scoring …")
BATCH = 64
for i in range(0, len(texts), BATCH):
    p, pr = score(texts[i : i + BATCH])
    all_preds.extend(p)
    all_probs.extend(pr)
    if i % 2048 == 0:
        print(f"  {i}/{len(texts)} rows done")

df["sent_pred"] = all_preds
df[["proba_neg", "proba_neu", "proba_pos"]] = pd.DataFrame(all_probs, index=df.index)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"✅  Saved predictions ➜  {OUT_PATH.resolve()}")



# --------------------------------------------------
#  SHAP global importance  (sample 500 tweets)
# --------------------------------------------------
print("🔍  Computing SHAP importances …")

if tok.pad_token_id is None:              # RoBERTa needs a pad‑token
    tok.pad_token = tok.eos_token

sample_texts = df["clean_text"].tolist()[:500]   # smaller = faster

masker = shap.maskers.Text(tokenizer=tok)

def f(x):
    enc = tok(list(x), padding=True, truncation=True,
              return_tensors="pt").to(device)
    with torch.no_grad():
        return model(**enc).logits.cpu()

explainer  = shap.Explainer(f, masker, output_names=["neg", "neu", "pos"])
shap_vals  = explainer(sample_texts)

# -------- flatten to (token, |SHAP|) pairs ----------
token_imps = {}
for svals, sdata in zip(shap_vals.values, shap_vals.data):
    for token, val in zip(sdata, svals.mean(axis=1)):   # mean over 3 classes
        token = token.strip()
        if not token.isalpha() or len(token) < 3:
            continue
        token_imps.setdefault(token, []).append(abs(val))

importances = {t: np.mean(vs) for t, vs in token_imps.items()}
shap_df = (pd.Series(importances, name="importance")
             .sort_values(ascending=False)
             .head(20)
             .reset_index()
             .rename(columns={"index": "token"}))

SHAP_OUT.parent.mkdir(parents=True, exist_ok=True)
shap_df.to_csv(SHAP_OUT, index=False)
print(f"✅  Saved SHAP global importances ➜  {SHAP_OUT.resolve()}")

#patch to run_model.py

def score_texts(texts, model, tokenizer):
    """Helper used by Streamlit Inference page."""
    import torch
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**enc).logits.softmax(-1)
    preds = out.argmax(-1).tolist()
    probas = out.max(-1).values.tolist()
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    return [(label_map[p], probas[i]) for i, p in enumerate(preds)]
