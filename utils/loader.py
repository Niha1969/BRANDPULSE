# utils/loader.py  ← make sure the path & filename are exact
from functools import lru_cache
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_DIR = Path("data")
MODEL_DIR = Path("models/twitter-roberta-base-sentiment")  # change if you fine-tune

@lru_cache(maxsize=1)
def load_data(sample=False):
    """Load cleaned tweets or tiny sample."""
    file = DATA_DIR / ("sample_tweets.csv" if sample else "dell_tweets_clean.csv")
    if not file.exists():
        return None
    df = pd.read_csv(file, parse_dates=["Datetime"], low_memory=False)
    return df

@lru_cache(maxsize=1)
def load_model():
    if not MODEL_DIR.exists():
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).eval()
    return model, tokenizer
