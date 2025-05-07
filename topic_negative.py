import pandas as pd
from bertopic import BERTopic

df = pd.read_csv("data/sentiment_scored.csv")
neg_texts = df.query("sent_pred == 0")["clean_text"].tolist()

topic_model = BERTopic(language="english")
topics, _ = topic_model.fit_transform(neg_texts)

topic_info = topic_model.get_topic_info().head(10)  # top 10 clusters
topic_info.to_csv("data/topic_info.csv", index=False)
print("📝  Saved data/topic_info.csv with", len(topic_info), "topics")
