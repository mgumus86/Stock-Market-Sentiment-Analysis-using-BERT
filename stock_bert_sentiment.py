import requests
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import finnhub
import time

# -----

# --- Step 2: Get stock prices (last 400 days, limit to 300) ---
symbol = 'TSLA'
price_df = yf.download(symbol, period="400d", interval="1d")
price_df = price_df.reset_index()                        # remove Date index
price_df["date"] = pd.to_datetime(price_df["Date"].dt.date)  # convert to plain date
price_df = price_df[["date", "Close"]].rename(columns={"Close": "close"})
price_df = price_df.tail(300)                            # keep last 300 trading days


# -----

# 🔍 Confirm both are flat DataFrames with clean 'date'
price_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in price_df.columns]

# -----

price_df = price_df.rename(columns={"date_": "date"})

# -----

# Setup client
api_key = 'd0c55f9r01qs9fjklatgd0c55f9r01qs9fjklau0'
finnhub_client = finnhub.Client(api_key=api_key)

symbol = 'TSLA'
today = datetime.today()
start_date = today - timedelta(days=400)

# Convert to string format YYYY-MM-DD
start_str = start_date.strftime('%Y-%m-%d')
end_str = today.strftime('%Y-%m-%d')

def fetch_news_chunk(symbol, start, end):
    try:
        news = finnhub_client.company_news(symbol, _from=start, to=end)
        return news
    except:
        return []

news_items = []

# Chunk loop
current = start_date
while current < today:
    chunk_start = current
    chunk_end = min(current + timedelta(days=30), today)

    chunk_news = fetch_news_chunk(symbol, chunk_start.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d'))
    news_items.extend(chunk_news)

    current = chunk_end
    time.sleep(1)  # avoid hitting rate limit

# -----

df_news = pd.DataFrame(news_items)

df_news['datetime'] = pd.to_datetime(df_news['datetime'], unit='s', errors='coerce')
df_news = df_news[df_news['datetime'].notnull()]  # drop any that still failed
df_news['date'] = df_news['datetime'].dt.date

# -----

# Clean output
df_news = df_news[['date', 'headline', 'summary']]
print(df_news.head())

# -----

# Combine all headlines and summaries per date
grouped_news = df_news.groupby("date").agg({
    "headline": lambda x: " | ".join(x),
    "summary": lambda x: " | ".join(x)
}).reset_index()

# Optional: rename columns
grouped_news = grouped_news.rename(columns={
    "headline": "all_headlines",
    "summary": "all_summaries"
})

# Preview
print(grouped_news.head())

# -----

price_df["date"] = pd.to_datetime(price_df["date"]).dt.date
grouped_news["date"] = pd.to_datetime(grouped_news["date"]).dt.date
merged_df = pd.merge(price_df, grouped_news, on="date", how="left")

# -----

merged_df.tail()

# -----

test = merged_df.iloc[[-1], :]

# -----

test

# -----

merged_df = merged_df.sort_values("date").reset_index(drop=True)

# Create 'next_day_close' by shifting the 'close' column
merged_df["next_day_close"] = merged_df["close_TSLA"].shift(-1)

# Create binary target: 1 if next day's close is higher, else 0
merged_df["target"] = (merged_df["next_day_close"] > merged_df["close_TSLA"]).astype(int)

merged_df = merged_df.dropna().reset_index(drop=True)

# -----

merged_df

# -----

def prepare_text(headlines, summaries):
    # Get first 2 headlines
    headlines_list = str(headlines).split(" | ")
    first_two = " | ".join(headlines_list[:2])

    # Limit summary to 200 characters
    trimmed_summary = str(summaries)[:200]

    return first_two + " " + trimmed_summary

# Apply to all rows
merged_df["text"] = merged_df.apply(
    lambda row: prepare_text(row["all_headlines"], row["all_summaries"]),
    axis=1
)

# Define input and labels
texts = list(merged_df["text"])
labels = list(merged_df["target"])

# -----

# ALL HEADLINES AND SUMMARIES

# Combine text columns for BERT input
# merged_df["text"] = (
  #  merged_df["all_headlines"].fillna("") + " " + merged_df["all_summaries"].fillna("")
#)

# Define inputs and labels
#texts = list(merged_df["text"])
#labels = list(merged_df["target"])

# -----

from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Train/val split
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenize with truncation/padding
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# -----

import torch

class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = BERTDataset(train_encodings, train_labels)
val_dataset = BERTDataset(val_encodings, val_labels)

# -----

from transformers import DistilBertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# -----

for epoch in range(5):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        # ✅ After (exclude token_type_ids if it exists):
        batch = {k: v.to(device) for k, v in batch.items() if k != "token_type_ids"}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

# -----

from sklearn.metrics import accuracy_score

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "token_type_ids"}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy: {acc:.4f}")

# -----

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    label = torch.argmax(probs).item()
    return label, probs.detach().cpu().numpy()

# Example
text = "Tesla earnings beat expectations. New product line launched."
label, probs = predict_sentiment(text)
print("Prediction:", "UP" if label == 1 else "DOWN", "→", probs)

# -----

test["text"] = test.apply(
    lambda row: prepare_text(row["all_headlines"], row["all_summaries"]),
    axis=1
)

test_text = list(test["text"])

# -----

model.eval()

# Tokenize and move to device
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()

# Show result
print("Prediction:", "UP" if prediction == 1 else "DOWN")
print("Confidence:", probs.cpu().numpy())