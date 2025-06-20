# 📈 Stock Market Sentiment Analysis with BERT

This project performs sentiment analysis on **Tesla (TSLA)** news headlines using a **BERT-based sentiment classifier**, then analyzes how sentiment aligns with stock price trends.

News data is collected using the **Finnhub API**, while historical stock prices come from **Yahoo Finance (via yfinance)**. The project applies a **BERT Sentiment Classifier** to evaluate the tone of daily headlines and visualizes potential connections to market movement.

---

## 🤖 BERT Sentiment Classifier

This project uses a **pretrained BERT model** from Hugging Face to perform sentiment classification on financial news headlines.

- The model is fine-tuned (or used with zero-shot capability) on financial headline sentiment.
- Sentiment categories include: `positive`, `negative`, and `neutral`.
- Results are aligned with stock price trends to detect possible correlation between market sentiment and performance.

We use the `transformers` library (`bert-base-uncased` or a similar model) and `pipeline("sentiment-analysis")` from Hugging Face for fast implementation.

---
