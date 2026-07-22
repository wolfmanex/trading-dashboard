import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


def get_ticker_news_sentiment(ticker: str):
    """
    Fetch news for a ticker via yfinance and analyze sentiment with VADER.
    Returns a tuple: (sentiment_label, sentiment_summary_text)
    """
    try:
        t = yf.Ticker(ticker)
        news_items = t.news
    except Exception as e:
        return "Neutral (Error)", f"Unable to fetch news for {ticker}: {str(e)}"

    if not news_items:
        return "Neutral (No News)", "No recent news articles found for this ticker."

    sia = SentimentIntensityAnalyzer()
    compound_scores = []
    summaries = []

    for item in news_items[:5]:  # Analyze top 5 recent articles
        # Handle yfinance v0.2.50+ nested structure safely
        content = item.get('content', item) if isinstance(item, dict) else item
        title = content.get('title', '') if isinstance(content, dict) else ''

        if title:
            scores = sia.polarity_scores(title)
            compound = scores['compound']
            compound_scores.append(compound)
            summaries.append(f"- \"{title}\" (Score: {compound:.2f})")

    if not compound_scores:
        return "Neutral (No News)", "No parseable news headlines found."

    avg_score = sum(compound_scores) / len(compound_scores)

    if avg_score >= 0.05:
        sentiment_label = f"Bullish (+{avg_score:.2f})"
    elif avg_score <= -0.05:
        sentiment_label = f"Bearish ({avg_score:.2f})"
    else:
        sentiment_label = f"Neutral ({avg_score:.2f})"

    summary_text = f"Average Sentiment Score: {avg_score:.2f}\n" + "\n".join(summaries)

    return sentiment_label, summary_text