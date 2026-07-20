import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 1. Force the lexicon download
nltk.download('vader_lexicon', quiet=True)

def test_news_pipeline(symbol):
    print(f"--- Fetching news for {symbol} ---")
    ticker = yf.Ticker(symbol)
    news_items = ticker.news
    
    print(f"Articles found: {len(news_items)}")
    
    if not news_items:
        print("Failure: Yahoo returned no news.")
        return

    # 2. Test VADER on the first article
    analyzer = SentimentIntensityAnalyzer()
    first_title = news_items[0].get('title', 'No Title Found')
    
    # 3. Calculate compound score
    scores = analyzer.polarity_scores(first_title)
    
    print(f"Latest Headline: '{first_title}'")
    print(f"VADER Sentiment Breakdown: {scores}")
    print(f"Final Compound Score: {scores['compound']}") # -1 to 1

test_news_pipeline("AMD")