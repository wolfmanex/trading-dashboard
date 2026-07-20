import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class NewsSentimentEngine:
    def __init__(self, symbol: str = None):
        # Download lexicon once when class is instantiated
        nltk.download('vader_lexicon', quiet=True)
        self.analyzer = SentimentIntensityAnalyzer()
        self.symbol = symbol

    def get_sentiment(self, symbol: str = None) -> dict:
        """
        Fetches news articles and returns a dictionary with 'score' and 'status'.
        """
        target_symbol = symbol or self.symbol
        if not target_symbol:
            print("News Engine Error: No symbol provided.")
            return {"score": 0.0, "status": "Neutral ⚖️"}

        try:
            ticker = yf.Ticker(target_symbol)
            news_items = ticker.news
            
            if not news_items:
                return {"score": 0.0, "status": "No Data ⚪"}

            scores = []
            for item in news_items:
                # Safely extract title from Yahoo's updated nested structure
                content = item.get('content', item)
                title = content.get('title', '')
                
                # Only score non-empty headline titles
                if title:
                    compound = self.analyzer.polarity_scores(title)['compound']
                    scores.append(compound)
            
            # Calculate average sentiment score
            avg_score = (sum(scores) / len(scores)) if scores else 0.0

            # Determine human-readable status label for Streamlit
            if avg_score >= 0.05:
                status = "Bullish 🚀"
            elif avg_score <= -0.05:
                status = "Bearish 🔻"
            else:
                status = "Neutral ⚖️"

            return {
                "score": round(avg_score, 3),
                "status": status
            }

        except Exception as e:
            print(f"News Engine Error: {e}")
            return {"score": 0.0, "status": "Error ⚠️"}

    # Aliases so any function call in app.py returns this same dictionary structure
    get_news_sentiment = get_sentiment
    analyze = get_sentiment
    analyze_sentiment = get_sentiment
    get_score = get_sentiment
    get_news_context = get_sentiment