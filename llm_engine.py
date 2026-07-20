from google import genai

class GeminiAdvisor:
    def __init__(self, api_key: str):
        # Initialize the client with your API key
        self.client = genai.Client(api_key=api_key)

    def analyze_trade_setup(self, symbol: str, tech_data: str, news_context: dict) -> str:
        """
        Passes market data to Gemini for a rapid synthesis.
        """
        # Construct the prompt using the data from your other engines
        prompt = f"""
        You are an expert quantitative trading assistant.
        Review the current live data for {symbol}:
        - Technical Signal: {tech_data}
        - News Sentiment Score: {news_context['score']} ({news_context['status']})

        Based on this data, provide a concise, two-sentence synthesis of the current trade setup. 
        Focus strictly on the alignment (or divergence) between the news sentiment and the technicals.
        Do not provide direct financial advice.
        """
        
        try:
            # Call the Gemini 2.5 Flash model for fast, lightweight reasoning
            response = self.client.models.generate_content(
                model='gemini-flash-latest',
                contents=prompt
            )
            return response.text
            
        except Exception as e:
            return f"Gemini API Error: {e}"