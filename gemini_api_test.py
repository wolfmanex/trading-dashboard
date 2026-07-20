from google import genai
import os

# Grab your key or paste it here temporarily
api_key = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
client = genai.Client(api_key=api_key)

print("--- Models available on your API key ---")
for model in client.models.list():
    if "generateContent" in model.supported_actions:
        print(model.name)