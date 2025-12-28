from google import genai
from google.genai import types
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

resp = client.models.generate_content(
    model="gemini-1.0-pro",
    contents="What is organic farming?",
)

print(resp.text)
