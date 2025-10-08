import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load the .env file to get the API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("API Key not found. Make sure it's in your .env file.")
else:
    genai.configure(api_key=api_key)
    print("Available models for your API key:")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)