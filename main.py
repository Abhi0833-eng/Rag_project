import os

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY is missing. Add it to your .env file.")

model = ChatMistralAI(model="mistral-small-latest", api_key=api_key)

result = model.invoke("Hello")
print(result.content)