import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY is missing. Add it to your .env file.")

file_path = Path(__file__).with_name("sample.txt")
text = file_path.read_text(encoding="utf-8")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.create_documents([text], metadatas=[{"source": file_path.name}])
print(f"Total chunks: {len(docs)}")

embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)
vector = embeddings.embed_query(docs[0].page_content)
print(f"Embedding dimension: {len(vector)}")
print("First 5 values:", vector[:5])
print("Sample text chunk:")
print(docs[0].page_content)
