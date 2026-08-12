import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    if not pages:
        raise ValueError(f"No text could be extracted from: {pdf_path}")

    text = "\n\n".join(page.page_content for page in pages)
    return text.strip()


def build_embeddings_from_text(text: str):
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is missing. Add it to your .env file.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([text])

    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)
    vectors = [embeddings.embed_query(chunk.page_content) for chunk in chunks]

    return chunks, vectors


def main():
    parser = argparse.ArgumentParser(description="Extract text from a PDF and generate Mistral embeddings.")
    parser.add_argument("pdf_path", help="Path to the PDF file you want to process")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    print(f"Reading PDF: {pdf_path}")

    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted text length: {len(text)} characters")

    chunks, vectors = build_embeddings_from_text(text)
    print(f"Total chunks created: {len(chunks)}")
    print(f"Embedding dimension: {len(vectors[0])}")
    print("First chunk preview:")
    print(chunks[0].page_content[:300])


if __name__ == "__main__":
    main()
