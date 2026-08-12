import os
from pathlib import Path

import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")


def fetch_page_text(url: str) -> str:
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    return soup.get_text(separator="\n", strip=True)


def extract_text_from_pdf(pdf_path: str) -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(path))
    pages = loader.load()
    if not pages:
        raise ValueError(f"No text could be extracted from: {pdf_path}")

    return "\n\n".join(page.page_content for page in pages)


def summarize_text(text: str) -> str:
    if not API_KEY:
        raise ValueError("MISTRAL_API_KEY is missing. Add it to your .env file.")

    from langchain_mistralai import ChatMistralAI

    model = ChatMistralAI(model="mistral-small-latest", api_key=API_KEY)
    result = model.invoke(f"Summarize the following text:\n\n{text}")
    return result.content


def main() -> None:
    st.set_page_config(page_title="RAG Project Streamlit", layout="wide")
    st.title("RAG Project Streamlit")
    st.write("Fetch web page text or extract text from a local PDF file.")

    st.sidebar.header("Options")
    task = st.sidebar.selectbox("Choose a task", ["Fetch web page text", "Extract PDF text"])

    if task == "Fetch web page text":
        url = st.text_input("Web page URL", "https://example.com")
        if st.button("Fetch text"):
            with st.spinner("Fetching page text..."):
                text = fetch_page_text(url)
            st.success("Text fetched successfully.")
            st.write(f"**Total characters:** {len(text)}")
            st.write(f"**Total words:** {len(text.split())}")
            st.text_area("Page text preview", text[:3000], height=300)
            if API_KEY and st.checkbox("Generate summary", value=False):
                with st.spinner("Generating summary..."):
                    summary = summarize_text(text)
                st.subheader("Summary")
                st.write(summary)

    else:
        pdf_path = st.text_input("Local PDF path", "document loaders/sample.pdf")
        if st.button("Extract PDF text"):
            try:
                with st.spinner("Extracting PDF text..."):
                    text = extract_text_from_pdf(pdf_path)
                st.success("PDF text extracted successfully.")
                st.write(f"**Total characters:** {len(text)}")
                st.write(f"**Total words:** {len(text.split())}")
                st.text_area("PDF text preview", text[:3000], height=300)
                if API_KEY and st.checkbox("Generate summary", value=False):
                    with st.spinner("Generating summary..."):
                        summary = summarize_text(text)
                    st.subheader("Summary")
                    st.write(summary)
            except FileNotFoundError as exc:
                st.error(str(exc))
            except ImportError:
                st.error("Missing package: install 'langchain-community' to extract PDF text.")
            except Exception as exc:
                st.error(f"Failed to extract text: {exc}")

    st.sidebar.markdown("---")
    st.sidebar.write("For PDF extraction, use a local file path relative to the project root.")
    if not API_KEY:
        st.sidebar.warning("Set MISTRAL_API_KEY in .env to enable text summarization.")


if __name__ == "__main__":
    main()
