# RAG Project

This project is a Streamlit-based local UI for a retrieval-augmented generation workflow.

## What it does

- Fetches text from a web page URL.
- Extracts text from a local PDF file.
- Optionally summarizes extracted text using Mistral AI if `MISTRAL_API_KEY` is set.

## Run locally

1. Activate your Python environment.
2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Start the Streamlit app:
   ```bash
   streamlit run main.py
   ```
4. Open the app in your browser:
   - `http://localhost:8501`

## Environment

Create a `.env` file in the project root with:

```env
MISTRAL_API_KEY=your_mistral_api_key
```

If `MISTRAL_API_KEY` is not set, the app will still fetch and extract text, but the summarization feature will be disabled.

## GitHub

- Repository: https://github.com/Abhi0833-eng/Rag_project
