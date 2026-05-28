# Back Office AI Router + RAG

AI-powered enterprise document intelligence system built using:

- Streamlit
- LangChain
- Groq LLM
- FAISS Vector Database
- HuggingFace Embeddings

## Features

- Document classification
- Department routing
- Confidence scoring
- Semantic document search
- Retrieval-Augmented Generation (RAG)
- Chat with uploaded documents
- PDF, DOCX, TXT support

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq API

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_key_here
````