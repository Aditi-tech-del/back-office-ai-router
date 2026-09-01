# Back Office AI Router

Upload PDF/DOCX/TXT office documents, classify each one with an LLM
(Invoice, Contract, HR Document, etc.), get a suggested department to
route it to, and chat with each document via a RAG assistant.

## Structure

| File | Responsibility |
|---|---|
| `app.py` | Entrypoint — wires everything together, holds the per-file loop |
| `config.py` | Page config, env var / API key loading, tunable constants |
| `styles.py` | Custom CSS injected into the page |
| `routing.py` | `ROUTING_MAP` (document type → department) |
| `utils.py` | `clean_llm_json` — extract JSON from an LLM response |
| `models.py` | Cached loaders for embeddings, chat LLM, classification chain |
| `document_loader.py` | Temp-file handling + PDF/DOCX/TXT loading |
| `classification.py` | Splits + classifies a document, returns `ClassificationResult` |
| `rag_engine.py` | Builds a FAISS retriever, answers questions from context |
| `ui_components.py` | All `st.*` rendering (header, cards, chat, export, footer) |

## Run

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in GROQ_API_KEY
streamlit run app.py
```

## Notes

- Business logic (classification, RAG) has no `st.*` calls inside it
  except where a spinner is the most natural place to show progress —
  this keeps `classification.py` and `rag_engine.py` independently
  testable.
- All rendering lives in `ui_components.py`, so restyling the app
  doesn't require touching the processing pipeline.
- `document_loader.save_temp_file` is a context manager, so the temp
  file is always cleaned up (replacing the original's `try/finally`).
