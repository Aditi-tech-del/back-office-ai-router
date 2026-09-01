"""
Retrieval-augmented question answering over a single document.
"""

from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE, RAG_RETRIEVER_K

AGENT_PROMPT_TEMPLATE = """
You are a helpful AI assistant.

Answer ONLY from the
provided context.

If the answer is not found,
say you could not find it.

Context:
{context}

Question:
{question}
"""


def build_retriever(docs: List[Document], embeddings):
    """Chunk a document, embed it, and return a FAISS retriever."""
    rag_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
    )

    rag_docs = rag_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(rag_docs, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": RAG_RETRIEVER_K})


def answer_question(
    user_question: str,
    retriever,
    chat_agent,
) -> Dict[str, Any]:
    """
    Retrieve relevant chunks for a question and ask the PII-protected
    chat agent to answer using only that context. Returns
    {"answer", "sources"}.

    PII in `user_question`, the retrieved `context`, and the model's
    answer is masked by the agent's PIIMiddleware stack (see
    pii_middleware.py) before/after the underlying model call.
    """
    retrieved_docs = retriever.invoke(user_question)

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = AGENT_PROMPT_TEMPLATE.format(
        context=context,
        question=user_question,
    )

    result = chat_agent.invoke({"messages": [HumanMessage(content=prompt)]})
    answer = result["messages"][-1].content

    return {
        "answer": answer,
        "sources": retrieved_docs,
    }
