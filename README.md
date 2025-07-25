# Retrieval-Augmented-Gen

A lightweight Retrieval-Augmented Generation (RAG) app using Streamlit to answer any questions from uploaded files (PDF, DOCX, TXT). Combines semantic search with FLAN-T5-based generation.

 Features:

Upload PDF, DOCX, or TXT files

Paragraph-aware chunking

FAISS-based semantic search

Keyword reranking to refine context

FLAN-T5 model generates detailed answers

Chat history persists in session

 Tech Stack:

Streamlit

sentence-transformers (all-MiniLM-L6-v2)

transformers (google/flan-t5-base)

faiss-cpu
