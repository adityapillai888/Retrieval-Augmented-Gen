import streamlit as st
from PyPDF2 import PdfReader
import docx
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import faiss
import numpy as np
import re

# --- File Parsing ---
def parse_txt(file):
    return file.read().decode("utf-8")

def parse_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def parse_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def parse_file(file):
    if file.type == "text/plain":
        return parse_txt(file)
    elif file.type == "application/pdf":
        return parse_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return parse_docx(file)
    else:
        return ""

# --- Paragraph-aware Chunking ---
def chunk_text(text, max_length=500):
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) <= max_length:
            current_chunk += " " + para
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# --- Load Models ---
@st.cache_resource(show_spinner=False)
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    return embedder, tokenizer, model

embedder, tokenizer, model = load_models()

# --- Embedding + FAISS ---
def embed_texts(texts):
    return embedder.encode(texts, convert_to_numpy=True)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_index(index, query_embedding, top_k=5):
    distances, indices = index.search(query_embedding, top_k)
    return indices[0]

# --- Keyword Reranking ---
def rerank_by_keyword(chunks, indices, query):
    keywords = [k.strip().lower() for k in query.lower().split() if len(k) > 3]
    scored = []
    for i in indices:
        chunk = chunks[i]
        score = sum(1 for k in keywords if k in chunk.lower())
        scored.append((score, chunk))
    scored.sort(reverse=True)
    return [chunk for _, chunk in scored[:3]]

# --- Generate Answer ---
def generate_answer(question, context):
    question = question.strip()
    context = re.sub(r'\s+', ' ', context.strip())

    prompt = (
        f"You are a helpful, intelligent assistant. Based on the CONTEXT provided below, "
        f"carefully read and understand the relevant information and provide a detailed, clear, and accurate answer to the QUESTION.\n\n"
        f"CONTEXT: {context}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )



    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)

    output = model.generate(
        input_ids,
        max_length=1000,
        num_beams=5,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
        early_stopping=False,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# --- Streamlit UI ---
st.title("📄 RAG Legal Document QA")

uploaded_files = st.file_uploader(
    "Upload documents (PDF, DOCX, TXT)",
    accept_multiple_files=True,
    type=['pdf', 'docx', 'txt']
)

if uploaded_files:
    with st.spinner("Parsing documents..."):
        docs = [parse_file(file) for file in uploaded_files]

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc))

    with st.spinner("Embedding & indexing chunks..."):
        embeddings = embed_texts(all_chunks)
        index = build_faiss_index(embeddings)

    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.text_input("Ask a legal/compliance question:")

    if st.button("Get Answer") and question:
        q_emb = embed_texts([question])
        top_k = search_index(index, q_emb, top_k=10)
        best_chunks = rerank_by_keyword(all_chunks, top_k, question)
        context = "\n\n".join(best_chunks)

        answer = generate_answer(question, context)
        st.session_state.history.append({"question": question, "answer": answer})

    st.markdown("---")
    st.header("Chat History")
    for chat in reversed(st.session_state.history):
        st.markdown(f"**Q:** {chat['question']}")
        st.markdown(f"**A:** {chat['answer']}")

else:
    st.info("Upload at least one document to begin.")
