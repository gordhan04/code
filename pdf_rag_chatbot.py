"""
PDF RAG Chatbot (Gradio + LangChain) — Improved, production-ready single-file script
- Uses LangChain for document loading & text splitting
- Uses SentenceTransformers for embeddings
- Uses FAISS as vectorstore
- Uses a lightweight custom Groq LLM wrapper (configure GROQ_API_URL + GROQ_API_KEY)
- Gradio UI for upload + chat (supports persistent index saving per-upload)

Notes:
- This script intentionally keeps the Groq call generic: set GROQ_API_URL to the Groq model generate endpoint you have access to
  (for example: https://api.groq.ai/v1/models/<model>/generate -- check your Groq docs/account and set the URL accordingly).
- For embeddings we use 'sentence-transformers/all-MiniLM-L6-v2' (fast & good) and LangChain's HuggingFaceEmbeddings.
- For small deployments, .py is recommended; for interactive experiments, use a notebook (.ipynb).

Usage:
1) pip install -r requirements.txt
2) export GROQ_API_KEY=your_key
   export GROQ_API_URL=https://...your groq generate url...
3) python pdf_rag_chatbot.py

"""

import os
import tempfile
import hashlib
import json
from typing import List, Optional, Dict

import requests
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms.base import LLM
from langchain.schema import LLMResult

import gradio as gr

# -------------------------------
# Config
# -------------------------------
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "4"))
INDEX_DIR = os.getenv("INDEX_DIR", "./indices")
os.makedirs(INDEX_DIR, exist_ok=True)

# Groq config via env vars
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL")  # e.g. https://api.groq.ai/v1/models/<model>/generate
GROQ_MODEL = os.getenv("GROQ_MODEL", "groq-model")

# -------------------------------
# Utilities
# -------------------------------

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# -------------------------------
# Groq LLM wrapper for LangChain
# -------------------------------
class GroqLLM(LLM):
    """Minimal LangChain-compatible LLM wrapper for Groq-like HTTP generation endpoints.
    This implementation is intentionally generic: set GROQ_API_URL to the exact generate endpoint.
    """

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None, model: Optional[str] = None, temperature: float = 0.0):
        self.api_key = api_key or GROQ_API_KEY
        self.api_url = api_url or GROQ_API_URL
        self.model = model or GROQ_MODEL
        self.temperature = temperature
        if not self.api_key or not self.api_url:
            raise ValueError("GROQ_API_KEY and GROQ_API_URL environment variables must be set.")

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": float(self.temperature),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
        try:
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Groq API error: {e} — response: {resp.text}")
        data = resp.json()
        # The field with generated text may differ by API. We attempt a few common keys.
        generated = None
        if isinstance(data, dict):
            # common possibilities
            for k in ("text", "output", "completion", "choices"):
                if k in data:
                    generated = data[k]
                    break
            # if choices is a list with text
            if generated is None and "choices" in data and isinstance(data["choices"], list) and len(data["choices"])>0:
                choice = data["choices"][0]
                if isinstance(choice, dict):
                    generated = choice.get("text") or choice.get("message") or json.dumps(choice)
        if generated is None:
            # fallback to raw json string
            generated = json.dumps(data)
        # if choices->text nested
        if isinstance(generated, list):
            # try to extract text fields
            if len(generated) > 0 and isinstance(generated[0], dict):
                generated = generated[0].get("text") or str(generated[0])
            else:
                generated = str(generated)
        return str(generated)

    def _identifying_params(self) -> Dict:
        return {"model": self.model, "temperature": self.temperature}


# -------------------------------
# PDF -> Documents -> Vector Store
# -------------------------------


def load_pdf_to_docs(pdf_path: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(pages)
    return docs


def build_embeddings_and_index(docs: List[Document], index_name: str) -> FAISS:
    """Create embeddings and FAISS index and persist it"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectordb = FAISS.from_documents(docs, embeddings)
    index_path = os.path.join(INDEX_DIR, f"{index_name}.faiss")
    # persist
    vectordb.save_local(os.path.join(INDEX_DIR, f"{index_name}"))
    return vectordb


def load_index_if_exists(index_name: str) -> Optional[FAISS]:
    path = os.path.join(INDEX_DIR, f"{index_name}")
    if os.path.exists(path):
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        return FAISS.load_local(path, embeddings)
    return None


# -------------------------------
# RAG pipeline: retrieve + call LLM
# -------------------------------

def answer_question(vectordb: FAISS, question: str, llm: GroqLLM, top_k: int = TOP_K) -> str:
    docs_and_scores = vectordb.similarity_search_with_relevance_scores(question, k=top_k)
    contexts = []
    for doc, score in docs_and_scores:
        contexts.append(doc.page_content)
    context_text = "\n\n---\n\n".join(contexts)

    prompt = (
        "You are an expert assistant. Use the following extracted sections from a PDF to answer the question. "
        "If the answer cannot be found in the provided context, say you don't know and do not hallucinate.\n\n"
        f"CONTEXT:\n{context_text}\n\nQUESTION: {question}\n\nAnswer concisely and provide citations in the form [source_chunk_i] when useful."
    )

    response = llm(prompt)
    return response


# -------------------------------
# Gradio app
# -------------------------------

class AppState:
    def __init__(self):
        self.vectordb = None
        self.index_name = None
        self.llm = None
        self.history = []

state = AppState()


def handle_upload(file_obj):
    # file_obj is a temp file provided by Gradio
    if file_obj is None:
        return "No file uploaded."
    tmp_path = file_obj.name
    # create stable index name by hashing file contents
    with open(tmp_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    index_name = f"pdf_{file_hash[:12]}"
    state.index_name = index_name

    existing = load_index_if_exists(index_name)
    if existing:
        state.vectordb = existing
        return f"Loaded existing index for this PDF (index: {index_name}). Ready to ask questions."

    docs = load_pdf_to_docs(tmp_path)
    if len(docs) == 0:
        return "Failed to extract text from PDF or PDF is empty."
    vectordb = build_embeddings_and_index(docs, index_name)
    state.vectordb = vectordb
    return f"Index built and saved as {index_name}. Ready to ask questions."


def init_llm(temperature: float = 0.0):
    # Lazily initialize LLM
    if state.llm is None:
        state.llm = GroqLLM(temperature=temperature)
    return state.llm


def ask_question(question: str, temperature: float = 0.0):
    if not state.vectordb:
        return "Please upload a PDF first and build the index."
    llm = init_llm(temperature=temperature)
    answer = answer_question(state.vectordb, question, llm)
    state.history.append((question, answer))
    return answer


def reset():
    state.vectordb = None
    state.index_name = None
    state.llm = None
    state.history = []
    return "Reset complete. Upload a new PDF to start."


def make_demo():
    with gr.Blocks(title="PDF RAG Chatbot (Groq)") as demo:
        gr.Markdown("# PDF RAG Chatbot — Gradio + LangChain + Groq\nUpload a PDF, build an index, then ask questions.")
        with gr.Row():
            with gr.Column(scale=1):
                pdf_in = gr.File(label="Upload PDF", file_count="single", file_types=[".pdf"])                
                upload_btn = gr.Button("Build Index")
                status = gr.Textbox(label="Status", interactive=False)
                reset_btn = gr.Button("Reset")
            with gr.Column(scale=2):
                question = gr.Textbox(label="Your question", placeholder="Ask something about the uploaded PDF...", lines=2)
                temp = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.0, label="Temperature")
                ask_btn = gr.Button("Ask")
                output = gr.Textbox(label="Answer", interactive=False)

        upload_btn.click(fn=handle_upload, inputs=[pdf_in], outputs=[status])
        ask_btn.click(fn=ask_question, inputs=[question, temp], outputs=[output])
        reset_btn.click(fn=reset, inputs=None, outputs=[status])

    return demo


if __name__ == "__main__":
    print("Starting PDF RAG Chatbot — ensure GROQ_API_KEY and GROQ_API_URL are set in your environment.")
    demo = make_demo()
    demo.launch(server_name="0.0.0.0", share=False)
