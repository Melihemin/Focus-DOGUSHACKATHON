import os
import json
import re
import hashlib
import logging
import urllib.request
import urllib.error
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import List, Tuple
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
from database.models import Courses

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "60"))

def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

ENABLE_QUERY_REWRITE = env_flag("ENABLE_QUERY_REWRITE", True)
ENABLE_HYBRID_RETRIEVAL = env_flag("ENABLE_HYBRID_RETRIEVAL", True)
ENABLE_RERANK = env_flag("ENABLE_RERANK", True)
ENABLE_COMPRESSION = env_flag("ENABLE_COMPRESSION", True)
ENABLE_SUFFICIENCY_CHECK = env_flag("ENABLE_SUFFICIENCY_CHECK", True)

DENSE_K = int(os.getenv("DENSE_K", "8"))
SPARSE_K = int(os.getenv("SPARSE_K", "12"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "4"))
FINAL_TOP_N = int(os.getenv("FINAL_TOP_N", "4"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CACHE_MAX = int(os.getenv("RERANK_CACHE_MAX", "128"))

DEFAULT_OLLAMA_OPTIONS = {
    "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.2")),
    "top_p": float(os.getenv("OLLAMA_TOP_P", "0.9")),
    "repeat_penalty": float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.1")),
    "num_predict": int(os.getenv("OLLAMA_MAX_TOKENS", "512")),
    "seed": int(os.getenv("OLLAMA_SEED", "42")),
}

REWRITE_OLLAMA_OPTIONS = {
    "temperature": float(os.getenv("OLLAMA_REWRITE_TEMPERATURE", "0.1")),
    "top_p": float(os.getenv("OLLAMA_REWRITE_TOP_P", "0.8")),
    "repeat_penalty": float(os.getenv("OLLAMA_REWRITE_REPEAT_PENALTY", "1.0")),
    "num_predict": int(os.getenv("OLLAMA_REWRITE_MAX_TOKENS", "64")),
    "seed": int(os.getenv("OLLAMA_SEED", "42")),
}

# --- MODEL SETTINGS ---

# 1. LLM (Response Generator): Local Ollama API
logging.info("Initializing local Ollama model...")
logging.info("Ollama model: %s", OLLAMA_MODEL)
logging.info("Generation options: %s", DEFAULT_OLLAMA_OPTIONS)

def call_ollama_chat(prompt: str, options: dict | None = None, retries: int = 2) -> str:
    options = options or DEFAULT_OLLAMA_OPTIONS
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "keep_alive": "10m",
    }
    if options:
        payload["options"] = options
    data = json.dumps(payload).encode("utf-8")

    for attempt in range(retries + 1):
        request = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            message = response_data.get("message", {}).get("content")
            if not message:
                message = response_data.get("response")
            if not message:
                logging.error("Ollama chat empty response keys: %s", list(response_data.keys()))
                raise ValueError("Empty response from Ollama API.")
            return message
        except (urllib.error.URLError, ValueError) as error:
            logging.error("Ollama request failed (attempt %s/%s): %s", attempt + 1, retries + 1, error)
            if attempt >= retries:
                break
            time.sleep(0.4)

    generate_payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": options,
    }
    generate_data = json.dumps(generate_payload).encode("utf-8")
    generate_request = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=generate_data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(generate_request, timeout=OLLAMA_TIMEOUT) as response:
            response_data = json.loads(response.read().decode("utf-8"))
        message = response_data.get("response")
        if not message:
            raise ValueError("Empty response from Ollama generate API.")
        return message
    except (urllib.error.URLError, ValueError) as error:
        logging.error("Ollama generate request failed: %s", error)
        raise

# 2. Embeddings (Vectorizer): Local HuggingFace
logging.info("Loading HuggingFace embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- PROMPT ---
rag_prompt_template = ChatPromptTemplate.from_template(
    "You are an education specialist for learners with ADHD.\n\n"
    "Explain the requested topic using ONLY the context below:\n\n"
    "IMPORTANT RULES:\n"
    "1. Use only the provided context; do not add external knowledge.\n"
    "2. If PDF sources exist, prioritize their information.\n"
    "3. Focus on one idea per paragraph.\n"
    "4. Use headings and subheadings for structure.\n"
    "5. Include concrete examples and analogies from daily life.\n"
    "6. Provide step-by-step explanations suitable for ADHD learners.\n"
    "7. End with a brief summary.\n\n"
    "===== CONTEXT =====\n"
    "{context}\n\n"
    "===== QUESTION =====\n"
    "{istek}\n\n"
    "===== EXPLANATION =====\n"
    "Use the following format in your response:\n"
    "- Definition\n"
    "- Core idea\n"
    "- Step-by-step logic\n"
    "- Example\n"
    "- Key takeaways\n"
)

retriever_cache = None
vector_store_cache = None
docs_cache: List[Document] = []
bm25_retriever_cache = None
reranker_cache: CrossEncoder | None = None
rerank_cache: dict[Tuple[str, Tuple[str, ...]], List[str]] = {}
FAISS_INDEX_PATH = "faiss_index"
PDF_FOLDER_PATH = "pdfs"  # PDFs should be in this folder

# --- TEXT SPLITTING SETTINGS ---
# Split text into ~300-500 token chunks with light overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1600,
    chunk_overlap=320,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def load_pdfs_from_folder():
    """
    Load all PDF files from the pdfs/ folder and apply chunking.
    """
    docs = []
    if not os.path.exists(PDF_FOLDER_PATH):
        logging.warning("'%s' folder not found. PDFs could not be loaded.", PDF_FOLDER_PATH)
        logging.warning("Create a 'pdfs' folder at %s and place PDFs there.", os.path.abspath(PDF_FOLDER_PATH))
        return docs
    
    try:
        pdf_files = [f for f in os.listdir(PDF_FOLDER_PATH) if f.endswith('.pdf')]
        logging.info("Opened '%s' folder.", PDF_FOLDER_PATH)
        
        if len(pdf_files) == 0:
            logging.warning("'%s' folder is empty. No PDF files found.", PDF_FOLDER_PATH)
            return docs
        
        logging.info("Found %s PDF files: %s", len(pdf_files), pdf_files)
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(PDF_FOLDER_PATH, pdf_file)
            try:
                logging.info("Loading PDF: %s", pdf_file)
                loader = PyPDFLoader(pdf_path)
                pdf_docs = loader.load()
                logging.info("   %s loaded successfully. Pages: %s", pdf_file, len(pdf_docs))
                
                # Split PDF pages into chunks
                split_docs = text_splitter.split_documents(pdf_docs)
                logging.info("   %s split into %s chunks", pdf_file, len(split_docs))
                
                # Add metadata
                for doc in split_docs:
                    doc.metadata["source"] = pdf_file
                    doc.metadata["type"] = "pdf"
                
                docs.extend(split_docs)
            except Exception as pdf_error:
                logging.error("   Failed to load %s: %s", pdf_file, pdf_error)
        
            logging.info("Loaded %s chunks from PDFs.", len(docs))
    except Exception as e:
            logging.error("Error reading PDF folder: %s", e)
    
    return docs

def create_and_save_retriever_from_db(db: Session):
    """
    Read documents from the database and PDFs, apply chunking, and create a retriever.
    """
    logging.info("=" * 60)
    logging.info("Building retriever (database + PDFs)...")
    logging.info("=" * 60)
    docs = []
    
    # 1. Load and chunk courses from the database
    logging.info("\nStep 1: Loading courses from database...")
    all_courses = db.query(Courses).all()
    logging.info("   Total courses found: %s", len(all_courses))
    
    for course in all_courses:
        if course.content:
            course_doc = Document(
                page_content=course.content,
                metadata={"source": "database", "course_id": course.id, "title": course.title, "type": "course"}
            )
            # Split course content
            split_course_docs = text_splitter.split_documents([course_doc])
            docs.extend(split_course_docs)
            logging.info("   %s: %s chunks", course.title, len(split_course_docs))

    logging.info("   Loaded %s chunks from database.\n", len(docs))
    
    # 2. Load documents from PDFs
    logging.info("Step 2: Loading PDFs...")
    pdf_docs = load_pdfs_from_folder()
    docs.extend(pdf_docs)
    logging.info("   Total chunks collected: %s\n", len(docs))
    
    if not docs:
        logging.error("No documents found. Check the database and PDF folder.")
        return None

    try:
        logging.info("Step 3: Vectorization started...")
        logging.info("   Vectorizing %s chunks (this may take time)...", len(docs))
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        logging.info("=" * 60)
        logging.info("Vector store created and saved to disk successfully.")
        logging.info("Total chunks: %s", len(docs))
        chunk_size = getattr(text_splitter, "chunk_size", getattr(text_splitter, "_chunk_size", "unknown"))
        chunk_overlap = getattr(text_splitter, "chunk_overlap", getattr(text_splitter, "_chunk_overlap", "unknown"))
        logging.info("Chunk size: %s characters", chunk_size)
        logging.info("Overlap: %s characters", chunk_overlap)
        logging.info("Path: %s", FAISS_INDEX_PATH)
        logging.info("=" * 60 + "\n")
        retriever = vector_store.as_retriever(search_kwargs={"k": DENSE_K})
        initialize_sparse_retriever(docs)
        set_vector_store_cache(vector_store)
        return retriever
    except Exception as e:
        logging.error("Error creating vector store: %s", e)
        return None

def load_retriever_from_disk():
    """
    Load the FAISS index saved on disk.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        logging.warning(f"No saved index found at '{FAISS_INDEX_PATH}'.")
        return None
    
    try:
        logging.info("Loading retriever from '%s'...", FAISS_INDEX_PATH)
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        logging.info("Retriever loaded from disk successfully.")
        set_vector_store_cache(vector_store)
        initialize_sparse_retriever(get_docs_from_vector_store(vector_store))
        return vector_store.as_retriever(search_kwargs={"k": DENSE_K})
    except Exception as e:
        logging.error("Error loading retriever from disk: %s", e)
        return None

def format_docs(docs: List[Document]) -> str:
    """Combine retriever documents into a single text and log details."""
    formatted = ""
    logging.info("\nRetriever returned %s documents:", len(docs))
    
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        doc_type = doc.metadata.get("type", "unknown")
        logging.info("   %s. %s (%s) - %s characters", i, source, doc_type, len(doc.page_content))
        formatted += f"\n--- Source: {source} ({doc_type}) ---\n{doc.page_content}"
    
    logging.info("")
    return formatted

def initialize_retriever(db: Session):
    """
    Load or create the retriever at application startup.
    """
    global retriever_cache
    retriever_cache = load_retriever_from_disk()
    if retriever_cache is None:
        logging.info("Failed to load retriever from disk. Creating a new one from database...")
        retriever_cache = create_and_save_retriever_from_db(db)
    return retriever_cache

def set_vector_store_cache(vector_store: FAISS):
    global vector_store_cache, docs_cache
    vector_store_cache = vector_store
    docs_cache = get_docs_from_vector_store(vector_store)

def get_docs_from_vector_store(vector_store: FAISS) -> List[Document]:
    try:
        docstore = getattr(vector_store, "docstore", None)
        if docstore and hasattr(docstore, "_dict"):
            return list(docstore._dict.values())
    except Exception as e:
        logging.error("Failed to extract documents from vector store: %s", e)
    return []

def initialize_sparse_retriever(docs: List[Document]):
    global bm25_retriever_cache
    if not docs:
        bm25_retriever_cache = None
        return
    try:
        bm25_retriever_cache = BM25Retriever.from_documents(docs)
        bm25_retriever_cache.k = SPARSE_K
    except Exception as e:
        logging.warning("BM25 retriever disabled: %s", e)
        bm25_retriever_cache = None

def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()

@lru_cache(maxsize=256)
def rewrite_query(query: str) -> str:
    if not ENABLE_QUERY_REWRITE:
        return query
    cleaned = normalize_query(query)
    if not cleaned:
        return query
    prompt = (
        "Rewrite the user question to improve retrieval. "
        "Clarify intent, expand abbreviations, and normalize terminology. "
        "Preserve original meaning. Output only the rewritten question.\n\n"
        f"Question: {cleaned}\n"
        "Rewritten:"
    )
    try:
        rewritten = call_ollama_chat(prompt, options=REWRITE_OLLAMA_OPTIONS).strip()
        if rewritten:
            return rewritten
    except Exception as e:
        logging.warning("Query rewrite failed: %s", e)
    return query

def doc_id(doc: Document) -> str:
    source = doc.metadata.get("source", "")
    doc_type = doc.metadata.get("type", "")
    content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
    return f"{source}:{doc_type}:{content_hash}"

def rrf_fuse(dense_docs: List[Document], sparse_docs: List[Document], rrf_k: int = 60) -> List[Document]:
    scores = {}
    doc_map = {}
    for rank, doc in enumerate(dense_docs, start=1):
        did = doc_id(doc)
        scores[did] = scores.get(did, 0.0) + 1.0 / (rrf_k + rank)
        doc_map[did] = doc
    for rank, doc in enumerate(sparse_docs, start=1):
        did = doc_id(doc)
        scores[did] = scores.get(did, 0.0) + 1.0 / (rrf_k + rank)
        doc_map[did] = doc
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_map[did] for did, _score in ranked]

def hybrid_retrieve(query: str) -> List[Document]:
    dense_docs: List[Document] = []
    sparse_docs: List[Document] = []

    if retriever_cache is None:
        return []

    def run_retriever(retriever):
        if hasattr(retriever, "invoke"):
            return retriever.invoke(query)
        return retriever.get_relevant_documents(query)

    if ENABLE_HYBRID_RETRIEVAL and bm25_retriever_cache is not None:
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(run_retriever, retriever_cache)
            sparse_future = executor.submit(run_retriever, bm25_retriever_cache)
            dense_docs = dense_future.result()
            sparse_docs = sparse_future.result()
    else:
        dense_docs = run_retriever(retriever_cache)

    logging.info("Dense hits: %s, Sparse hits: %s", len(dense_docs), len(sparse_docs))

    if ENABLE_HYBRID_RETRIEVAL and sparse_docs:
        return rrf_fuse(dense_docs, sparse_docs)
    return dense_docs

def get_reranker() -> CrossEncoder | None:
    global reranker_cache
    if not ENABLE_RERANK:
        return None
    if reranker_cache is None:
        logging.info("Loading reranker model: %s", RERANK_MODEL_NAME)
        reranker_cache = CrossEncoder(RERANK_MODEL_NAME)
    return reranker_cache

def rerank_docs(query: str, docs: List[Document], top_n: int) -> List[Document]:
    reranker = get_reranker()
    if reranker is None or not docs:
        return docs
    cache_key = (query, tuple(doc_id(doc) for doc in docs))
    if cache_key in rerank_cache:
        ordered_ids = rerank_cache[cache_key]
        doc_map = {doc_id(doc): doc for doc in docs}
        return [doc_map[doc_id] for doc_id in ordered_ids if doc_id in doc_map][:top_n]

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs, batch_size=8)
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda item: item[1], reverse=True)
    ordered_ids = [doc_id(doc) for doc, _score in scored_docs]
    if len(rerank_cache) >= RERANK_CACHE_MAX:
        rerank_cache.clear()
    rerank_cache[cache_key] = ordered_ids

    top_docs = [doc for doc, _score in scored_docs[:top_n]]
    score_values = [float(score) for _doc, score in scored_docs]
    if score_values:
        logging.info("Rerank score min: %.4f max: %.4f", min(score_values), max(score_values))
    return top_docs

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())

def context_sufficient(query: str, docs: List[Document], min_overlap: float = 0.15) -> bool:
    if not docs:
        return False
    query_terms = set(tokenize(query))
    if not query_terms:
        return True
    best_overlap = 0.0
    for doc in docs:
        doc_terms = set(tokenize(doc.page_content))
        if not doc_terms:
            continue
        overlap = len(query_terms & doc_terms) / max(len(query_terms), 1)
        best_overlap = max(best_overlap, overlap)
    logging.info("Context overlap ratio: %.3f", best_overlap)
    return best_overlap >= min_overlap

def compress_context(docs: List[Document], query: str) -> str:
    if not docs:
        return ""
    query_terms = set(tokenize(query))
    compressed_parts: List[str] = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        doc_type = doc.metadata.get("type", "unknown")
        sentences = re.split(r"(?<=[.!?])\s+", doc.page_content.strip())
        scored_sentences: List[Tuple[float, str]] = []
        for sentence in sentences:
            sentence_terms = set(tokenize(sentence))
            if not sentence_terms:
                continue
            overlap = len(query_terms & sentence_terms)
            score = overlap / max(len(sentence_terms), 1)
            if score > 0:
                scored_sentences.append((score, sentence))
        scored_sentences.sort(key=lambda item: item[0], reverse=True)
        selected = [s for _score, s in scored_sentences[:3]]
        if not selected and sentences:
            selected = sentences[:1]
        if selected:
            compressed_parts.append(f"--- Source: {source} ({doc_type}) ---\n" + " ".join(selected))

    compressed = "\n\n".join(compressed_parts)
    if len(compressed) > MAX_CONTEXT_CHARS:
        compressed = compressed[:MAX_CONTEXT_CHARS]
    logging.info("Compressed context length: %s characters", len(compressed))
    return compressed

def generate_rag_answer(request_text: str, db: Session = None) -> str:
    """
    Generate an explanation based on the request.
    If the retriever is not ready and a db session is provided, it attempts to build it.
    """
    global retriever_cache
    
    # 1. Check: If cache is empty, try to populate it
    if retriever_cache is None:
        logging.warning("Retriever cache is empty. Attempting to load...")
        if db:
            initialize_retriever(db)
        else:
            return "Error: Retriever could not be initialized because no database session was provided."

    # 2. Check: If still empty
    if retriever_cache is None:
        logging.error("Retriever could not be initialized.")
        return "Error: Knowledge base could not be created."

    normalized_query = normalize_query(request_text)
    if not normalized_query:
        return "Please provide a question."

    rewritten_query = rewrite_query(normalized_query)
    retrieval_query = rewritten_query or normalized_query
    logging.info("Query rewrite enabled: %s", ENABLE_QUERY_REWRITE)
    if rewritten_query != normalized_query:
        logging.info("Rewritten query: %s", rewritten_query)

    docs = hybrid_retrieve(retrieval_query)
    if ENABLE_RERANK:
        docs = rerank_docs(retrieval_query, docs, RERANK_TOP_N)

    docs = docs[:FINAL_TOP_N]
    logging.info("Selected docs for context: %s", len(docs))

    if ENABLE_SUFFICIENCY_CHECK and not context_sufficient(retrieval_query, docs):
        return "Insufficient context to answer accurately. Please provide more details or upload relevant materials."

    if ENABLE_COMPRESSION:
        context_text = compress_context(docs, retrieval_query)
    else:
        context_text = format_docs(docs)
        logging.info("Context length: %s characters", len(context_text))
    prompt_text = rag_prompt_template.format(context=context_text, istek=normalized_query)

    try:
        response_text = call_ollama_chat(prompt_text, options=DEFAULT_OLLAMA_OPTIONS)
        return response_text
    except Exception as e:
        logging.error("Error generating LLM response: %s", e)
        return "Sorry, an error occurred while generating a response."


def evaluate_user_performance(dashboard_data: dict) -> dict:
    """
    Evaluate user performance with AI and provide recommendations.
    Analyze dashboard data and provide detailed feedback.
    """
    
    eye_tracking = dashboard_data.get("eye_tracking", {})
    focus = eye_tracking.get("focus_breakdown", {})
    
    # Prepare data
    evaluation_prompt = f"""
    Evaluate a user's performance on an e-learning platform and provide recommendations. 
    Format your response in Markdown.
    
    ## User Performance Data
    
    - **Total Sessions**: {eye_tracking.get('total_sessions', 0)}
    - **Total Study Hours**: {eye_tracking.get('total_duration_hours', 0)} hours
    - **Average Session Duration**: {eye_tracking.get('avg_session_duration_minutes', 0)} minutes
    - **Total Tab Switches**: {eye_tracking.get('total_tab_switches', 0)} times
    
    ## Focus Statistics
    
    - **Good Focus**: {focus.get('good', {}).get('percentage', 0)}% ({focus.get('good', {}).get('seconds', 0)} seconds)
    - **Distracted**: {focus.get('warning', {}).get('percentage', 0)}% ({focus.get('warning', {}).get('seconds', 0)} seconds)
    - **Lost Focus**: {focus.get('alert', {}).get('percentage', 0)}% ({focus.get('alert', {}).get('seconds', 0)} seconds)
    
    ## Evaluation Criteria
    
    Please provide an evaluation including the following sections:
    
    ### Performance Analysis
    Identify the user's strengths and weaknesses. Reference specific numbers.
    
    ### Overall Assessment
    Determine performance level: **Excellent** / **Good** / **Needs Improvement**
    
    ### Specific Recommendations
    Provide 3-5 concrete, actionable recommendations for improving focus and attention.
    
    ### Motivation & Encouragement
    Use a positive and constructive tone. Encourage the user to continue.
    
    ### Next Steps
    List the top 3 priority actions:
    - Number each action
    - Explain each action in 1-2 sentences
    
    ## Important Guidelines
    
    - Use proper Markdown formatting (headings, lists, bold, italic)
    - Use clear structure and hierarchy
    - Language: Professional and encouraging
    - Keep response concise and actionable
    """
    
    try:
        response_text = call_ollama_chat(evaluation_prompt, options=DEFAULT_OLLAMA_OPTIONS)
        
        return {
            "success": True,
            "evaluation": response_text
        }
    except Exception as e:
        logging.error(f"Error during AI evaluation: {e}")
        return {
            "success": False,
            "evaluation": "Evaluation could not be retrieved. Please try again later."
        }