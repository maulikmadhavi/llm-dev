import os
import time
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow
import gradio as gr

# LangChain RAG dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# ------------------- Configuration ------------------- #
pdf_path = "/mnt/c/Users/mauli/Downloads/2503.19903v1.pdf"
vector_root = "/mnt/d/llm-devs/langchain/vector_stores"
nvidia_model = "nvidia/llama-3.3-nemotron-super-49b-v1"
embedding_model_name = "BAAI/bge-large-en-v1.5"  # or 'large'
reranker_model_name = "BAAI/bge-reranker-large"  # or 'large'
chunk_size = 1000
chunk_overlap = 200

# Extract base name for cache
pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
vector_store_path = os.path.join(vector_root, f"{pdf_name}_{embedding_model_name.replace('/', '_')}")

# ------------------- Init Traceloop ------------------- #
Traceloop.init(app_name="pdf_rag_chat", api_key=os.getenv("TRACELOOP_API_KEY"))

# ------------------- Build or Load Vector DB ------------------- #
def load_vector_db():
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

    if os.path.exists(vector_store_path):
        print(f"✅ Loading cached vector store from {vector_store_path}")
        try:
            return FAISS.load_local(vector_store_path, embeddings=embedding, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"❌ Error loading cached vector store: {e}")
            print("🔄 Rebuilding vector store with current embedding model...")
            # If loading fails (usually due to dimension mismatch), continue to rebuild
            
    # Code to build new vector store
    print("📄 Loading PDF...")
    docs = PyPDFLoader(pdf_path).load()

    print("✂️ Splitting into token chunks...")
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        disallowed_special=()
    )
    chunks = splitter.split_documents(docs)

    print("🔎 Embedding and indexing...")
    vectordb = FAISS.from_documents(chunks, embedding)
    os.makedirs(vector_root, exist_ok=True)
    vectordb.save_local(vector_store_path)
    print(f"💾 Vector store saved at {vector_store_path}")
    return vectordb

# Build everything once
vectordb = load_vector_db()
retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 20, "verbose": True})
llm = ChatNVIDIA(model=nvidia_model)

# =============================================================================
# RAG SYSTEM FUNCTIONS
# =============================================================================
def build_custom_prompt(context, question):
    """Create a prompt that instructs the LLM to use the provided context to answer the question."""
    return f"""Use the following document context to answer the question:

{context}

Question: {question}
Answer:"""


# Load the reranker model
reranker = CrossEncoder(reranker_model_name)  # or 'large'

@workflow(name="rerank_documents")
def rerank(query, docs, top_k=5):    
    pairs = [[query, doc.page_content] for doc in docs]
    
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

@workflow(name="rag_ask_question")
def ask_question(query, retriever, llm, top_k=5):
    """
    Core RAG function that:
    1. Retrieves relevant document chunks
    2. Builds context from top chunks
    3. Creates a prompt with context and question
    4. Gets answer from LLM
    """
    docs = retriever.invoke(query)
    print(f"🔍 Retrieved {len(docs)} chunks from retriever.")
    # Rerank the documents using the reranker model
    docs = rerank(query, docs, top_k=top_k)  # Adjust top_k as needed    
    context = "\n\n".join(d.page_content for d in docs)
    print(f"🔍 Found {len(docs)} relevant chunks, using top {top_k} for context.")
    prompt = build_custom_prompt(context, query)
    return llm.invoke(prompt), docs



# ------------------- RAG Chat Workflow ------------------- #
@workflow(name="chat_with_rag")
def do_chat(in_message: str):
    start_time = time.time()
    answer, docs = ask_question(in_message, retriever, llm)
    elapsed_time = time.time() - start_time
    print(f"🕐 Took {elapsed_time:.2f}s")
    return answer,  docs

# ------------------- Example Usage ------------------- #
if __name__ == "__main__":
    def chat_wrapper(question):
        answer, docs = do_chat(question)
        # answer is a LangChain LLMResult or string; ensure string
        if hasattr(answer, "content"):
            answer_text = answer.content
        else:
            answer_text = str(answer)
        return answer_text

    gr.Interface(
        fn=chat_wrapper,
        inputs=gr.Textbox(label="Enter your question about the PDF"),
        outputs=gr.Markdown(label="Answer"),
        title="PDF RAG Chat",
        description="Ask questions about the PDF document using RAG. (Markdown supported in answers.)",
    ).launch()