import os
import time
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

# LangChain RAG dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# ------------------- Configuration ------------------- #
pdf_path = "/mnt/c/Users/mauli/Downloads/2503.19903v1.pdf"
vector_root = "/mnt/d/llm-devs/langchain/vector_stores"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
chunk_size = 1000
chunk_overlap = 200

# Extract base name for cache
pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
vector_store_path = os.path.join(vector_root, pdf_name)

# ------------------- Init Traceloop ------------------- #
Traceloop.init(app_name="pdf_rag_chat", api_key=os.getenv("TRACELOOP_API_KEY"))

# ------------------- Build or Load Vector DB ------------------- #
def load_vector_db():
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

    if os.path.exists(vector_store_path):
        print(f"✅ Loading cached vector store from {vector_store_path}")
        return FAISS.load_local(vector_store_path, embeddings=embedding, allow_dangerous_deserialization=True)

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
retriever = vectordb.as_retriever()
llm = ChatNVIDIA(model="nvidia/llama-3.3-nemotron-super-49b-v1")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# ------------------- RAG Chat Workflow ------------------- #
@workflow(name="chat_with_rag")
def do_chat(in_message: str):
    start_time = time.time()
    result = qa_chain.invoke({"query": in_message})
    print(f"🕐 Took {time.time() - start_time:.2f}s")
    return result["result"]

# ------------------- Example Usage ------------------- #
if __name__ == "__main__":
    while True:
        question = input("❓ Ask a question about the PDF (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        if not question.strip():
            print("Please enter a valid question.")
            continue
        
        # Call the workflow
        response = do_chat(question)    
        print("🔍 Query:", question)
        print("💬 Answer:", do_chat(question))