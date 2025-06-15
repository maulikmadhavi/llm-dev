from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# Loaders
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def load_docx(file_path):
    loader = Docx2txtLoader(file_path)
    return loader.load()

# Splitter
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Custom prompt builder
def build_custom_prompt(context, question):
    return f"""Use the following document context to answer the question:

{context}

Question: {question}
Answer:"""

if __name__ == "__main__":
    pdf_file = "/mnt/d/llm-devs/paper_qa/papers/9 Submission.pdf"
    
    documents = load_pdf(pdf_file)
    splited_docs = split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(splited_docs, embeddings)

    retriever = vectorstore.as_retriever()
    llm = Ollama(model="qwen3:0.6b", temperature=0.1)

    # Ask a question
    query = "What is in the introduction section?"

    # Get top-k relevant chunks
    retrieved_docs = retriever.get_relevant_documents(query)
    top_k_context = "\n\n".join(doc.page_content for doc in retrieved_docs[:3])

    # Build prompt and get answer
    prompt = build_custom_prompt(top_k_context, query)
    answer = llm.invoke(prompt)

    print("Question:", query)
    print("\nAnswer:\n", answer)

    print("\n--- Retrieved Chunks ---")
    for i, doc in enumerate(retrieved_docs[:3]):
        print(f"\n[Chunk {i+1}]\n{doc.page_content}")
