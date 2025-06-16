import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings

# === your existing functions ===
def load_pdf(file_path):
    return PyPDFLoader(file_path).load()

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def build_custom_prompt(context, question):
    return f"""Use the following document context to answer the question:

{context}

Question: {question}
Answer:"""

def ask_question(query, retriever, llm):
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs[:3])
    prompt = build_custom_prompt(context, query)
    return llm.invoke(prompt), docs

# === setup outside Gradio ===
pdf_file = "/mnt/d/llm-devs/paper_qa/papers/9 Submission.pdf"
docs = load_pdf(pdf_file)
chunks = split_documents(docs)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()
llm = OllamaLLM(model="gemma3:4b", temperature=0.1)

# === Gradio UI ===
with gr.Blocks() as demo:
    mode = gr.Radio(["With History", "Single Turn"], value="With History",
                    label="Chat Mode")

    chatbot = gr.Chatbot(type="messages", show_label=False)
    txt = gr.Textbox(placeholder="Ask a question...", label="")

    def gr_chat(message, history, mode):
        answer, docs = ask_question(message, retriever, llm)
        if mode == "With History":
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer}
            ]
        else:
            history = [{"role": "assistant", "content": answer}]
        return history, history

    txt.submit(gr_chat, inputs=[txt, chatbot, mode], outputs=[chatbot, chatbot])
    txt.submit(lambda: "", None, txt)  # clear input

demo.launch()
