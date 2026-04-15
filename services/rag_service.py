from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ==============================
# 🔑 CONFIG
# ==============================
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load embeddings ONCE (important for speed)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# ==============================
# 📄 EXTRACT TEXT
# ==============================
def process_pdf(file_bytes):
    from io import BytesIO

    pdf_reader = PdfReader(BytesIO(file_bytes))
    text = ""

    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content

    return text


# ==============================
# 🔪 SPLIT TEXT
# ==============================
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_text(text)


# ==============================
# 🧠 CREATE VECTOR STORE
# ==============================
def create_vector_store(chunks):
    return FAISS.from_texts(chunks, embeddings)


# ==============================
# 🚀 FULL PIPELINE
# ==============================
def process_and_store(file_bytes):
    text = process_pdf(file_bytes)
    chunks = split_text(text)
    return create_vector_store(chunks)


# ==============================
# 🤖 ASK QUESTION
# ==============================
def ask_question(vector_store, question):
    docs = vector_store.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a financial analyst AI.

    Answer ONLY from context.
    If not found, say "Not found in document".

    Context:
    {context}

    Question:
    {question}
    """

    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(prompt)

    return response.text