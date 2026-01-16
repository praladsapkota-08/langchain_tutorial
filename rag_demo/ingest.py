# ingest.py
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

# ✅ Update this path to your actual PDF file
pdf_path = "/home/prahlad-maharaj/langchain/rag_demo/data/book.pdf"

# Check file exists
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF not found at {pdf_path}")

# Load PDF
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Embeddings model (free Hugging Face sentence transformer)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ✅ Extract text + metadata safely
texts = []
metadatas = []
for doc in chunks:
    if doc.page_content and isinstance(doc.page_content, str):
        clean_text = doc.page_content.strip()
        if clean_text != "":
            texts.append(clean_text)
            metadatas.append(doc.metadata)

print(f"✅ Prepared {len(texts)} chunks for embedding")

# Debug: show first few chunks
for i, t in enumerate(texts[:3]):
    print(f"Chunk {i}: type={type(t)}, length={len(t)}")

# Build FAISS vector store
vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
vectorstore.save_local("vectorstore/index")

print("✅ Book ingested and vectorstore saved.")
