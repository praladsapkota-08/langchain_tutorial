# app.py
import streamlit as st
import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Load vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vectorstore/index", embeddings)
retriever = vectorstore.as_retriever()

# Base LLM (Zephyr for free chatbot/Q&A)
endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=500,
    temperature=0.7,
)
llm = ChatHuggingFace(llm=endpoint)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the retrieved context to answer."),
    ("human", "Question: {question}\n\nContext: {context}")
])
chain = prompt | llm | StrOutputParser()

# Streamlit UI
st.title("📚 RAG Chatbot (600-page Book)")
user_input = st.text_input("Ask a question about the book:")

if user_input:
    docs = retriever.get_relevant_documents(user_input)
    context = "\n\n".join([d.page_content for d in docs[:3]])  # top 3 chunks
    response = chain.invoke({"question": user_input, "context": context})
    st.write(response)
