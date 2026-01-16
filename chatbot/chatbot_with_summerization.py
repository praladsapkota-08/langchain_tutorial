import streamlit as st
import os
from dotenv import load_dotenv

from transformers import pipeline
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --------------------------------------------------
# CHATBOT MODEL (Zephyr – CHAT MODEL, HF Endpoint)
# --------------------------------------------------
chat_endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=300,
    temperature=0.7,
)

chat_llm = ChatHuggingFace(llm=chat_endpoint)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

chat_chain = chat_prompt | chat_llm | StrOutputParser()

# --------------------------------------------------
# SUMMARIZATION MODEL (BART – LOCAL PIPELINE ✅)
# --------------------------------------------------
@st.cache_resource
def load_summarizer():
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=-1   # ✅ force CPU to avoid CUDA errors
    )
    return HuggingFacePipeline(pipeline=summarizer)


summ_llm = load_summarizer()

def summarize_text(text: str):
    return summ_llm.invoke(text)

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Chatbot + Summarizer")
st.title("🎓 AI App: Chatbot & Summarization")

mode = st.radio("Choose mode:", ["Chatbot / Q&A", "Summarization"])

# ---------------- CHATBOT ----------------
if mode == "Chatbot / Q&A":
    user_input = st.text_input("Ask a question:")

    if user_input:
        with st.spinner("Thinking..."):
            response = chat_chain.invoke({"question": user_input})
        st.markdown("### ✅ Answer")
        st.write(response)

# ---------------- SUMMARIZATION ----------------
elif mode == "Summarization":
    text_input = st.text_area("Paste text to summarize:", height=200)

    if text_input:
        if len(text_input.split()) < 50:
            st.warning("Please provide at least 50 words.")
        else:
            with st.spinner("Summarizing..."):
                summary = summarize_text(text_input)
            st.markdown("### ✂️ Summary")
            st.write(summary)
