import streamlit as st
import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Base HF endpoint
endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=500,
    temperature=0.7,
)

# 🔑 Convert endpoint → chat model
llm = ChatHuggingFace(llm=endpoint)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}")
    ]
)

chain = prompt | llm | StrOutputParser()

st.title("LangChain + HuggingFace + Streamlit")

user_input = st.text_input("Search the topic you want")

if user_input:
    response = chain.invoke({"question": user_input})
    st.write(response)
