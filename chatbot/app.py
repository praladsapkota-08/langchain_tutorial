from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

langchain_api_key = os.getenv("LANGCHAIN_API_KEY") 
openai_api_key = os.getenv("OPENAI_API_KEY") 
project_name = os.getenv("LANGSMITH_PROJECT")

os.environ["LANGCHAIN_TRACING_V2"] = "true"

## prompt Template

prompt =  ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question: {question}")
    ]
)

st.title('Langchain Demo With LLAMA2 API')
input_txt = st.text_input('search the topic you want')

llm = Ollama(model = 'llama2')
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_txt:
    st.write(chain.invoke({'question':input_txt}))