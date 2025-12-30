import requests
import streamlit as st

def get_ollama_response(input_text):
    response = requests.post(
    "http://localhost:8000/essay/invoke",
    json = {"input":{'topic':input_text}}
    )
    try: 
        data = response.json() 
        return data.get("output", f"Unexpected response: {data}") 
    
    except ValueError: 
        return f"Non‑JSON response: {response.text}"
    
st.title('langchain demo with llama2 api')
input_text = st.text_input('write an essay on')

if input_text:
    st.write(get_ollama_response(input_text))


