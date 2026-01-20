import streamlit as st
import os
from dotenv import load_dotenv

from transformers import pipeline
fromlangchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --------------------------------------------------
# CHATBOT (Zephyr)
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
# SUMMARIZER (BART)
# --------------------------------------------------
@st.cache_resource
def load_summarizer():
    pipe = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=-1
    )
    return HuggingFacePipeline(pipeline=pipe)

summ_llm = load_summarizer()

def summarize_text(text):
    return summ_llm.invoke(text)

# --------------------------------------------------
# TRANSLATION (English <-> Nepali/Hindi)
# --------------------------------------------------
@st.cache_resource
def load_en_to_ne():
    return pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-en-hi",
        device=-1
    )

@st.cache_resource
def load_ne_to_en():
    return pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-hi-en",
        device=-1
    )

translator_en_ne = load_en_to_ne()
translator_ne_en = load_ne_to_en()

def translate_en_to_ne(text):
    return translator_en_ne(text)[0]["translation_text"]

def translate_ne_to_en(text):
    return translator_ne_en(text)[0]["translation_text"]

# --------------------------------------------------
# NEXT WORD PREDICTION
# --------------------------------------------------
@st.cache_resource
def load_next_word_model():
    # GPT-2 for next word prediction
    return pipeline(
        "text-generation",
        model="gpt2",
        tokenizer="gpt2",
        device=-1,
        pad_token_id=50256  # Avoid warning for GPT2
    )

next_word_model = load_next_word_model()

def predict_next_word(text, max_new_tokens=5):
    """
    Predicts the next word(s) given an input text.
    """
    generated = next_word_model(
        text,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    result = generated[0]["generated_text"]
    # Remove the original input to get only the new words
    next_words = result[len(text):].strip()
    return next_words

# --------------------------------------------------
# UI
# --------------------------------------------------
st.set_page_config(page_title="AI App")
st.title("🎓 AI App: Chat + Summary + Translation + Next Word Prediction")

mode = st.radio("Choose mode:", [
    "Chatbot / Q&A",
    "Summarization",
    "English → Nepali Translation",
    "Nepali → English Translation",
    "Next Word Prediction"
])

# ----------------- Chatbot -----------------
if mode == "Chatbot / Q&A":
    q = st.text_input("Ask something:")
    if q:
        with st.spinner("Thinking..."):
            ans = chat_chain.invoke({"question": q})
        st.write(ans)

# ----------------- Summarization -----------------
elif mode == "Summarization":
    txt = st.text_area("Paste text:", height=200)
    if txt and len(txt.split()) > 50:
        with st.spinner("Summarizing..."):
            st.write(summarize_text(txt))

# ----------------- English -> Nepali -----------------
elif mode == "English → Nepali Translation":
    txt = st.text_area("English text:", height=200)
    if txt:
        with st.spinner("Translating..."):
            st.write(translate_en_to_ne(txt))

# ----------------- Nepali -> English -----------------
elif mode == "Nepali → English Translation":
    txt = st.text_area("Nepali text:", height=200)
    if txt:
        with st.spinner("Translating..."):
            st.write(translate_ne_to_en(txt))

# ----------------- Next Word Prediction -----------------
elif mode == "Next Word Prediction":
    txt = st.text_area("Type a sentence or phrase:", height=150)
    max_tokens = st.slider("How many words to predict?", 1, 20, 5)
    if txt:
        with st.spinner("Predicting next word(s)..."):
            prediction = predict_next_word(txt, max_new_tokens=max_tokens)
        st.write(prediction)
