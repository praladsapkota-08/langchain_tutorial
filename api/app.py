from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title = "Langchain serve",
    version = "1.0",
    description = "A simple API Serve"

)

llm = Ollama(model = "llama2")

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")

add_routes(
    app,
    prompt1|llm,
    path="/essay"

)

if __name__== "__main__":
    uvicorn.run(app, host="localhost",port = 8000)