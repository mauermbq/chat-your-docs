import streamlit as st
from dotenv import load_dotenv, find_dotenv
import os
import openai
import sys
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

## configs and runtime environment

# environment variables
sys.path.append('../..')
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

# llm model
llm_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_name, temperature=0)

# chroma vector store
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# template and
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
retriever=vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)

# init 
chat_history = []
answer = ""
db_query  = ""
db_response = []

loaded_file = "docs/MachineLearning-Lecture01.pdf" #replace by file handler

def qa(query, chat_history):
    # query the chatbot
    result = qa({"question": query, "chat_history": chat_history})
    return result

def get_sources():
    
    return db_response
