import streamlit as st
from dotenv import load_dotenv, find_dotenv
import os
import sys
import openai
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


## configs and runtime environment

# environment variables
# to be injected during deployment
sys.path.append('../..')
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.environ['OPENAI_API_KEY']

# relevant configs to configure the whole chain
# tbd note: make this as editable in the UI later
chain_config = {
    "llm_name": "gpt-3.5-turbo",
    "temperature":0,
    "persist_directory": "docs/chroma/",
    "embeddings": OpenAIEmbeddings(),
    "chunk_size": 1000, 
    "chunk_overlap" : 150,
    "k": 3,
    "search_type": "similarity",
    "vectordb": Chroma(persist_directory="docs/chroma/", embedding_function=OpenAIEmbeddings()),
    "chain_type": "stuff", # stuff is the default and uses the StuffDocumentChain (all documents are used to generate the answer)
    "memory": ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
}
# put chain_config into session state 
st.session_state.chain_config = chain_config

# template and
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

loaded_file = "docs/MachineLearning-Lecture01.pdf" #replace by file handler

"""
    
    def convchain(self, query):
        if not query:
            return pn.WidgetBox(pn.Row('User:', pn.pane.Markdown("", width=600)), scroll=True)
        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer'] 
        self.panels.extend([
            pn.Row('User:', pn.pane.Markdown(query, width=600)),
            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=600, style={'background-color': '#F6F6F6'}))
        ])
        inp.value = ''  #clears loading indicator when cleared
        return pn.WidgetBox(*self.panels,scroll=True)

list sources
for doc in self.db_response:
            rlist.append(pn.Row(pn.pane.Str(doc)))        
        
"""



def qa(query, chat_history):
    # query the chatbot
    result = qa({"question": query, "chat_history": chat_history})
    return result

# create st.from for prompt
st.title("Chatbot")
st.write("Ask me anything about the Document you provided")
query = st.text_input("Enter your question here")
if st.button("Ask"):
    result = qa(query, chat_history)
    st.write(result)
    chat_history = result["chat_history"]
    st.session_state.chat_history = chat_history