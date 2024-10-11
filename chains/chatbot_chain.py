import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

def load_db(file_name):
    # load documents
    loader = PyPDFLoader(file_name)
    documents = loader.load()
    # get params from st.session_state chain_config dict
    chunk_size = int(st.session_state.chain_config["chunk_size"])
    chunk_overlap = int(st.session_state.chain_config["chunk_overlap"])
    k = int(st.session_state.chain_config["k"])
    embeddings = st.session_state.chain_config["embeddings"]
    search_type = st.session_state.chain_config["search_type"]
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever, similarity search
    retriever = db.as_retriever(search_type=search_type, search_kwargs={"k": k})
    return retriever
    

def get_concersational_retrieval_chain(file):
    """
    create a chatbot chain. Memory is managed externally.
    ConversationalRetrievalChain calls the StuffDocumentChain at some point, which collates documents from the retriever.
    This context ist then passed to an LLMChain for generating the final answer.
    Note: You can also change the main prompt in ConversationalRetrievalChain by passing it in via combine_docs_chain_kwargs 
   (from_llm(combine_docs_chain_kwargs={"prompt": your_prompt}))

    :param file: file file name
    return: ConversationalRetrievalChain
    """
    llm_name = st.session_state.chain_config["llm_name"]
    chain_type = st.session_state.chain_config["chain_type"]
    llm_name = st.session_state.chain_config["llm_name"]
    temperature = int(st.session_state.chain_config["temperature"])
    retriever = load_db(file)

    crc = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=temperature), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return crc