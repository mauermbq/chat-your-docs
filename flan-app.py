import streamlit as streamlit
from InstructorEmbedding import INSTRUCTOR
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.embeddings import HuggingFaceEmbeddings


EMB_INSTRUCTOR_XL = "hkunlp/instructor-xl"
EMB_SBERT_MPNET_BASE = "sentence-transformers/all-mpnet-base-v2"
LLM_FLAN_T5_XXL = "google/flan-t5-xxl"
LLM_FLAN_T5_XL = "google/flan-t5-xl"
LLM_FASTCHAT_T5_XL = "lmsys/fastchat-t5-3b-v1.0"
LLM_FLAN_T5_SMALL = "google/flan-t5-small"
LLM_FLAN_T5_BASE = "google/flan-t5-base"
LLM_FLAN_T5_LARGE = "google/flan-t5-large"
LLM_FALCON_SMALL = "tiiuae/falcon-7b-instruct"

config = {"persist_directory":None,
          "load_in_8bit":False,
          "embedding" : EMB_SBERT_MPNET_BASE,
          "llm":LLM_FLAN_T5_BASE,
          }

def create_sbert_mpnet():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbeddings(model_name=EMB_SBERT_MPNET_BASE, model_kwargs={"device": device})


def create_flan_t5_base(load_in_8bit=False):
        
        model="google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model)
        return pipeline(
            task="text2text-generation",
            model=model,
            tokenizer = tokenizer,
            max_new_tokens=100,
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )



if config["embedding"] == EMB_SBERT_MPNET_BASE:
    embedding = create_sbert_mpnet()
load_in_8bit = config["load_in_8bit"]
if config["llm"] == LLM_FLAN_T5_BASE:
    llm = create_flan_t5_base(load_in_8bit=load_in_8bit)