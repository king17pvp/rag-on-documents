import torch
import requests
import chainlit as cl
import json 
import warnings 
from transformers import AutoTokenizer
from chainlit.types import AskFileResponse 
from sentence_transformers import SentenceTransformer

from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableMap
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub

from utils import *

warnings.filterwarnings("ignore", category=FutureWarning)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
LLM = LlamaLLM()
embedding = get_embeddings(
    "intfloat/multilingual-e5-base",
    {'device': 'cpu'},
    {'normalize_embeddings': False}
)
def process_input(file: AskFileResponse):
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader
    
    loader = Loader(file.path)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source {i}"
    return docs
    
def get_vector_db(file: AskFileResponse):
    docs = process_input(file)
    cl.user_session.set("docs", docs)
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding
    )
    return vector_db

prompt = RunnableLambda(
    lambda inputs: get_prompt(
        question=inputs["question"], 
        context=inputs["context"]
    )
)
