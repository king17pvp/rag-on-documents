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

loader = Loader("./pdfs/1706.03762v7.pdf")
documents = loader.load()
docs = text_splitter.split_documents(documents)
embedding = get_embeddings(
    "intfloat/multilingual-e5-base",
    {'device': 'cpu'},
    {'normalize_embeddings': False}
)
vector_db = Chroma.from_documents(
    documents=docs,
    embedding=embedding
)
retriever = vector_db.as_retriever()

LLM = LlamaLLM()
prompt = RunnableLambda(
    lambda inputs: get_prompt(
        question=inputs["question"], 
        context=inputs["context"]
    )
)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | LLM | StrOutputParser()
)

USER_QUESTION = "WHAT IS TRANSFORMER"
output = rag_chain.invoke(USER_QUESTION)
answer = output.strip()

print(answer)