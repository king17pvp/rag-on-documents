import PyPDF2
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
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableMap
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub

from utils import *
PERSIST_DIRECTORY = "vectorstore/" 
WELCOME_MESSAGE = f"""Welcome to the PDF Q&A Program, to get started, please do the following:
1. Upload a PDF or Text file
2. Ask a question about the file
"""
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

@cl.on_chat_start
async def on_chat_start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=WELCOME_MESSAGE,
            accept=["text/plain", "application/pdf"],
            max_size_mb=10,
            timeout=180
        ).send()
    file = files[0]
    message = cl.Message(
        content=f"Processing {file.name}...",
    )
    await message.send()
    pdf = PyPDF2.PdfReader(file.path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
        

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(pdf_text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    vector_db = await cl.make_async(Chroma.from_texts)(
        texts, embedding, metadatas=metadatas, persist_directory=PERSIST_DIRECTORY
    )
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_history=message_history,
        return_messages=True
    )
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4
        }
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=LLM,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True       
    )
    
    message.content = f"File {file.name} processed successfully, now you can ask question about the PDF file"

    await message.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def on_message(message: cl.Message):
    print(message.content)
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]
    text_elements = []
    
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"Source {source_idx}"
            text_elements.append(
                cl.Text(
                    content=source_doc.page_content,
                    name=source_name
                )
            )
    await cl.Message(content=answer).send()