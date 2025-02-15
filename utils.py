import os 
import requests
import json
from huggingface_hub import InferenceClient
from typing import Any, List, Mapping, Optional
from pydantic import Extra

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from chainlit.types import AskFileResponse
class LlamaLLM(LLM):
    API_KEY = ""
    client = InferenceClient(
        provider="together",
        api_key=API_KEY
    )
    class Config:
        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        return "LlamaLLM"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        completion = self.client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct", 
            messages=messages, 
            max_tokens=512,
        )

        # print("API Response:", response.json())

        return completion.choices[0].message.content

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"llmUrl": "Dunno"}
def get_prompt(question, context):
    return f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_embeddings(model_name, model_kwargs, encode_kwargs):
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return hf