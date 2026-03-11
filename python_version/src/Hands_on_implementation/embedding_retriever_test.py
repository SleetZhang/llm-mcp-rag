from __future__ import annotations

import os

import requests

from .utils import log_title
#操作向量数据库
from .vector_store import VectorStore

#RAG的编排器，负责调用embedding的api，调用向量数据库
class EmbeddingRetriever:
    def __init__(self,embedding_model:str) ->None:
        self.embedding_model=embedding_model
        self.vector_store=VectorStore()
    
    #文本向量化并入库
    def embed_document(self,document:str)->list[float]:
        log_title("EMBEDDING DOCUMENT")
        embedding=self._embed(document)
        self.vector_store.add_embedding(embedding=embedding,document=document)
        return embedding
    
    #query向量化
    def embed_query(self,query:str)->list[float]:
        log_title("EMBEDDING QUERY")
        return self._embed(query)
    
    #开始做调用向量数据库做检索
    def retrieve(self,query:str,top_k:int=3)->list[str]:
        query_embedding=self.embed_query(query)
        return self.vector_store.search(query_embedding,top_k)

    #真正的调用embedding api
    def _embed(self,text:str)->list[float]:
        response=requests.post(
            #url
            f"{os.environ['EMBEDDING_BASE_URL']}/embeddings",
            #请求头
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['EMBEDDING_KEY']}",
            },
            #请求体
            json={
                "model":self.embedding_model,
                "input":text,
                "encoding_format": "float",
            },
            timeout=30,
        )
        response.raise_for_status()
        data=response.json()
        return data["data"][0]["embedding"]