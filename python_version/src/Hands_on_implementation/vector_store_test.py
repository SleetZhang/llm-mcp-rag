from __future__ import annotations

from dataclasses import dataclass

#先把接下来要写的vectorstoreitem类表示为数据类
@dataclass
class VectorStoreItem:
    embedding:list[float]
    document:str

class VectorStore:
    def __init__(self) -> None:
        self._items:list[VectorStoreItem]=[]
    
    def add_embedding(self,embedding:list[float],document:str) -> None:
        self._items.append(VectorStoreItem(embedding=embedding,document=document))

    #搜索与query相近的top_k个返回document
    def search(self,query_embedding:list[float],top_k:int=3) ->list[str]:
        #先打分,元组形式存储分数和document
        scored=[(item.document,self._cosine_similarity(query_embedding,item.embedding))
            for item in self._items
        ]
        scored.sort(key=lambda x:x[1],reverse=True)
        return [doc for doc,_ in scored[:top_k]]
    
    #计算余弦相似度的函数
    @staticmethod
    def _cosine_similarity(vec_a:list[float],vec_b:list[float]) ->float:
        dot=sum(a*b for a,b in zip(vec_a,vec_b))
        norm_a=sum(a*a for a in vec_a)**0.5
        norm_b=sum(b*b for b in vec_b)**0.5
        if norm_a==0 or norm_b==0:
            return 0.0
        return dot/(norm_a*norm_b)