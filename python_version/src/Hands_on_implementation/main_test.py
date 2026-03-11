from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from .agent import Agent
from .mcp_client import MCPClient
from .utils import log_title
from .embedding_retriever import EmbeddingRetriever

#先加载环境变量
load_dotenv()

#任务模板
TASK = """
告诉我Antonette的信息,先从我给你的context中找到相关信息,总结后创作一个关于她的故事
把故事和她的基本信息保存到{out_path}/antonette.md,输出一个漂亮md文件
"""

#先根据任务做rag，首先把本地知识库嵌入进向量数据库，然后检索top3，合成最终的context
async def retrieve_context(task:str)->str:
    #检索器
    retriever=EmbeddingRetriever("BAAI/bge-m3")
    #知识库目录
    knowledge_dir=Path.cwd()/"knowledge"
    #对每个file进行嵌入,file是path对象
    for file in knowledge_dir.glob("*.md"):
        retriever.embed_document(file.read_text(encoding="utf-8"))
    #开始检索并拼接
    context="\n".join(retriever.retrieve(task,top_k=3))
    log_title("CONTEXT")
    print(context)
    return context

#开始做主要任务
async def main()->None:
    #先得到真正的任务指令
    out_path=Path.cwd()/"output"
    out_path.mkdir(exist_ok=True)
    task=TASK.format(out_path=out_path)

    #创建两个供agent使用的mcp client
    fetch_mcp=MCPClient("mcp-server-fetch", "uvx", ["mcp-server-fetch"])
    file_mcp=MCPClient("mcp-server-file",
        "npx",
        ["-y", "@modelcontextprotocol/server-filesystem", str(out_path)],
    )#out_path是path对象，命令里要转为字符串

    #调用rag，得到上下文
    context=await retrieve_context(task)
    #创建agent
    agent=Agent(
        model=os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini"),
        mcp_clients=[fetch_mcp,file_mcp],
        context=context
    )
    #等待agent初始化mcp
    await agent.init()

    #agent开始做任务
    try:
        await agent.invoke(task)
    finally:
        await agent.close()


if __name__=="__main__":
    asyncio.run(main())