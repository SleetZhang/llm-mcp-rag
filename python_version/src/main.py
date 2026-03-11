from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from .agent import Agent
from .embedding_retriever import EmbeddingRetriever
from .mcp_client import MCPClient
from .utils import log_title

load_dotenv()

TASK="""
告诉我Antonette的信息,先从我给你的context中找到相关信息,总结后创作一个关于她的故事
把故事和她的基本信息保存到{out_path}/antonette_new.md,输出一个漂亮md文件
"""


async def retrieve_context(task: str) -> str:
    retriever = EmbeddingRetriever("BAAI/bge-m3")
    knowledge_dir = Path.cwd() / "knowledge"

    for file in knowledge_dir.glob("*.md"):
        retriever.embed_document(file.read_text(encoding="utf-8"))

    context = "\n".join(retriever.retrieve(task, top_k=3))
    log_title("CONTEXT")
    print(context)
    return context


async def main() -> None:
    out_path = Path.cwd() / "output"
    out_path.mkdir(exist_ok=True)
    task = TASK.format(out_path=out_path)

    fetch_mcp = MCPClient("mcp-server-fetch", "uvx", ["mcp-server-fetch"])
    file_mcp = MCPClient(
        "mcp-server-file",
        "npx",
        ["-y", "@modelcontextprotocol/server-filesystem", str(out_path)],
    )

    context = await retrieve_context(task)
    agent = Agent(
        model=os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini"),
        mcp_clients=[fetch_mcp, file_mcp],
        context=context,
    )

    await agent.init()
    try:
        await agent.invoke(task)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
