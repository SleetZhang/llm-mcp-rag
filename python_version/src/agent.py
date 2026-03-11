from __future__ import annotations

import json

from .chat_openai import ChatOpenAI
from .mcp_client import MCPClient
from .utils import log_title


class Agent:
    def __init__(
        self,
        model: str,
        mcp_clients: list[MCPClient],
        system_prompt: str = "",
        context: str = "",
    ) -> None:
        self.model = model
        self.mcp_clients = mcp_clients
        self.system_prompt = system_prompt
        self.context = context
        self.llm: ChatOpenAI | None = None

    async def init(self) -> None:
        log_title("TOOLS")
        for client in self.mcp_clients:
            await client.init()

        tools = []
        for client in self.mcp_clients:
            for tool in client.get_tools():
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        },
                    }
                )
        self.llm = ChatOpenAI(self.model, tools, self.system_prompt, self.context)

    async def close(self) -> None:
        for client in self.mcp_clients:
            try:
                await client.close()
            except Exception as e:
                print(f"[WARN] close client failed: {e}")
    async def invoke(self, prompt: str) -> str:
        if self.llm is None:
            raise RuntimeError("Agent is not initialized")

        content, tool_calls = self.llm.chat(prompt)
        while True:
            if tool_calls:
                for tool_call in tool_calls:
                    name = tool_call["function"]["name"]
                    args = json.loads(tool_call["function"]["arguments"] or "{}")
                    target_client = next(
                        (c for c in self.mcp_clients if any(t.name == name for t in c.get_tools())),
                        None,
                    )
                    if target_client is None:
                        self.llm.append_tool_result(tool_call["id"], {"error": "Tool not found"})
                        continue

                    log_title("TOOL USE")
                    print(f"Calling tool: {name}")
                    print(f"Arguments: {args}")
                    result = await target_client.call_tool(name, args)
                    #新增内容，result为实例对象不能直接被append_tool_result函数传入
                    if hasattr(result, "model_dump"):
                        payload = result.model_dump()
                    elif isinstance(result, (dict, list, str, int, float, bool)) or result is None:
                        payload = result
                    else:
                        payload = str(result)
                    print(f"Result: {payload}")
                    self.llm.append_tool_result(tool_call["id"], payload)

                content, tool_calls = self.llm.chat()
                continue

            return content
