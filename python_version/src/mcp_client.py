from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#后加的，防止退出爆了
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class ToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]


class MCPClient:
    def __init__(self, name: str, command: str, args: list[str]) -> None:
        self.name = name
        self.command = command
        self.args = args
        self._session: ClientSession | None = None
        self._stdio_context = None
        self._tools: list[ToolInfo] = []
        #后加的退出栈
        self._stack: AsyncExitStack | None = None

    async def init(self) -> None:
        #后加的
        if self._stack is not None:
            return  # 幂等
        self._stack = AsyncExitStack()
        
        server = StdioServerParameters(command=self.command, args=self.args)
        self._stdio_context = stdio_client(server)
        read, write = await self._stdio_context.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        tools_result = await self._session.list_tools()
        self._tools = [
            ToolInfo(
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema,
            )
            for t in tools_result.tools
        ]

    async def close(self) -> None:
        # if self._session is not None:
        #     await self._session.__aexit__(None, None, None)
        #     self._session = None
        # if self._stdio_context is not None:
        #     await self._stdio_context.__aexit__(None, None, None)
        #     self._stdio_context = None
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
        self._session = None

    def get_tools(self) -> list[ToolInfo]:
        return self._tools

    
    async def call_tool(self, name: str, params: dict[str, Any]) -> Any:
        if self._session is None:
            raise RuntimeError("MCP client not initialized")
        return await self._session.call_tool(name, params)
