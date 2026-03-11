from __future__ import annotations

from dataclasses import dataclass

from typing import Any

from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client

@dataclass
class ToolInfo:
    name:str
    description:str
    input_schema:dict[str,Any]

class MCPClient:
    def __init__(self,name:str,command:str,args:list[str])->None:
        self.name=name
        self.command=command
        self.args=args
        self._session:ClientSession|None =None
        self._stdio_context=None
        self._tools:list[ToolInfo]=[]

    #MCP客户端启动server流程
    async def init(self)->None:
        #先形成启动配置
        server=StdioServerParameters(command=self.command,args=self.args)
        #通过配置建立输入输出流的上下文管理
        self._stdio_context=stdio_client(server)
        #建立连接管道
        read,write=await self._stdio_context.__aenter__()
        #建立会话
        self._session=ClientSession(read,write)
        await self._session.__aenter__()
        await self._session.initialize()
        tools_result=await self._session.list_tools()
        self._tools=[
            ToolInfo(
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema,
            )
            for t in tools_result.tools
        ]
    
    #先关会话，再关管道
    async def close(self)->None:
        if self._session is not None:
            await self._session.__aexit__(None,None,None)
            self._session=None
        if self._stdio_context is not None:
            await self._stdio_context.__aexit__(None,None,None)
            self._stdio_context=None

    def get_tools(self)->list[ToolInfo]:
        return self._tools
    
    #进行工具调用一定建立了会话，所以不为none
    async def call_tool(self,name:str,params:dict[str,Any])->Any:
        if self._session is None:
            raise RuntimeError("MCP client not initialized")
        return await self._session.call_tool(name,params)