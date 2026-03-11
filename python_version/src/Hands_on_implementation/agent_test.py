from __future__ import annotations

import json

from .chat_openai import ChatOpenAI
from .mcp_client import MCPClient
from .utils import log_title

class Agent:
    #Agent能够调用的llm和mcp参数配置
    def __init__(
            self,
            model:str,
            mcp_clients:list[MCPClient],
            system_prompt:str="",
            context:str="",
    )->None:
        self.model=model
        self.mcp_clients=mcp_clients
        self.system_prompt=system_prompt
        self.context=context
        self.llm:ChatOpenAI |None =None

    #开始初始化mcp_client和chat_llm
    async def init(self)->None:
        log_title("TOOLS")
        for client in self.mcp_clients:
            await client.init()
        
        #得到tool_list来初始化self.llm
        tools=[]
        for client in self.mcp_clients:
            #tool=client.get_tools()
            #上面返回的是当前client的工具列表的信息，但不是openai格式的，只有name ，description， input_schema
            for tool in client.get_tools():
                tools.append(
                    {
                        "type":"function",
                        "function":{
                            "name":tool.name,
                            "description":tool.description,
                            "parameters":tool.input_schema
                        }
                    }
                )
        self.llm=ChatOpenAI(self.model,tools,self.system_prompt,self.context)

    #关闭client
    async def close(self)->None:
        for client in self.mcp_clients:
            await client.close()

    #真正的开始chat
    async def invoke(self,prompt:str)->str:
        if self.llm is None:
            raise RuntimeError("Agent is not initialized")
        
        content,tool_calls=self.llm.chat(prompt)
        #如果有工具调用，继续chat，没有就直接return
        while True:
            #有工具调用
            if tool_calls:
                #执行每一个工具调用,取出name和args，根据name找到匹配的client
                for tool_call in tool_calls:
                    name=tool_call["function"]["name"]
                    #json字符串格式解析成python字典格式
                    args=json.loads(tool_call["function"]["arguments"] or "{}")
                    #取出第一个可用的client
                    target_client=next(
                       (c for c in self.mcp_clients if any(name==t.name for t in c.get_tools())),
                       None
                    )
                    #如果没有
                    if target_client is None:
                        self.llm.append_tool_result(tool_call["id"],{"error": "Tool not found"})
                        continue

                    log_title("TOOL USE")
                    print(f"Calling tool: {name}")
                    print(f"Arguments: {args}")
                    result=await target_client.call_tool(name,args)
                    print(f"Result: {result}")
                    self.llm.append_tool_result(tool_call["id"],result)
                content,tool_calls=self.llm.chat()
                continue
            #无工具调用
            return  content