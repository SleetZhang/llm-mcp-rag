from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .utils import log_title

class ChatOpenAI:
    def __init__(
            self,
            model:str,
            tools:list[dict[str,Any]] |None = None,
            system_prompt:str ="",
            context:str ="",
    )->None:
        self.client=OpenAI()
        self.model=model
        self.tools=tools or []
        self.messages:list[dict[str,Any]]=[]
        if system_prompt:
            self.messages.append({"role":"system","content":system_prompt})
        if context:
            self.messages.append({"role":"user","content":context})
    
    def chat(self,prompt:str|None=None)->tuple[str,list[dict[str,Any]]]:
        log_title("CHAT")
        if prompt:
            self.messages.append({"role":"user","content":prompt})
        #流式chat
        stream=self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tools,
            stream=True,
        )
        #拼接流的信息
        content=""
        tool_calls:list[dict[str,Any]]=[]
        log_title("RESPONSE")
        #遍历流
        for chunk in stream:
            #拿到增量信息
            delta=chunk.choices[0].delta
            #增量信息有content
            if delta.content:
                content+=delta.content
                #打印到控制台
                print(delta.content, end="", flush=True)
            #增量信息有多个工具调用
            if delta.tool_calls:
                for tool_call_chunk in delta.tool_calls:
                    #根据工具的索引信息来拼接
                    index=tool_call_chunk.index
                    #保存工具信息的列表不够长，逐步用空的内容扩充
                    while len(tool_calls)<=index:
                        tool_calls.append({"id":"","type":"function","function":{"name":"","arguments":""}})
                    #开始对index处拼接
                    current=tool_calls[index]
                    if tool_call_chunk.id:
                        current["id"]+=tool_call_chunk.id
                    if tool_call_chunk.function and tool_call_chunk.function.name:
                        current["function"]["name"]+=tool_call_chunk.function.name
                    if tool_call_chunk.function and tool_call_chunk.function.arguments:
                        current["function"]["arguments"]+=tool_call_chunk.function.arguments
        #将本轮所有信息添加到messages
        assistant_message:dict[str,Any]={"role":"assistant","content":content}
        if tool_calls:
            assistant_message["tool_calls"]=tool_calls
        self.messages.append(assistant_message)
        print()
        #返回值的tool_calls供agent参考是否去调用工具
        return content,tool_calls
    
    #工具调用结果通过tool_output传入，解析后添加到messages
    def append_tool_result(self,tool_call_id:str,tool_output:Any)->None:
        self.messages.append(
            {
                "role":"tool",
                "tool_call_id":tool_call_id,
                "content":json.dumps(tool_output,ensure_ascii=False),
            }
        )