from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .utils import log_title


class ChatOpenAI:
    def __init__(
        self,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
        context: str = "",
    ) -> None:
        self.client = OpenAI()
        self.model = model
        self.tools = tools or []
        self.messages: list[dict[str, Any]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        if context:
            self.messages.append({"role": "user", "content": context})

    def chat(self, prompt: str | None = None) -> tuple[str, list[dict[str, Any]]]:
        log_title("CHAT")
        if prompt:
            self.messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tools,
            stream=True,
        )

        content = ""
        tool_calls: list[dict[str, Any]] = []
        log_title("RESPONSE")
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content
                print(delta.content, end="", flush=True)

            if delta.tool_calls:
                for tool_call_chunk in delta.tool_calls:
                    index = tool_call_chunk.index
                    while len(tool_calls) <= index:
                        tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    current = tool_calls[index]
                    if tool_call_chunk.id:
                        current["id"] += tool_call_chunk.id
                    if tool_call_chunk.function and tool_call_chunk.function.name:
                        current["function"]["name"] += tool_call_chunk.function.name
                    if tool_call_chunk.function and tool_call_chunk.function.arguments:
                        current["function"]["arguments"] += tool_call_chunk.function.arguments

        assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        self.messages.append(assistant_message)
        print()
        return content, tool_calls

    def append_tool_result(self, tool_call_id: str, tool_output: Any) -> None:
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(tool_output, ensure_ascii=False),
            }
        )
