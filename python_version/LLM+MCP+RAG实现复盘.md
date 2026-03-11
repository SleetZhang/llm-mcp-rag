# LLM+MCP+RAG实现复盘

## 一、RAG的流程

1.索引知识库，loader将其转为文档，splitter切分这些文档，
 2.将上面的块通过embedding machine转为向量，保存到向量数据库
 3.用户prompt通过相同的embedding machine，查询最相近的向量，取出该片段
 4.获取到的知识片段，将它们与自定义系统提示和我们的问题一起格式化，

## 二、手搓项目代码流程（python version）

### 1.VectorStore：

向量数据库能够向里面添加新的item：（document，embedding）形式，能够实现相似度计算，然后查询与query_embedding最相近的topk个向量并返回内容

### 2.EmbeddingRetriever：

RAG的编排器，调用 embedding API + 调用向量库，把文本送去api向量化 + 调用向量库中的检索模块

### 3.MCPClient：

 作用：启动 通过标准I/O通信的服务端，与之建立会话，读取服务端提供的Tool list，然后调用工具
 StdioServerParameters：如何启动server的参数信息
 stdio_client：建立通信的管道
 ClientSession：真正的会话，真正的与server进行通信，发送 initialize / list_tools / call_tool

### 4.ChatOpenAI：

 先把与LLM的对话初始状态预置，初始化
 开始调用openai client进行聊天，得到流式输出，对流式输出进行拼接content和tool_calls，作为assistant的消息保存到历史消息
 添加tool_call的结果作为新的消息添加回去

### 5.Agent：

 Agent 负责 MCP 客户端的初始化/关闭，初始化 MCP 后并把 tools 清单交给 LLM
 `init()`：初始化MCP_Client和ChatOpenAI做异步连接和资源准备
 `close()`：做资源释放
 invoke():真正的chat以及交互逻辑，比较难的点是根据要工具调用的结果去找可以提供该工具的client，并且工具调用成功或者失败的结果都要写会历史messages

### 6.main：

1. 加载环境变量；
2. 生成任务文本；
3. 初始化两个 MCP 客户端；
4. 先做 RAG 拿到 context；
5. 创建并运行 Agent；
6. 确保退出时关闭资源。

### 7.agent+chat_openai搭配工作如下：

1. `Agent.invoke(prompt)` 先调用 `llm.chat(prompt)`；
2. `ChatOpenAI.chat()` 返回 `content + tool_calls`；
3. 如果聊天结果有 `tool_calls`，Agent 解析参数并通过MAPClient调用 MCP 工具；
4. Agent 再用 `append_tool_result(...)` 把MCP工具的输出塞回消息历史；
5. Agent 再次 `llm.chat()`，模型基于工具结果继续说；
6. 直到不再有 tool_calls，流程结束返最终内容。
7. 全流程：
              Agent
    LLM               MCP

 `main.py` 先做 RAG，得到 context。

 `Agent.init()` 初始化 MCP 并把工具 schema (工具的使用说明书)交给 `ChatOpenAI`。

 `Agent.invoke(task)` 发给 LLM。

 LLM 若返回 tool_calls，Agent 路由到 MCP 执行。.

 工具结果回填后再次问 LLM，循环直到无 tool_calls。

### 8.代码中的小bug

异步退出会报错，其余功能正常