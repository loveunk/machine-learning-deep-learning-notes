# AI Agent

AI Agent 是能基于目标、上下文和工具执行多步骤任务的 LLM 系统。它不是“更聪明的聊天机器人”，而是一个带工具、状态、权限和评估的工作流。

## 一个实用定义

```text
Agent = LLM + tools + state + workflow + guardrails + evaluation
```

其中：

- LLM：理解任务、生成参数、解释结果。
- tools：搜索、数据库、计算、文件、API 等外部能力。
- state：保存任务进度、工具结果、用户确认和错误。
- workflow：控制任务怎么推进、何时停止、何时转人工。
- guardrails：约束输入、工具和输出。
- evaluation：证明系统是否可靠。

## Agent 和普通 Chatbot 的区别

| 能力 | Chatbot | Agent |
| --- | --- | --- |
| 回答问题 | 可以 | 可以 |
| 调用工具 | 通常没有 | 核心能力 |
| 多步骤任务 | 弱 | 需要 workflow |
| 状态管理 | 对话历史为主 | 显式任务状态 |
| 权限控制 | 较少 | 必须 |
| 可观测性 | 可选 | 必须 |

## 什么时候需要 Agent

适合：

- 需要查询多个系统。
- 需要根据中间结果决定下一步。
- 需要调用工具完成动作。
- 需要人机协作审批。
- 需要长期任务或可恢复任务。

不适合：

- 简单问答。
- 单次文本改写。
- 纯文档问答。
- 无法验证结果的高风险任务。

这些场景优先用 Prompt、RAG 或普通程序解决。

## Agent 的核心组件

### 工具

工具要有清晰 schema、权限和错误格式。见 [Agent 工具调用](agent-tools.md)。

### 状态

状态记录：

- 用户目标。
- 当前步骤。
- 已调用工具。
- 工具结果。
- 已确认动作。
- 错误和重试次数。

### 工作流

工作流决定 Agent 如何推进。见 [Agent 工作流模式](agent-workflows.md)。

### 评估

Agent 评估要覆盖：

- 是否选择正确工具。
- 参数是否正确。
- 是否正确理解工具结果。
- 是否在需要时停止或转人工。
- 最终结果是否满足目标。

## 最小工具型 Agent 流程

```text
用户：查一下这篇文档里 RAG 评估怎么做
  -> 模型决定调用 search_docs
  -> 程序校验 query 和 top_k
  -> 工具返回相关片段
  -> 模型基于片段回答并给引用
```

注意：工具执行必须由程序控制，不能让模型绕过权限和参数校验。

## 主要风险

| 风险 | 说明 | 控制 |
| --- | --- | --- |
| 工具误用 | 调错工具或参数 | schema、校验、eval |
| 越权操作 | 调用不该调用的系统 | 权限层、人工确认 |
| 循环失控 | 反复调用工具 | 最大步数、超时 |
| 幻觉动作结果 | 没按 observation 回答 | 结构化工具结果、引用检查 |
| prompt injection | 文档诱导模型忽略规则 | 不信任外部内容、权限隔离 |
| 成本失控 | 多轮调用和重试 | 路由、缓存、预算 |

## 学习顺序

1. [Agent 工具调用与结构化输出](agent-tools.md)
2. [Agent 工作流模式](agent-workflows.md)
3. [Agent 生产化](agent-production.md)

如果你还没有做过 RAG，建议先学 [RAG](rag.md)，因为很多 Agent 都会把检索作为基础工具。

## 参考资料

- OpenAI Agents SDK: https://developers.openai.com/api/docs/guides/agents
- LangGraph Overview: https://docs.langchain.com/oss/python/langgraph/overview
- Hugging Face Agents Course: https://huggingface.co/learn/agents-course/en/unit0/introduction
- smolagents: https://huggingface.co/docs/smolagents/en/index
