# 大语言模型与 Agent 学习路径

本目录面向想从“会用 ChatGPT”走到“能构建 LLM 应用”的学习者。重点不是追最新模型名，而是掌握稳定的工程能力：模型调用、上下文组织、检索、评估、工具调用、Agent 工作流和生产化。

## 学习顺序

| 阶段 | 章节 | 目标 |
| --- | --- | --- |
| 1 | [LLM 入门](intro.md) | 知道 LLM 能做什么、不能做什么，以及如何安全调用 |
| 2 | [API 与模型选型](api-and-models.md) | 会按任务选择模型、接口、成本和延迟策略 |
| 3 | [Prompt 与上下文工程](prompting.md) | 会设计指令、上下文、few-shot、结构化输出 |
| 4 | [Transformer](transformer.md) | 理解现代 LLM 的基本架构 |
| 5 | [Embedding 与向量检索](embeddings.md) | 会把文本变成可检索的向量索引 |
| 6 | [RAG](rag.md) 与 [从零实现 RAG](rag-from-scratch.md) | 能构建最小知识库问答系统 |
| 7 | [RAG 评估](rag-evaluation.md) | 能证明检索和生成是否变好 |
| 8 | [微调方法](fine-tuning.md) | 会判断何时需要 LoRA/QLoRA/全量微调 |
| 9 | [AI Agent](agent.md) | 理解 Agent 的边界和风险 |
| 10 | [工具调用](agent-tools.md)、[工作流](agent-workflows.md)、[生产化](agent-production.md) | 能设计可控、可观测、可评估的 Agent 系统 |

## 四个能力层级

### Level 1: 会调用模型

你应该能做到：

- 使用一个托管模型 API 完成文本生成。
- 设置 system/user/developer 指令。
- 处理超时、重试、限流和日志。
- 不把 API key 写进代码仓库。

对应章节：[LLM 入门](intro.md)、[API 与模型选型](api-and-models.md)。

### Level 2: 会控制输出

你应该能做到：

- 把任务拆成指令、上下文、输出格式和约束。
- 使用 few-shot 示例稳定格式。
- 对关键任务使用结构化输出或 JSON schema。
- 设计失败时的重试和校验。

对应章节：[Prompt 与上下文工程](prompting.md)。

### Level 3: 会接入私有知识

你应该能做到：

- 对文档做 chunking、embedding 和向量检索。
- 区分召回问题、排序问题和生成问题。
- 给答案附引用，并能解释引用是否支持答案。
- 用评估集比较不同 RAG 配置。

对应章节：[Embedding 与向量检索](embeddings.md)、[RAG](rag.md)、[从零实现 RAG](rag-from-scratch.md)、[RAG 评估](rag-evaluation.md)。

### Level 4: 会做 Agent 工程

你应该能做到：

- 把工具写成清晰的函数契约。
- 使用状态机或 workflow 管理多步骤任务。
- 设置权限、人工确认、guardrails 和审计日志。
- 用 tracing 和 eval 定位 Agent 失败原因。

对应章节：[AI Agent](agent.md)、[工具调用](agent-tools.md)、[Agent 工作流](agent-workflows.md)、[Agent 生产化](agent-production.md)。

## 什么时候用 Prompt、RAG、微调或 Agent

| 方案 | 适合 | 不适合 |
| --- | --- | --- |
| Prompt | 任务规则清晰、知识在上下文中、输出要求稳定 | 需要大量私有知识或长期记忆 |
| RAG | 知识经常变化、需要引用来源、企业文档问答 | 需要模型学习新行为或新风格 |
| 微调 | 输出风格、格式、领域语言或分类边界稳定 | 只是补充事实知识 |
| Agent | 需要调用工具、查询外部系统、多步骤执行 | 简单问答、风险高且无法验证的任务 |

## 推荐阅读源

- OpenAI API 文档：[Responses、工具、Agents、评估与生产化](https://developers.openai.com/api/docs/guides/agents)
- Hugging Face：[Transformers](https://huggingface.co/docs/transformers/index)、[smolagents](https://huggingface.co/docs/smolagents/en/index)、[Agents Course](https://huggingface.co/learn/agents-course/en/unit0/introduction)
- LangChain / LangGraph：[LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)
- LlamaIndex：[LlamaIndex docs](https://docs.llamaindex.ai/)

## 学习检查清单

- [ ] 能写一个最小 LLM API 调用脚本。
- [ ] 能解释 temperature、max tokens、上下文长度、结构化输出的作用。
- [ ] 能把 10 篇文档做成一个可检索知识库。
- [ ] 能建立 20-50 条评估问题，比较 RAG 改动。
- [ ] 能写 3 个工具并让模型按 schema 调用。
- [ ] 能画出 Agent 工作流的状态图。
- [ ] 能记录每次模型调用、工具调用、成本、延迟和失败原因。
