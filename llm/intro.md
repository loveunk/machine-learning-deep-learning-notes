# LLM 入门

本章目标：用最短路径理解大语言模型是什么、适合做什么、不适合做什么，以及如何开始构建一个可靠的 LLM 应用。

## 什么是 LLM

LLM（Large Language Model，大语言模型）是在大规模文本和多模态数据上训练的模型，擅长根据上下文生成、改写、总结、分类、抽取和推理文本。

更工程化地看，LLM 是一个“上下文到输出”的函数：

```text
instructions + context + examples + user input -> model output
```

你能控制的不是模型内部参数，而是：

- 给它什么任务。
- 给它什么上下文。
- 要它用什么格式输出。
- 如何验证它的输出。
- 什么时候让它调用工具。

## LLM 能做什么

| 能力 | 示例 |
| --- | --- |
| 文本生成 | 写邮件、写说明、改写文案 |
| 文本理解 | 摘要、分类、情感分析、信息抽取 |
| 代码辅助 | 生成代码、解释代码、写测试、排查错误 |
| 知识问答 | 结合 RAG 回答企业文档问题 |
| 工具调用 | 查询数据库、搜索文档、调用 API |
| 多模态理解 | 图片问答、图文检索、视觉辅助分析 |

## LLM 不适合直接做什么

| 问题 | 原因 | 更好的做法 |
| --- | --- | --- |
| 精确事实查询 | 模型可能幻觉或知识过期 | RAG、搜索、数据库 |
| 数值计算 | 自然语言模型不保证精确 | 调用计算工具 |
| 私有知识问答 | 训练数据里没有你的文档 | RAG |
| 高风险自动执行 | 工具可能造成真实影响 | 权限、人工确认、审计 |
| 稳定结构化结果 | 自然语言格式会漂移 | 结构化输出、schema 校验 |

## 第一个 API 调用

托管 API 的优势是快速验证想法。示例：

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="your-model",
    input="用初学者能理解的话解释什么是梯度下降。"
)

print(response.output_text)
```

注意：

- 不要把 API key 写进代码。
- 生产环境要处理超时、限流、重试和日志。
- 不要把用户敏感数据随意发给外部 API。

## 第一个本地模型调用

```python
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device_map="auto"
)

result = generator("用一句话解释什么是过拟合：", max_new_tokens=80)
print(result[0]["generated_text"])
```

本地模型适合学习和隐私敏感场景，但要自己承担模型下载、显存、吞吐和质量评估。

## LLM 应用的基本架构

```text
user input
  -> input validation
  -> prompt/context construction
  -> model call
  -> output validation
  -> final response
```

如果需要私有知识：

```text
user input
  -> retrieve relevant documents
  -> prompt with context
  -> model answer with citations
```

如果需要外部动作：

```text
user input
  -> model chooses tool
  -> program validates tool arguments
  -> tool executes
  -> model explains result
```

## 学习路线

1. [API 与模型选型](api-and-models.md)：知道如何选择模型和接口。
2. [Prompt 与上下文工程](prompting.md)：学会控制输入和输出。
3. [Transformer](transformer.md)：理解架构基础。
4. [Embedding 与向量检索](embeddings.md)：理解语义检索。
5. [RAG](rag.md)：解决私有知识和引用问题。
6. [AI Agent](agent.md)：让模型安全调用工具并完成多步骤任务。

## 关键判断

| 需求 | 先尝试 |
| --- | --- |
| 改善回答格式 | Prompt + 结构化输出 |
| 回答企业文档问题 | RAG |
| 固定领域风格或标签 | 微调 |
| 查询实时系统 | 工具调用 |
| 多步骤执行 | Agent workflow |

## 参考资料

- OpenAI API Docs: https://developers.openai.com/api/docs/
- Hugging Face NLP Course: https://huggingface.co/learn/nlp-course/
- Hugging Face Transformers: https://huggingface.co/docs/transformers/index
