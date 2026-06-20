# API 与模型选型

本章目标：学会把一个 LLM 需求拆成“任务、输入、输出、质量、成本、延迟、安全”几个维度，然后选择合适的模型和 API。

## 先判断任务类型

| 任务 | 典型输入 | 典型输出 | 优先关注 |
| --- | --- | --- | --- |
| 文本生成 | 指令、上下文 | 自然语言 | 质量、风格、成本 |
| 结构化抽取 | 文档、网页、日志 | JSON、表格 | 格式稳定、字段召回 |
| 代码生成 | 需求、上下文代码 | patch、函数、解释 | 正确性、测试、上下文长度 |
| 文档问答 | 问题、检索片段 | 带引用答案 | 引用准确、幻觉控制 |
| 多模态理解 | 图片、视频、音频、文本 | 描述、分类、问答 | 输入模态、延迟、隐私 |
| Agent | 目标、工具、状态 | 工具调用、最终结果 | 可控性、权限、可观测性 |

不要先问“哪个模型最强”，先问“这个任务失败时会造成什么损失”。

## API 形态

### 托管模型 API

适合：

- 快速验证产品想法。
- 不想维护推理基础设施。
- 需要高质量通用模型、多模态能力或工具调用。

注意：

- 成本随 token、工具调用、检索和重试增长。
- 数据合规和日志策略要提前确定。
- 生产环境必须有超时、重试、限流和降级。

示例：

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="your-model",
    input="用三句话解释什么是过拟合。"
)

print(response.output_text)
```

### 开源模型本地推理

适合：

- 数据不能出内网。
- 任务稳定且吞吐可预测。
- 可以接受模型能力和维护成本的权衡。

注意：

- 需要评估显存、吞吐、量化精度和并发。
- 小模型未必更便宜，取决于部署、运维和利用率。
- 本地模型也需要 eval，不要只凭主观体验上线。

示例：

```python
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device_map="auto"
)

result = generator("解释什么是梯度下降：", max_new_tokens=120)
print(result[0]["generated_text"])
```

## 模型选择维度

| 维度 | 你要问的问题 |
| --- | --- |
| 质量 | 是否能通过你的真实评估集，而不是只通过 demo 问题 |
| 上下文 | 文档、代码、对话历史是否放得下 |
| 延迟 | P50/P95 延迟是否满足交互要求 |
| 成本 | 单次请求成本、重试成本、评估成本、缓存收益是多少 |
| 工具能力 | 是否稳定支持 tool calling、结构化输出、并行工具 |
| 多模态 | 是否需要图片、音频、视频输入或输出 |
| 部署 | 托管 API、本地推理、私有云还是混合 |
| 合规 | 数据是否可外发，日志是否可保存 |

## 推荐选型流程

1. 写 20-50 条真实评估样例。
2. 先用强模型做 baseline，确认任务上限。
3. 再用便宜或本地模型做对比。
4. 记录质量、成本、延迟、失败类型。
5. 只在评估集上证明“便宜模型够用”后再降级。

表格模板：

| 模型 | 通过率 | 平均成本 | P95 延迟 | 主要失败 |
| --- | --- | --- | --- | --- |
| strong-model | 92% | 高 | 3.2s | 少量引用错误 |
| small-model | 78% | 低 | 1.1s | 长文档漏召回 |
| local-model | 70% | 固定成本 | 0.8s | 格式不稳定 |

## 常见错误

### 只看排行榜

排行榜不能代表你的数据。模型选型必须基于自己的评估集。

### 只看输入 token 价格

真实成本还包括：

- 输出 token。
- embedding 和 rerank。
- 工具调用。
- 重试。
- 缓存命中率。
- 人工审核成本。

### 忽略失败处理

生产代码至少要处理：

- 超时。
- 限流。
- 模型返回空结果。
- JSON 解析失败。
- 工具调用参数不合法。
- 上游文档或数据库不可用。

### 把模型选择和架构选择混在一起

如果模型回答缺少私有知识，优先考虑 RAG，而不是微调。

如果输出格式不稳定，优先考虑结构化输出、校验和重试，而不是换更大模型。

如果任务需要外部动作，才考虑 Agent。

## 下一步

- 学会控制输入和输出：[Prompt 与上下文工程](prompting.md)
- 学会接入知识库：[Embedding 与向量检索](embeddings.md)
- 学会构建问答系统：[RAG](rag.md)
- 学会工具调用：[Agent 工具调用](agent-tools.md)

## 参考资料

- OpenAI API Docs: https://developers.openai.com/api/docs/
- OpenAI Agents SDK: https://developers.openai.com/api/docs/guides/agents
- Hugging Face Transformers: https://huggingface.co/docs/transformers/index
