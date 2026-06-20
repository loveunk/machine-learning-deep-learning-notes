# GPT 系列与 Decoder-only 模型

GPT 系列的核心意义不在某一个模型名，而在于验证了“自回归预训练 + 指令对齐 + 大规模扩展”可以产生通用生成能力。今天主流生成式 LLM 大多沿用了 decoder-only Transformer 的路线。

## 关键演进

| 阶段 | 代表 | 关键贡献 |
| --- | --- | --- |
| GPT-1 | 2018 | 生成式预训练 + 下游微调 |
| GPT-2 | 2019 | 更大规模语言模型展示通用生成能力 |
| GPT-3 | 2020 | few-shot / in-context learning 变得明显可用 |
| InstructGPT / ChatGPT | 2022 | 指令微调和 RLHF 让模型更适合对话和任务执行 |
| 多模态与工具使用 | 2023 以后 | 模型开始处理图像、音频、代码、工具调用和复杂工作流 |

具体模型名称会不断变化，但这条技术线的核心问题比较稳定：

- 如何扩大模型、数据和计算规模。
- 如何让模型听懂指令。
- 如何让模型减少幻觉和有害输出。
- 如何让模型调用工具、处理长上下文和多模态输入。
- 如何评估真实任务能力。

## Decoder-only 为什么适合生成

Decoder-only 模型使用 causal mask，只能看到当前位置之前的 token，因此天然适合“预测下一个 token”。

```text
tokens:      我   喜   欢   机器   学习
can attend: 我
             我   喜
             我   喜   欢
             我   喜   欢   机器
```

这种训练目标简单、可扩展，和文本生成任务一致。

## GPT 和 BERT 的区别

| 维度 | GPT / Decoder-only | BERT / Encoder-only |
| --- | --- | --- |
| 训练目标 | 预测下一个 token | 预测被 mask 的 token |
| 注意力 | 单向 causal attention | 双向 attention |
| 强项 | 生成、对话、代码、Agent | 分类、抽取、检索表征 |
| 典型使用 | Chat、RAG 生成、工具调用 | 文本分类、NER、embedding/rerank |

## GPT 类模型的工程能力

现代 GPT 类模型常被用作：

- 文本生成器。
- 结构化抽取器。
- 代码助手。
- RAG 答案生成器。
- 工具调用决策器。
- Agent workflow 中的局部推理节点。

工程上不要把它看成万能大脑，而要把它看成可组合组件。

## 学习重点

你不需要记住每代模型的参数量。更重要的是理解：

1. decoder-only 的生成方式。
2. in-context learning 为什么让 prompt 变重要。
3. 指令对齐为什么改变用户体验。
4. 工具调用为什么把模型接入外部世界。
5. eval 为什么比主观试用更重要。

## 下一步

- 对比理解型模型：[BERT 系列](bert-series.md)
- 回到架构基础：[Transformer](transformer.md)
- 学工程控制：[Prompt 与上下文工程](prompting.md)
