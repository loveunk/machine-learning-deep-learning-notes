# Transformer 架构

Transformer 是现代 LLM 的基础架构。你不需要一开始推完所有公式，但必须理解它为什么能处理序列、为什么注意力重要、以及 encoder-only、decoder-only、encoder-decoder 的区别。

## Transformer 解决的问题

RNN 按时间步顺序处理序列，长距离依赖和并行训练都比较困难。Transformer 用 self-attention 让每个 token 直接和其他 token 交互，从而更适合大规模并行训练。

```text
输入 token -> embedding -> self-attention -> feed-forward -> 输出表示
```

## Self-attention

每个 token 会生成三个向量：

- Query：我想找什么信息。
- Key：我能提供什么信息。
- Value：真正被汇总的信息。

注意力分数：

```text
score = QK^T / sqrt(d_k)
attention = softmax(score) V
```

直觉：一个 token 会根据相关性，从其他 token 中加权收集信息。

## Multi-head attention

单个 attention head 只能从一种角度看关系。Multi-head attention 并行学习多种关系：

- 主谓关系。
- 指代关系。
- 位置关系。
- 代码中的变量引用。
- 文档中的标题和段落关系。

多个 head 的结果会拼接后再投影回隐藏维度。

## Position encoding

Self-attention 本身不包含顺序信息，因此需要位置编码。常见方式：

- Sinusoidal position encoding。
- Learnable position embedding。
- RoPE（Rotary Position Embedding）。
- ALiBi 等长上下文相关方法。

位置编码让模型知道“词出现在哪里”。

## Feed-forward network

每个 Transformer block 里还有前馈网络：

```text
hidden -> linear -> activation -> linear -> hidden
```

粗略理解：

- Attention 负责 token 间信息交换。
- FFN 负责对每个位置的表示做非线性变换和知识存储。

## 三类架构

| 架构 | 代表 | 适合 |
| --- | --- | --- |
| Encoder-only | BERT、RoBERTa、DeBERTa | 分类、抽取、语义表示 |
| Decoder-only | GPT、Llama、Qwen 等 | 生成、对话、代码、Agent |
| Encoder-decoder | T5、BART | 翻译、摘要、文本到文本任务 |

## Causal mask

Decoder-only 模型生成文本时不能偷看未来 token，因此使用 causal mask。

```text
第 1 个 token 只能看第 1 个
第 2 个 token 可以看第 1-2 个
第 3 个 token 可以看第 1-3 个
```

这和“预测下一个 token”的训练目标一致。

## 为什么 Transformer 能扩展

关键原因：

- 训练可以高度并行。
- 架构统一，适合堆叠层数和扩大宽度。
- 预训练目标简单，能利用海量数据。
- 同一架构可迁移到文本、代码、图像、音频和多模态。

## 最小 PyTorch 形状理解

```python
import torch
import torch.nn as nn

batch_size = 2
seq_len = 4
hidden_size = 8
num_heads = 2

x = torch.randn(batch_size, seq_len, hidden_size)

attention = nn.MultiheadAttention(
    embed_dim=hidden_size,
    num_heads=num_heads,
    batch_first=True
)

output, weights = attention(x, x, x)

print(output.shape)   # [2, 4, 8]
print(weights.shape)  # [2, 4, 4]
```

这里 `x, x, x` 分别作为 query、key、value，这就是 self-attention。

## 学习重点

你应该能解释：

- Q/K/V 分别是什么。
- attention score 为什么要 softmax。
- multi-head 为什么有用。
- position encoding 解决什么问题。
- encoder-only 和 decoder-only 的差异。
- 为什么 GPT 类模型适合生成。

## 下一步

- [GPT 系列](gpt-series.md)
- [BERT 系列](bert-series.md)
- [Prompt 与上下文工程](prompting.md)
- [RAG](rag.md)
