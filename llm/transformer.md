# Transformer 架构详解

> **理解现代 LLM 的基石**

---

## 概述

Transformer 是 2017 年 Google 提出的架构，彻底改变了 NLP 和深度学习领域。它是 GPT、BERT、LLaMA 等所有大语言模型的基础。

## 核心组件

### 1. 自注意力机制 (Self-Attention)

```
Query, Key, Value → Attention Score → Weighted Sum
```

### 2. 多头注意力 (Multi-Head Attention)

- 多个注意力头并行工作
- 每个头关注不同的语义关系

### 3. 位置编码 (Positional Encoding)

- 让模型理解序列顺序
- Sinusoidal 或 Learnable

## 架构变体

| 架构 | 特点 |
|------|------|
| **Encoder-only** | BERT - 适合理解任务 |
| **Decoder-only** | GPT - 适合生成任务 |
| **Encoder-Decoder** | T5 - 适合翻译任务 |

## 为什么 Decoder-only 胜出？

- 训练效率高
- 生成能力更强
- 可扩展性更好

## 实战代码

```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

    def forward(self, values, keys, query, mask):
        # Attention 计算
        # ...
        pass
```

---

**继续学习 [GPT 系列](gpt-series.md)！**
