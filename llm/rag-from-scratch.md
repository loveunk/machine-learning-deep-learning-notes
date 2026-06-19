# 从零实现 RAG

本章用最少组件实现一个 RAG。目标不是替代 LlamaIndex 或 LangChain，而是理解 RAG 的关键数据流。

## 最小架构

```text
load documents
  -> split into chunks
  -> embed chunks
  -> retrieve top-k chunks
  -> build prompt with citations
  -> call LLM
  -> return answer and sources
```

## Step 1: 准备文档

```python
documents = [
    {
        "id": "doc-1",
        "title": "过拟合",
        "text": "过拟合是指模型在训练集表现很好，但在测试集或真实数据上表现较差。常见缓解方法包括正则化、增加数据、早停和降低模型复杂度。"
    },
    {
        "id": "doc-2",
        "title": "梯度下降",
        "text": "梯度下降是一种优化方法。它沿着损失函数梯度的反方向更新参数，从而逐步降低损失。"
    }
]
```

真实项目中要保留 `source`、`url`、`section`、`updated_at` 等 metadata。

## Step 2: 切分 chunk

```python
def split_text(text: str, chunk_size: int = 120, overlap: int = 20) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


chunks = []
for doc in documents:
    for i, text in enumerate(split_text(doc["text"])):
        chunks.append({
            "chunk_id": f"{doc['id']}-{i}",
            "doc_id": doc["id"],
            "title": doc["title"],
            "text": text
        })
```

## Step 3: 生成向量

```python
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

texts = [chunk["text"] for chunk in chunks]
vectors = embedder.encode(texts, normalize_embeddings=True)
```

## Step 4: 检索

```python
import numpy as np


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    query_vector = embedder.encode([query], normalize_embeddings=True)[0]
    scores = vectors @ query_vector
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        item = dict(chunks[idx])
        item["score"] = float(scores[idx])
        results.append(item)
    return results
```

## Step 5: 构造上下文

```python
def build_context(results: list[dict]) -> str:
    lines = []
    for i, item in enumerate(results, start=1):
        lines.append(
            f"[{i}] title={item['title']} source={item['chunk_id']}\n{item['text']}"
        )
    return "\n\n".join(lines)
```

## Step 6: 调用模型

```python
from openai import OpenAI

client = OpenAI()


def answer(query: str) -> str:
    results = retrieve(query)
    context = build_context(results)

    prompt = f"""
你是严谨的知识库问答助手。

规则：
1. 仅依据 context 回答。
2. 如果 context 没有答案，回答“资料不足”。
3. 每个关键结论都要标注引用编号，例如 [1]。

context:
{context}

question:
{query}
"""

    response = client.responses.create(
        model="your-model",
        input=prompt
    )
    return response.output_text
```

## 完整流程检查

用这个问题测试：

```python
print(answer("什么是过拟合？怎么缓解？"))
```

理想答案应该：

- 说明过拟合是训练好、测试差。
- 给出缓解方法。
- 引用相关 chunk。
- 不编造 context 中没有的方法。

## 什么时候引入框架

从零实现能帮你理解核心流程，但真实项目通常需要框架：

| 需求 | 可以引入 |
| --- | --- |
| 多格式文档解析 | LlamaIndex、LangChain document loaders |
| 复杂索引 | LlamaIndex、向量数据库 |
| 多步骤工作流 | LangGraph、LlamaIndex Workflows |
| 评估和观测 | Ragas、DeepEval、LangSmith、Arize Phoenix、Langfuse |

## 常见改进点

### Query rewrite

用户问题可能太短或上下文不足，可以先让模型改写查询：

```text
把用户问题改写成适合检索知识库的查询，保留原意，不添加新事实。
```

### Hybrid search

向量检索适合语义相似，关键词检索适合精确术语、代码符号、ID、日期。实际系统经常组合 BM25 + embedding。

### Rerank

先召回更多 chunk，再用 reranker 排序，可以提升引用质量。

### Citation check

生成答案后再检查每个引用是否真的支持对应句子。

## 下一步

- 理解架构边界：[RAG](rag.md)
- 建立评估集：[RAG 评估](rag-evaluation.md)
- 需要工具调用时学习：[Agent 工具调用](agent-tools.md)
