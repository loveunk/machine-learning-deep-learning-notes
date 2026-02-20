# RAG（检索增强生成）

> **解决幻觉问题，增强模型能力**

---

## 什么是 RAG？

**RAG = Retrieval-Augmented Generation**

结合：
- **检索**（从知识库找相关信息）
- **生成**（用 LLM 生成答案）

---

## 为什么需要 RAG？

### LLM 的问题

- ❌ 知识截止日期
- ❌ 会产生幻觉
- ❌ 缺少私有数据

### RAG 的优势

- ✅ 实时信息
- ✅ 有据可依
- ✅ 私有数据支持
- ✅ 可解释性强

---

## RAG 架构

```
用户提问
    ↓
向量检索
    ↓
获取相关文档
    ↓
拼接上下文 + 问题
    ↓
LLM 生成答案
    ↓
返回结果
```

---

## 实战代码

### 1. 向量化

```python
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')

documents = ["文档1内容", "文档2内容", ...]
document_embeddings = embedder.encode(documents)
```

### 2. 检索

```python
query = "用户的问题"
query_embedding = embedder.encode(query)

from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity([query_embedding], document_embeddings)
top_k_indices = similarities[0].argsort()[-3:][::-1]
```

### 3. 生成

```python
from openai import OpenAI

client = OpenAI()

context = "\n".join([documents[i] for i in top_k_indices])

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": f"根据以下信息回答：\n{context}"},
        {"role": "user", "content": query}
    ]
)
```

---

## RAG 工具

| 工具 | 特点 |
|------|------|
| **LangChain** | 最流行的框架 |
| **LlamaIndex** | 数据连接器丰富 |
| **Haystack** | 企业级 |
| **Chroma** | 向量数据库 |

---

**继续学习 [AI Agent](agent.md)！**
