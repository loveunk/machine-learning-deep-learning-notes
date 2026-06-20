# Embedding 与向量检索

Embedding 是把文本、图片或其他对象映射成向量的过程。向量检索的目标是：给定一个查询，找到语义上最相关的内容片段。

在 RAG 中，Embedding 负责“找资料”，LLM 负责“基于资料回答”。

## 基本流程

```text
文档
  -> 切分 chunk
  -> 生成 embedding
  -> 写入向量索引

用户问题
  -> 生成 query embedding
  -> top-k 检索
  -> 可选 rerank
  -> 交给 LLM 回答
```

## Chunking

切分文档比选择向量数据库更重要。

常见策略：

| 策略 | 适合 | 风险 |
| --- | --- | --- |
| 固定长度 | 普通文本、快速 baseline | 可能切断语义 |
| 按标题层级 | 文档、手册、课程笔记 | 标题结构不稳定时效果差 |
| 按段落 | 文章、FAQ | 长段落可能超出上下文 |
| 滑动窗口 | 需要保留上下文 | 索引变大、重复内容多 |

建议起点：

- 中文文档：300-800 字一个 chunk。
- 代码文档：按函数、类、文件结构切。
- 表格：保留表头和行语义，不要只切单元格。

## 最小示例

下面示例用 `sentence-transformers` 和 NumPy 展示核心思想，不依赖向量数据库。

```python
import numpy as np
from sentence_transformers import SentenceTransformer

documents = [
    "过拟合是指模型在训练集表现很好，但在测试集表现差。",
    "正则化可以限制模型复杂度，降低过拟合风险。",
    "梯度下降通过沿负梯度方向更新参数来最小化损失函数。"
]

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

doc_vectors = model.encode(documents, normalize_embeddings=True)

query = "模型测试效果很差但训练效果很好是什么问题？"
query_vector = model.encode([query], normalize_embeddings=True)[0]

scores = doc_vectors @ query_vector
top_indices = np.argsort(scores)[::-1][:2]

for idx in top_indices:
    print(float(scores[idx]), documents[idx])
```

## 检索质量怎么判断

不要只看“感觉相关”。应该做一个小评估集：

```text
question: 什么是过拟合？
expected_doc_ids: ["ml-overfitting-001", "regularization-003"]
```

常用指标：

- Recall@k：正确文档是否出现在前 k 个结果中。
- MRR：正确文档排得越靠前越好。
- nDCG：多个相关文档有不同重要性时使用。
- Hit rate：是否至少命中一个相关文档。

## Rerank

Embedding 检索通常是第一阶段召回。Rerank 是第二阶段排序：

1. 先用向量检索取 top 20 或 top 50。
2. 再用 reranker 或 LLM 判断 query 与每个 chunk 的相关性。
3. 取 top 3-8 放进上下文。

Rerank 能改善“召回到了但排序靠后”的问题，但会增加延迟和成本。

## 常见错误

### Chunk 太大

模型拿到的信息太杂，答案容易引用无关内容。

修复：减小 chunk，按标题/段落切分，保留 metadata。

### Chunk 太小

单个片段缺少上下文，检索结果看似相关但无法支撑答案。

修复：增加 overlap，或在检索结果中带上父标题和相邻段落。

### 只存正文不存 metadata

没有文档名、章节、URL、更新时间，答案就很难引用和排查。

建议 metadata：

```json
{
  "doc_id": "guide-001",
  "title": "模型评估指南",
  "section": "过拟合",
  "source": "docs/evaluation.md",
  "updated_at": "2026-06-20"
}
```

### 把向量数据库当核心问题

多数早期 RAG 失败不是数据库问题，而是 chunking、query rewrite、metadata、rerank、prompt 和评估问题。

## 选型建议

| 阶段 | 推荐 |
| --- | --- |
| 学习和原型 | NumPy、FAISS、Chroma |
| 中小型应用 | Chroma、Qdrant、Milvus、pgvector |
| 已有 Postgres | pgvector |
| 大规模/多租户 | 专门向量数据库或托管检索服务 |

## 下一步

- 继续学习 [RAG](rag.md)。
- 如果想理解完整实现，看 [从零实现 RAG](rag-from-scratch.md)。
- 如果要衡量效果，看 [RAG 评估](rag-evaluation.md)。

## 参考资料

- OpenAI Embeddings: https://developers.openai.com/api/docs/guides/embeddings
- Sentence Transformers: https://www.sbert.net/
- FAISS: https://faiss.ai/
