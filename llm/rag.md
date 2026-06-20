# RAG（检索增强生成）

RAG（Retrieval-Augmented Generation）把“检索”和“生成”结合起来：先从外部知识库找资料，再让 LLM 基于资料回答。

## 为什么需要 RAG

LLM 的常见问题：

- 不知道训练数据之后的新信息。
- 不知道企业内部文档。
- 会编造看似合理的答案。
- 难以给出可追溯引用。

RAG 的价值：

- 把知识从模型参数中移到可更新的文档库。
- 回答可以附引用。
- 文档更新不需要重新训练模型。
- 可以单独优化检索和生成。

## 基本架构

```text
用户问题
  -> 查询改写（可选）
  -> 向量检索 / 关键词检索 / 混合检索
  -> rerank（可选）
  -> 拼接上下文
  -> LLM 生成答案
  -> 引用和事实校验
```

## 最小 RAG 流程

1. 收集文档。
2. 切分成 chunk。
3. 生成 embedding。
4. 写入向量索引。
5. 对用户问题做检索。
6. 把 top-k 结果放进 prompt。
7. 要求模型仅依据上下文回答并给引用。
8. 评估检索和生成质量。

## RAG 组件

| 组件 | 作用 | 常见问题 |
| --- | --- | --- |
| 文档解析 | 读取 PDF、网页、Markdown、表格 | 表格、图片、页眉页脚噪声 |
| chunking | 将文档切成可检索片段 | 太大噪声多，太小缺上下文 |
| embedding | 把 chunk 转成向量 | 领域术语、中文、代码符号 |
| 检索 | 找相关 chunk | 召回不足、排序差 |
| rerank | 重新排序候选结果 | 成本和延迟增加 |
| prompt | 约束模型基于资料回答 | 幻觉、引用错位 |
| eval | 衡量效果 | 没有标注集、只凭主观判断 |

## 什么时候不该用 RAG

| 场景 | 更好的方案 |
| --- | --- |
| 只是要求固定输出格式 | Prompt 或结构化输出 |
| 需要模型学习稳定风格 | 微调 |
| 需要执行外部动作 | Agent 工具调用 |
| 文档质量很差且无人维护 | 先治理数据 |
| 答案必须来自结构化数据库 | SQL/工具查询，必要时再由 LLM 解释 |

## 常见失败

### 检索不到

原因可能是：

- chunk 切分不合理。
- query 太短或表达和文档不一致。
- embedding 模型不适合语言或领域。
- 只用向量检索，漏掉精确关键词。

解决：

- 调整 chunk size 和 overlap。
- 加 query rewrite。
- 尝试 hybrid search。
- 建立 Recall@k 评估。

### 检索到了但答错

原因可能是：

- top-k 太多，噪声污染上下文。
- prompt 没限制仅依据 context。
- 模型把多个片段错误拼接。
- 引用粒度太粗。

解决：

- 加 rerank。
- 降低 top-k 或压缩 context。
- 要求逐条引用。
- 做 faithfulness 检查。

### 答案没有引用

解决：

- 在 prompt 中要求每个关键结论标注引用。
- 检索结果中保留稳定 `source_id`。
- 输出后做引用存在性检查。

## 推荐学习顺序

1. [Embedding 与向量检索](embeddings.md)
2. [从零实现 RAG](rag-from-scratch.md)
3. [RAG 评估](rag-evaluation.md)
4. [Prompt 与上下文工程](prompting.md)

## 参考资料

- LlamaIndex Docs: https://docs.llamaindex.ai/
- LangChain Retrieval: https://docs.langchain.com/oss/python/langchain/retrieval
- Ragas: https://docs.ragas.io/
