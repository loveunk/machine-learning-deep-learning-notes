# BERT 系列

> **理解任务的首选**

---

## BERT vs GPT

| 特性 | BERT | GPT |
|------|------|-----|
| 架构 | Encoder-only | Decoder-only |
| 任务 | 理解（分类、抽取） | 生成 |
| 训练 | Masked LM + Next Sentence Prediction | Causal LM |
| 方向 | 双向 | 单向 |

## BERT 家族

| 模型 | 特点 |
|------|------|
| BERT-Base | 12层，110M参数 |
| BERT-Large | 24层，340M参数 |
| RoBERTa | 改进的BERT |
| ALBERT | 参数共享 |
| DeBERTa | 解耦注意力 |

## 经典任务

### 文本分类

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
```

### 命名实体识别

```python
from transformers import BertForTokenClassification

model = BertForTokenClassification.from_pretrained('bert-base-cased')
```

### 问答系统

```python
from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking')
```

---

**继续学习 [微调方法](fine-tuning.md)！**
