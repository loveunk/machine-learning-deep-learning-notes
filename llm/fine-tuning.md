# 微调方法

微调的目标不是把事实塞进模型，而是让模型在某类任务上形成更稳定的行为、格式、风格或决策边界。

## 先判断是否真的需要微调

| 需求 | 优先方案 |
| --- | --- |
| 回答企业文档中的事实 | RAG |
| 输出格式不稳定 | Prompt、结构化输出、校验 |
| 需要实时数据 | 工具调用 |
| 固定分类标准 | 微调或小模型分类器 |
| 固定写作风格 | 微调 |
| 领域术语表达稳定 | 微调 |
| 降低推理成本 | 微调小模型或蒸馏 |

如果问题是“模型不知道某个事实”，通常先做 RAG。如果问题是“模型知道信息但行为不稳定”，才考虑微调。

## 微调类型

### Full fine-tuning

更新全部参数。

适合：

- 数据量大。
- 算力充足。
- 需要深度改变模型行为。

风险：

- 成本高。
- 容易过拟合。
- 可能损伤通用能力。

### LoRA

LoRA（Low-Rank Adaptation）冻结原模型，只训练少量低秩适配参数。

适合：

- 单任务或少数任务适配。
- 显存有限。
- 希望保留基础模型能力。

示例：

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

### QLoRA

QLoRA = 4-bit 量化基础模型 + LoRA 训练。

适合：

- 单卡训练较大模型。
- 原型实验。

示例：

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

## 数据比算法更重要

微调数据要关注：

- 输入是否代表真实场景。
- 输出是否符合最终验收标准。
- 是否覆盖失败和边界案例。
- 是否有冲突标注。
- 是否混入低质量自动生成内容。

一个常见格式：

```json
{
  "messages": [
    {"role": "system", "content": "你是严谨的机器学习助教。"},
    {"role": "user", "content": "什么是过拟合？"},
    {"role": "assistant", "content": "过拟合是模型在训练集表现很好，但在测试集或真实数据上表现较差的现象。"}
  ]
}
```

## 微调流程

1. 明确任务和验收标准。
2. 建立 baseline：prompt/RAG/强模型表现如何。
3. 准备训练集、验证集、测试集。
4. 清洗和去重数据。
5. 训练 LoRA 或 QLoRA。
6. 在固定评估集上比较。
7. 检查过拟合、幻觉、格式和安全问题。
8. 小流量上线，持续收集失败样例。

## 评估指标

| 任务 | 指标 |
| --- | --- |
| 分类 | accuracy、F1、confusion matrix |
| 信息抽取 | 字段级 precision/recall、JSON 解析率 |
| 风格生成 | 人工偏好、格式符合率 |
| 对话 | 任务完成率、拒答正确率、安全违规率 |
| 代码 | 单元测试通过率、静态检查 |

微调前后必须用同一评估集比较，否则无法证明有效。

## 常见错误

### 用微调记事实

事实会变化，且模型不保证逐字记忆。企业知识优先 RAG。

### 数据太少且质量不稳定

几十条样例可能只会让模型学到噪声。先提升数据质量和 prompt baseline。

### 没有负例

只给正确示例，模型不一定知道什么时候拒答、什么时候转人工。

### 没有保留测试集

训练集表现好不代表上线效果好。

### 只看主观体验

必须有固定评估集和上线监控。

## 微调和 RAG 的组合

常见组合：

- RAG 提供事实。
- 微调稳定回答格式、语气和任务策略。
- 工具调用执行实时查询。

例如客服助手：

```text
RAG: 查政策和订单知识
微调: 学会公司回复风格和分类标准
工具: 查询订单、创建工单
```

## 下一步

- 如果只是知识缺失，回到 [RAG](rag.md)。
- 如果要比较效果，先学 [RAG 评估](rag-evaluation.md) 中的评估方法。
- 如果要让模型调用外部动作，学习 [Agent 工具调用](agent-tools.md)。

## 参考资料

- Hugging Face PEFT: https://huggingface.co/docs/peft/index
- Hugging Face TRL: https://huggingface.co/docs/trl/index
- OpenAI Fine-tuning: https://developers.openai.com/api/docs/guides/fine-tuning
