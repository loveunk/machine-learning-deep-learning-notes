# 微调方法

> **让通用模型适应你的任务**

---

## 为什么微调？

通用模型很强大，但：
- 不知道你的领域知识
- 不懂你的业务逻辑
- 可能风格不匹配

微调 = 让模型学你的数据

---

## 微调方法

### 1. Full Fine-tuning

更新所有参数

```python
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

### 2. LoRA (Low-Rank Adaptation)

只训练少量参数

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)
```

### 3. QLoRA

4-bit 量化 + LoRA

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
)
```

---

## 什么时候用哪种方法？

| 场景 | 推荐 |
|------|------|
| 大数据 + 有算力 | Full Fine-tuning |
| 小数据 + 有限算力 | LoRA |
| 单卡 + 大模型 | QLoRA |

---

**继续学习 [RAG](rag.md)！**
