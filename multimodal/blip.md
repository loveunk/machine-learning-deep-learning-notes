# BLIP 系列

> **理解 + 生成双任务的多模态模型**

---

## 概述

**BLIP** (Bootstrapping Language-Image Pre-training) 由 Salesforce 在 2022 年提出。

创新点：**同时训练理解和生成能力**

---

## BLIP vs CLIP

| 特性 | CLIP | BLIP |
|------|------|------|
| 任务 | 对比学习 | 理解 + 生成 |
| 能力 | 图文匹配 | 图文理解 + 描述生成 |
| 架构 | 对比学习 | 多任务学习 |

---

## BLIP 家族

| 模型 | 发布时间 | 特点 |
|------|----------|------|
| BLIP | 2022.01 | 首次提出 |
| BLIP-2 | 2023.01 | 冻结 ViT + Q-Former |
| BLIP-3 | 2023 | 更大模型，更好性能 |

---

## BLIP-2 核心创新

### Q-Former 架构

```
图像 → 冻结的 ViT → Q-Former → LLM
文本 → 冻结的 LLM

优势：
- 只训练 Q-Former（轻量级）
- 利用强大的预训练模型
- 训练成本大幅降低
```

---

## 实战代码

### 图像描述生成

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image = Image.open("image.jpg")

# 无条件描述
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print(f"Caption: {caption}")

# 条件描述
inputs = processor(image, text="a photo of", return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print(f"Caption: {caption}")
```

### 视觉问答 (VQA)

```python
from transformers import BlipProcessor, BlipForQuestionAnswering

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

image = Image.open("image.jpg")
question = "What color is the cat?"

inputs = processor(image, question, return_tensors="pt")
out = model.generate(**inputs)
answer = processor.decode(out[0], skip_special_tokens=True)
print(f"Answer: {answer}")
```

---

## 应用场景

- 📝 图像描述生成
- ❓ 视觉问答
- 🔍 图文检索
- 📊 图像-文本对过滤

---

**继续学习 [LLaVA](llava.md)！**
