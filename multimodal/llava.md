# LLaVA (Large Language-and-Vision Assistant)

> **连接视觉和语言模型，实现多模态对话**

---

## 概述

**LLaVA** 于 2023 年提出，是一个连接视觉编码器和 LLM 的多模态模型。

核心思想：**将图像向量作为 LLM 的输入，实现图文对话**

---

## 架构

```
图像 → CLIP ViT → 线性投影 → LLM Token
文本 → LLM Token

拼接后输入 LLM
```

### 关键组件

1. **视觉编码器**：CLIP ViT
2. **投影层**：线性层 + MLP
3. **语言模型**：Vicuna / LLaMA

---

## 训练方法

### 两阶段训练

1. **预训练**
   - 冻结视觉编码器和 LLM
   - 只训练投影层
   - 使用图像-文本对

2. **微调**
   - 解冻投影层和部分 LLM 参数
   - 使用多模态对话数据
   - 端到端优化

---

## 实战代码

### 多模态对话

```python
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

prompt = "USER: <image>\nWhat's in this image?\nASSISTANT:"
image = Image.open("image.jpg")

inputs = processor(text=prompt, images=image, return_tensors="pt")

output = model.generate(**inputs, max_new_tokens=100)
response = processor.decode(output[0], skip_special_tokens=True)
print(response)
```

### 复杂推理

```python
prompt = """USER: <image>
请详细描述这张图片中的：
1. 主要物体
2. 场景背景
3. 颜色和风格
4. 可能的故事情节

ASSISTANT:"""

inputs = processor(text=prompt, images=image, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
response = processor.decode(output[0], skip_special_tokens=True)
print(response)
```

---

## LLaVA 家族

| 模型 | 参数量 | 特点 |
|------|--------|------|
| LLaVA-1.5 | 7B/13B | 轻量级，效果好 |
| LLaVA-NeXT | 34B | 更强推理能力 |
| LLaVA-1.6 | 72B | 最强性能 |

---

## 应用场景

- 💬 多模态对话
- 🔍 图像理解
- 🎨 视觉推理
- 📊 OCR + 语义理解
- 📐 数学图像理解

---

## 优势

- ✅ 强大的对话能力
- ✅ 灵活的多模态理解
- ✅ 可以处理复杂指令
- ✅ 开源且可部署

---

## 限制

- ❌ 计算成本高（需要大模型）
- ❌ 长文本推理仍有挑战
- ❌ 视觉细节理解有限

---

## 相关资源

- [LLaVA GitHub](https://github.com/haotian-liu/LLaVA)
- [LLaVA Paper](https://arxiv.org/abs/2304.08485)
- [Hugging Face Models](https://huggingface.co/models?search=llava)

---

**恭喜！你已经完成了多模态的核心学习！** 🎉
