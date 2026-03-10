# CLIP (Contrastive Language-Image Pre-training)

> **图文理解的基石**

---

## 概述

**CLIP** 由 OpenAI 在 2021 年提出，是一个连接文本和图像的模型。

核心思想：**用对比学习让文本和图像映射到同一语义空间**

---

## 工作原理

### 1. 对比学习

```
图像编码器 → 图像向量
文本编码器 → 文本向量

损失函数：
- 正样本对（图像-文本匹配）：拉近
- 负样本对（图像-文本不匹配）：推远
```

### 2. 零样本能力

训练好的 CLIP 可以：
- 不需要额外训练
- 直接对图像分类
- 理解未见过的类别

---

## 实战代码

### 零样本图像分类

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("cat.jpg")
texts = ["a cat", "a dog", "a bird", "a car"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1).tolist()[0]

for text, prob in zip(texts, probs):
    print(f"{text}: {prob:.2%}")
```

### 图文检索

```python
from sklearn.metrics.pairwise import cosine_similarity

# 编码图像
image_features = model.get_image_features(inputs.pixel_values)
# 编码文本
text_features = model.get_text_features(inputs.input_ids)

# 计算相似度
similarity = cosine_similarity(
    image_features.detach().numpy(),
    text_features.detach().numpy()
)
```

---

## 应用场景

- 📸 图像分类
- 🔍 图文检索
- 🎨 图像生成（作为引导）
- 📊 图像-文本对齐

---

## 优缺点

### 优点
- ✅ 零样本能力强
- ✅ 训练数据简单（图像-文本对）
- ✅ 可迁移性强

### 缺点
- ❌ 对细粒度理解有限
- ❌ 计算成本较高
- ❌ 需要大量训练数据

---

**继续学习 [BLIP 系列](blip.md)！**
