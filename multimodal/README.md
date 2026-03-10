# 多模态模型 (Multimodal)

> **从单一模态到多模态，AI 正在迈向真正理解世界。**

---

## 🎯 学习目标

用 **2 小时** 理解：
1. 什么是多模态模型
2. 主流模型和架构
3. 如何使用多模态模型
4. 最新进展和应用

---

## 📚 快速模式（30 分钟）

### 什么是多模态？

**多模态 = 处理多种类型的数据**

常见模态：
- 📝 **文本**（Text）
- 🖼️ **图像**（Image）
- 🎬 **视频**（Video）
- 🔊 **音频**（Audio）
- 📊 **数据**（Structured Data）

### 为什么需要多模态？

**人类是多模态的：**
- 看图 + 读文 = 更好理解
- 听声音 + 看表情 = 准确判断情绪
- 触觉 + 视觉 = 更好感知世界

**AI 也应该一样！**

### 应用场景

| 任务 | 输入 | 输出 | 例子 |
|------|------|------|------|
| **图文理解** | 图像 + 文本 | 文本/分类 | "这张图片里有什么？" |
| **图像生成** | 文本 | 图像 | "画一只猫" |
| **视频理解** | 视频 + 文本 | 文本/分类 | "这段视频在讲什么？" |
| **音频理解** | 音频 + 文本 | 文本/分类 | "这首歌的情绪是什么？" |

### 实战：用 CLIP 做图文检索

```python
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

# 加载模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 输入
image = Image.open("cat.jpg")
texts = ["a cat", "a dog", "a car"]

# 处理
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# 推理
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # 图像和文本的相似度
probs = logits_per_image.softmax(dim=1).tolist()[0]

# 结果
for text, prob in zip(texts, probs):
    print(f"{text}: {prob:.2%}")
# Output: a cat: 95%, a dog: 3%, a car: 2%
```

---

## 📖 深度模式（2 小时）

### 主流多模态模型

#### 1. CLIP (Contrastive Language-Image Pre-training)

**提出：** OpenAI (2021)

**核心思想：** 对比学习，把图像和文本映射到同一空间

```
图像编码器 → 图像向量
文本编码器 → 文本向量

对比损失：同一对的向量应该接近，不同对的向量应该远离
```

**应用：**
- 零样本图像分类
- 图文检索
- 图像生成引导

#### 2. BLIP (Bootstrapping Language-Image Pre-training)

**提出：** Salesforce (2022)

**核心创新：**
- 理解 + 生成 双任务
- CAPTION (生成描述)
- RETRIEVAL (检索)

**应用：**
- 图像描述
- 图文检索
- 视觉问答

#### 3. LLaVA (Large Language-and-Vision Assistant)

**提出：** 2023

**核心思想：** 连接视觉编码器和 LLM

```
图像 → CLIP ViT → 图像向量
文本 → LLM

将图像向量作为 LLM 的输入
```

**应用：**
- 多模态对话
- 图像理解
- 视觉推理

#### 4. GPT-4V

**提出：** OpenAI (2023)

**特点：**
- 强大的多模态能力
- 理解复杂场景
- 绘图和标注

#### 5. 其他重要模型

| 模型 | 公司 | 特点 |
|------|------|------|
| **DALL·E** | OpenAI | 文本生成图像 |
| **Stable Diffusion** | Stability AI | 开源图像生成 |
| **Midjourney** | Midjourney | 艺术风格图像生成 |
| **Flamingo** | DeepMind | 少样本视觉学习 |

---

## 架构对比

### CLIP 架构

```
图像 → Vision Transformer (ViT) → 图像向量
文本 → Text Transformer → 文本向量

对比损失：拉近相关对，推开不相关对
```

### BLIP-2 架构（2023）

```
图像 → 冻结的 ViT → Q-Former → LLM
文本 → 冻结的 LLM

只训练 Q-Former（轻量级适配器）
```

### LLaVA 架构

```
图像 → CLIP ViT → 线性投影 → LLM Token
文本 → LLM Token

拼接后输入 LLM
```

---

## 实战：构建多模态应用

### 项目 1: 图文问答

```python
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image

# 加载模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# 加载图像
image = Image.open("image.jpg")

# 提问
question = "What is in this image?"

# 处理
inputs = processor(image, question, return_tensors="pt")

# 推理
out = model.generate(**inputs)
answer = processor.decode(out[0], skip_special_tokens=True)

print(answer)
```

### 项目 2: 图像生成

```python
import torch
from diffusers import StableDiffusionPipeline

# 加载模型
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# 生成
prompt = "A cat sitting on a couch, digital art"
image = pipe(prompt).images[0]

# 保存
image.save("cat.png")
```

### 项目 3: 图像描述

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image = Image.open("image.jpg")

inputs = processor(image, return_tensors="pt")

out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print(caption)
```

---

## 多模态模型的应用

### 1. 内容创作

- 根据文本生成图像
- 图像编辑
- 视频生成

### 2. 搜索和检索

- 以图搜图
- 文本搜图
- 图文混合检索

### 3. 医疗诊断

- 医学影像分析
- 结合病历诊断
- 辅助手术

### 4. 教育

- 图文教学
- 交互式学习
- 视频理解

### 5. 自动驾驶

- 场景理解
- 多传感器融合
- 决策辅助

---

## 多模态的挑战

| 挑战 | 说明 |
|------|------|
| **对齐** | 如何让不同模态对齐到同一语义空间 |
| **数据** | 高质量多模态数据稀缺 |
| **计算** | 处理图像和视频需要大量计算 |
| **评估** | 如何评估多模态理解能力 |
| **偏见** | 可能继承多个模态的偏见 |

---

## 最新进展 (2024-2025)

### 1. 视频多模态

- 视频理解（Video-LLaVA、InternVideo）
- 视频生成（Sora、Runway）

### 2. 音频多模态

- 语音理解（Whisper）
- 语音合成（TTS）
- 音频分类

### 3. 3D 多模态

- 3D 物体识别
- 3D 生成
- 场景理解

### 4. 传感器融合

- 激光雷达 + 摄像头
- 多模态自动驾驶

---

## 如何选择模型？

| 任务 | 推荐模型 |
|------|----------|
| **图文检索** | CLIP, BLIP |
| **图像理解** | LLaVA, GPT-4V |
| **图像生成** | Stable Diffusion, DALL·E |
| **视频理解** | Video-LLaVA, InternVideo |
| **音频理解** | Whisper |

---

## 💡 学习建议

### 必须理解

- ✅ 对比学习的概念
- ✅ 视觉编码器（ViT）
- ✅ 如何连接视觉和语言模型
- ✅ 零样本学习

### 可以简化

- 💡 具体的数学推导
- 💡 训练细节（预训练 + 微调）

---

## 📝 总结

### 关键要点

- ✅ 多模态 = 处理多种数据类型
- ✅ CLIP 是基础架构
- ✅ LLaVA 连接视觉和 LLM
- ✅ 应用广泛（搜索、生成、理解）

### 下一步

- [ ] 用 CLIP 做图文检索
- [ ] 用 Stable Diffusion 生成图像
- [ ] 尝试 LLaVA 进行图文对话
- [ ] 关注最新的多模态论文

---

## 🔗 相关资源

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [BLIP Paper](https://arxiv.org/abs/2201.12086)
- [LLaVA Paper](https://arxiv.org/abs/2304.08485)
- [Hugging Face Multimodal](https://huggingface.co/docs/transformers/multimodal)

---

**继续学习 [LLM](../llm/README.md) 或 [深度学习](../deep-learning/README.md)！** 🚀
