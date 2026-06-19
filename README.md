# 深度学习与大模型学习路径（2026）

面向中文学习者的 AI 学习路线图：从机器学习、深度学习基础，到大语言模型、RAG、Agent 和多模态工程实践。

本仓库的定位不是资料堆砌，而是帮你回答三个问题：

1. 现在应该先学什么。
2. 每个主题学到什么程度算够用。
3. 如何从概念走到可运行的 AI 应用。

## 学习理念

AI 工具普及后，学习方式应该从“先补完所有理论”变成“先跑通最小闭环，再按问题补理论”。

本仓库采用四个原则：

- 先实践后理论：先让模型、数据和代码跑起来。
- 只深挖关键概念：向量化、梯度、过拟合、Transformer、检索、工具调用、评估。
- 按需回溯：遇到诊断、调优、评估问题时再回到数学和经典算法。
- 关注工程闭环：能部署、能评估、能定位错误，比只看懂概念更重要。

## 该从哪里开始

| 目标 | 推荐入口 | 结果 |
| --- | --- | --- |
| 零基础进入 AI | [30 天新人路线](#30-天新人路线) | 会用 Python、sklearn、PyTorch 和 LLM API 完成基础任务 |
| 转向 LLM 工程 | [LLM 工程师路线](#llm-工程师路线) | 能构建 Prompt、Embedding、RAG、评估和微调流程 |
| 学 Agent | [Agent 工程路线](#agent-工程路线) | 理解工具调用、工作流、记忆、guardrails 和上线风险 |
| 补经典深度学习 | [经典深度学习补课路线](#经典深度学习补课路线) | 补齐 CNN、RNN、优化、正则化和训练策略 |

## 30 天新人路线

适合：从零开始，希望快速进入 AI/ML 领域。

| 周期 | 内容 | 学到什么程度 |
| --- | --- | --- |
| 第 1 周 | Python、NumPy、Pandas、Matplotlib | 能读写数据、做基础清洗和可视化 |
| 第 2 周 | 线性回归、逻辑回归、聚类、模型评估 | 能用 sklearn 完成一个小型监督学习任务 |
| 第 3 周 | 神经网络、PyTorch、CNN/RNN 基础 | 能训练和调试一个小型深度学习模型 |
| 第 4 周 | LLM API、Prompt、Embedding、RAG 入门 | 能调用模型并做一个最小知识库问答 |

建议顺序：

1. [Python 基础](python/python-basic/README.md)
2. [NumPy](python/numpy/README.md)
3. [Pandas](python/pandas/README.md)
4. [机器学习绪论](machine-learning/machine-learning-intro.md)
5. [线性回归](machine-learning/linear-regression.md)
6. [逻辑回归](machine-learning/logistic-regression.md)
7. [深度学习基础](deep-learning/1.deep-learning-basic.md)
8. [PyTorch 基础](pytorch/pytorch_basic.md)
9. [LLM 入门](llm/intro.md)
10. [Prompt 与上下文工程](llm/prompting.md)

## LLM 工程师路线

适合：已有 Python/深度学习基础，希望构建 RAG、微调和大模型应用。

| 阶段 | 主题 | 内容 |
| --- | --- | --- |
| 1 | 模型调用 | [LLM 入门](llm/intro.md)、[API 与模型选型](llm/api-and-models.md) |
| 2 | 控制输出 | [Prompt 与上下文工程](llm/prompting.md)、结构化输出、few-shot |
| 3 | 理解架构 | [Transformer](llm/transformer.md)、[GPT 系列](llm/gpt-series.md)、[BERT 系列](llm/bert-series.md) |
| 4 | 检索增强 | [Embedding 与向量检索](llm/embeddings.md)、[RAG](llm/rag.md)、[从零实现 RAG](llm/rag-from-scratch.md) |
| 5 | 评估优化 | [RAG 评估](llm/rag-evaluation.md)、成本、延迟、召回率、答案可信度 |
| 6 | 定制模型 | [微调方法](llm/fine-tuning.md)、LoRA、QLoRA、数据集质量 |

学习完成标志：

- 能解释 Prompt、RAG、微调三者的适用边界。
- 能为一个业务问题选择“只提示词 / RAG / 微调 / Agent”。
- 能用评估集证明改动是否真的变好。

## Agent 工程路线

适合：希望从 Chatbot 升级到能调用工具、处理多步骤任务的智能体系统。

| 阶段 | 主题 | 内容 |
| --- | --- | --- |
| 1 | 基本概念 | [AI Agent 总览](llm/agent.md) |
| 2 | 工具调用 | [工具调用与结构化输出](llm/agent-tools.md) |
| 3 | 工作流 | [Agent 工作流模式](llm/agent-workflows.md) |
| 4 | 记忆与状态 | 短期上下文、长期记忆、任务状态、人工介入 |
| 5 | 生产化 | [Agent 生产化](llm/agent-production.md)、guardrails、tracing、eval、权限 |

Agent 学习的重点不是“让模型自己想办法”，而是把任务边界、工具契约、状态转移、失败处理和评估设计清楚。

## 经典深度学习补课路线

适合：已经在用 LLM，但希望补齐深度学习底层能力。

| 阶段 | 内容 |
| --- | --- |
| 神经网络基础 | [深度学习基础](deep-learning/1.deep-learning-basic.md)、[神经网络](machine-learning/neural-networks.md) |
| 优化与调参 | [实践层面](deep-learning/2.improving-deep-neural-networks-1.practical-aspects.md)、[优化算法](deep-learning/2.improving-deep-neural-networks-2.optimization-algorithms.md)、[超参数调试](deep-learning/2.improving-deep-neural-networks-3.pyperparameter-tuning.md) |
| CNN | [CNN 基础](deep-learning/4.convolutional-neural-network-1.foundations-of-cnn.md)、[经典网络](deep-learning/4.convolutional-neural-network-2.deep-convolutional-models.md)、[目标检测](deep-learning/4.convolutional-neural-network-3.object-detection.md) |
| 序列模型 | [RNN](deep-learning/5.sequence-model-1.recurrent-neural-netoworks.md)、[词嵌入](deep-learning/5.sequence-model-2.nlp-and-word-embeddings.md)、[注意力机制](deep-learning/5.sequence-model-3.sequence-models-and-attention-machanism.md) |

## 每章学习方式

每个主题建议按两层学习：

快速模式，15-30 分钟：

- 知道它解决什么问题。
- 跑通一个最小示例。
- 理解输入、输出和常见失败模式。
- 能用 AI 编码助手生成相似代码，并能检查关键逻辑。

深度模式，1-3 小时：

- 理解核心公式或架构。
- 能解释关键参数为什么影响结果。
- 能设计评估指标。
- 能定位错误并提出改进方案。

## 目录结构

### 数学基础

- [微积分](math/calculus.md)
- [线性代数](math/linear-algebra.md)
- [PCA](math/pca.md)

### Python 与数据处理

- [Python 基础](python/python-basic/README.md)
- [NumPy](python/numpy/README.md)
- [Pandas](python/pandas/README.md)
- [Matplotlib](python/Matplotlib/README.md)
- [Scikit-Learn](python/Sklearn/README.md)

### 机器学习算法

- [机器学习绪论](machine-learning/machine-learning-intro.md)
- [线性回归](machine-learning/linear-regression.md)
- [逻辑回归](machine-learning/logistic-regression.md)
- [神经网络](machine-learning/neural-networks.md)
- [SVM](machine-learning/svm.md)
- [聚类](machine-learning/clustering.md)
- [数据降维](machine-learning/dimension-reduction.md)
- [异常检测](machine-learning/anomaly-detection.md)
- [推荐系统](machine-learning/recommender-system.md)
- [机器学习系统设计](machine-learning/advice-for-appying-and-system-design.md)

### 深度学习

- [深度学习总览](deep-learning/README.md)
- [深度学习基础](deep-learning/1.deep-learning-basic.md)
- [优化与调参](deep-learning/2.improving-deep-neural-networks-2.optimization-algorithms.md)
- [机器学习策略](deep-learning/3.structuring-machine-learning-1.ml-strategy.md)
- [CNN](deep-learning/4.convolutional-neural-network-1.foundations-of-cnn.md)
- [RNN 与注意力](deep-learning/5.sequence-model-3.sequence-models-and-attention-machanism.md)
- [PyTorch 基础](pytorch/pytorch_basic.md)

### 大语言模型与 Agent

- [LLM 学习路径](llm/README.md)
- [LLM 入门](llm/intro.md)
- [API 与模型选型](llm/api-and-models.md)
- [Prompt 与上下文工程](llm/prompting.md)
- [Transformer](llm/transformer.md)
- [Embedding 与向量检索](llm/embeddings.md)
- [RAG](llm/rag.md)
- [从零实现 RAG](llm/rag-from-scratch.md)
- [RAG 评估](llm/rag-evaluation.md)
- [微调方法](llm/fine-tuning.md)
- [AI Agent](llm/agent.md)
- [Agent 工具调用](llm/agent-tools.md)
- [Agent 工作流](llm/agent-workflows.md)
- [Agent 生产化](llm/agent-production.md)

### 多模态

- [多模态总览](multimodal/README.md)
- [CLIP](multimodal/clip.md)
- [BLIP](multimodal/blip.md)
- [LLaVA](multimodal/llava.md)

### 实践与竞赛

- [项目实战](projects/README.md)
- [Kaggle](competitions/kaggle.md)

## 推荐工具

| 工具 | 用途 |
| --- | --- |
| Google Colab / Kaggle Notebook | 低成本运行实验 |
| VS Code / Cursor / PyCharm | 本地开发 |
| PyTorch | 深度学习训练 |
| Hugging Face Transformers | 开源模型加载、推理、微调 |
| LlamaIndex / LangChain / LangGraph | RAG、工具调用和 Agent 工作流 |
| OpenAI / Anthropic / Google / Qwen API | 托管模型调用 |

## 贡献方式

欢迎提交 Issue 或 PR。建议优先贡献：

- 修正文档错误和失效链接。
- 给 LLM/RAG/Agent 章节补充可验证的最小示例。
- 补充评估方法、失败案例和排错经验。
- 改善中文表达和学习路径衔接。

贡献前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)，长期规划见 [ROADMAP.md](ROADMAP.md)。

## 更新记录

- 2026-06：重构为“深度学习 + 大模型 + Agent”学习路径，补充 LLM 工程化章节。
- 2025-02：重构学习路径，加入现代化学习理念。
- 2023：添加 LLM 和多模态内容。
- 2016：初始版本。

## License

MIT License
