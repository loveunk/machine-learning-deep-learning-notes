# 可信来源

这个列表用于维护“热门 AI 技术文章阅读”的候选来源。收录不等于默认推荐，最终仍需要人工筛选。

## 自动 RSS / Atom 来源

这些来源会被 GitHub Actions 定期抓取，生成候选 issue。自动源优先选择能稳定返回 RSS/Atom、且和本仓库主题强相关的站点。

### 官方实验室与研究机构

| 来源 | RSS / Atom | 主要主题 |
| --- | --- | --- |
| OpenAI News | <https://openai.com/news/rss.xml> | 模型、Agent、Codex、评估、安全 |
| Anthropic Engineering | <https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_engineering.xml> | Claude、Claude Code、Agent 工程、安全边界 |
| Anthropic Research | <https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_research.xml> | 安全、评估、可解释性、前沿研究 |
| Google DeepMind Blog | <https://deepmind.google/blog/rss.xml> | 研究、安全、多模态、Agent、科学智能 |
| Google Research Blog | <https://research.google/blog/rss/> | 研究进展、模型、数据、系统 |
| BAIR Blog | <https://bair.berkeley.edu/blog/feed.xml> | 学术研究、Agent、机器人、多模态 |
| MIT News AI | <https://news.mit.edu/rss/topic/artificial-intelligence2> | AI 研究、机器人、科学智能 |

Anthropic 当前没有稳定官方 RSS，因此自动源暂时使用第三方生成的公开 feed；人工精选时仍应打开 Anthropic 原文确认。

### Agent、编程智能体与工程实践

| 来源 | RSS / Atom | 主要主题 |
| --- | --- | --- |
| LangChain Blog | <https://www.langchain.com/blog/rss.xml> | Agent、LangGraph、RAG、评估、观测 |
| GitHub AI and ML Blog | <https://github.blog/ai-and-ml/feed/> | Copilot、Coding Agent、开发者工作流 |
| Sourcegraph Blog | <https://sourcegraph.com/blog/rss.xml> | 代码智能体、代码检索、软件工程 |
| ColaOS Blog | <https://colaos.ai/rss.xml> | Agent Harness、Agent 记忆、Prompt caching、Sandbox、AI Native 编程 |
| Cloudflare AI Blog | <https://blog.cloudflare.com/tag/ai/rss/> | AI infra、Agent 安全、部署 |
| AWS Machine Learning Blog | <https://aws.amazon.com/blogs/machine-learning/feed/> | Bedrock、RAG、Agent、生产化 |
| Databricks Blog | <https://www.databricks.com/feed> | 数据、RAG、评估、MLOps |
| Together AI Blog | <https://www.together.ai/blog/rss.xml> | 推理、成本、模型服务、训练 |
| Replicate Blog | <https://replicate.com/blog/rss> | 模型部署、多模态、推理 |
| NVIDIA Generative AI Blog | <https://developer.nvidia.com/blog/category/generative-ai/feed/> | 推理、Agent、多模态、GPU 优化 |
| PyTorch Blog | <https://pytorch.org/blog/feed.xml> | 框架、训练、推理、性能 |
| TensorFlow Blog | <https://blog.tensorflow.org/feeds/posts/default?alt=rss> | 框架、部署、移动端、模型优化 |

### 论文与深度个人博客

| 来源 | RSS / Atom | 主要主题 |
| --- | --- | --- |
| arXiv cs.AI | <https://export.arxiv.org/api/query?search_query=cat:cs.AI&sortBy=submittedDate&sortOrder=descending&max_results=50> | Agent、规划、推理、通用 AI |
| arXiv cs.CL | <https://export.arxiv.org/api/query?search_query=cat:cs.CL&sortBy=submittedDate&sortOrder=descending&max_results=50> | LLM、RAG、评测、NLP |
| arXiv cs.LG | <https://export.arxiv.org/api/query?search_query=cat:cs.LG&sortBy=submittedDate&sortOrder=descending&max_results=50> | 训练、学习算法、评测、系统 |
| Lilian Weng Blog | <https://lilianweng.github.io/index.xml> | LLM、Agent、RAG、安全、评估 |
| Chip Huyen Blog | <https://huyenchip.com/feed.xml> | LLM 应用、MLOps、评估、生产化 |
| Simon Willison | <https://simonwillison.net/atom/everything/> | LLM 工具、安全、Prompt Injection、开发者实践 |
| Latent Space | <https://www.latent.space/feed> | AI 工程、Agent、模型生态 |
| The Gradient | <https://thegradient.pub/rss/> | 研究解读、安全、AI 社会影响 |

### 中文来源

| 来源 | RSS / Atom | 主要主题 |
| --- | --- | --- |
| 量子位 | <https://www.qbitai.com/feed> | 大模型、具身智能、多模态、产业 |
| InfoQ 中国 | <https://www.infoq.cn/feed> | 工程实践、架构、开发者、AI 应用 |
| OSChina AI | <https://www.oschina.net/news/rss?tag=ai> | 开源、AI 工具、开发者生态 |
| 36Kr | <https://36kr.com/feed> | AI 产业、创业、投资、产品趋势 |
| 美团技术团队 | <https://tech.meituan.com/feed/> | 大模型、AIGC、多模态、工程落地 |

中文源里新闻和产业内容比例更高，进入候选后需要更严格判断：只有包含可迁移方法、工程细节、论文/代码/产品架构信息的文章，才适合进入正式阅读列表。

## 人工观察来源

这些来源很重要，但 RSS 不稳定、被反爬拦截，或目前没有可靠 feed。适合人工定期检查，不建议直接放进自动 Action：

| 来源 | 地址 | 主要主题 | 暂不自动化原因 |
| --- | --- | --- | --- |
| Hugging Face Blog | <https://huggingface.co/blog> | 开源模型、Agent、Transformers、评测 | 本地抓取出现 TLS EOF，需观察 GitHub Actions 是否稳定 |
| Mistral AI News | <https://mistral.ai/news> | 模型发布、Agent、产品工程 | 本地抓取出现 TLS EOF |
| OpenAI Cookbook | <https://developers.openai.com/cookbook> | 可运行示例、Agent loop、评估、RAG | 当前无稳定 RSS |
| OpenAI Codex Docs | <https://developers.openai.com/codex> | 编程智能体、CLI、IDE、MCP、Skills | 文档型来源，适合人工筛选更新 |
| LlamaIndex Blog | <https://www.llamaindex.ai/blog> | RAG、Agent、数据连接 | 未找到稳定 RSS |
| Cursor Blog | <https://cursor.com/blog> | AI IDE、Coding Agent | 未找到稳定 RSS |
| 机器之心 | <https://www.jiqizhixin.com/> | AI 研究、论文、产业 | 官网 RSS 当前 XML 不稳定 |
| 雷峰网 AI 科技评论 | <https://www.leiphone.com/category/ai> | AI 产业、研究、产品 | RSS 当前返回 500 |
| PaperWeekly | <https://www.paperweekly.site/> | 论文解读、研究动态 | 本地抓取出现 TLS EOF |

## 收录原则

- 优先一手资料。
- 优先工程细节和失败经验。
- 产品发布只有在包含可迁移方法论时才收录。
- 中文来源优先选择有技术细节、论文链接、代码链接或架构拆解的文章。
- 二次解读必须能指向原文、论文或代码。
- 自动候选 issue 只是初筛，不直接写入正式阅读列表。
