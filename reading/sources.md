# 可信来源

这个列表用于维护“热门 AI 技术文章阅读”的候选来源。收录不等于默认推荐，最终仍需要人工筛选。

## 自动 RSS 来源

这些来源会被 GitHub Actions 定期抓取，生成候选 issue：

| 来源 | RSS | 主要主题 |
| --- | --- | --- |
| OpenAI News | <https://openai.com/news/rss.xml> | 模型、Agent、产品工程、评估、安全 |
| Google DeepMind Blog | <https://deepmind.google/blog/rss.xml> | 研究、安全、多模态、Agent、科学智能 |
| LangChain Blog | <https://www.langchain.com/blog/rss.xml> | Agent 工程、LangGraph、评估、观测、生产化 |

## 人工精选来源

这些来源很重要，但 RSS 不稳定或需要人工判断：

| 来源 | 地址 | 主要主题 |
| --- | --- | --- |
| Anthropic Engineering | <https://www.anthropic.com/engineering> | Agent、Claude Code、MCP、工具设计、上下文工程 |
| Anthropic Research | <https://www.anthropic.com/research> | Agent、安全、评估、可解释性 |
| OpenAI Cookbook | <https://developers.openai.com/cookbook> | 可运行示例、Agent loop、评估、RAG |
| OpenAI Codex Docs | <https://developers.openai.com/codex> | 编程智能体、CLI、IDE、MCP、Skills |
| arXiv cs.AI / cs.CL / cs.LG | <https://arxiv.org/list/cs.AI/recent> | 论文和新方法 |

## 收录原则

- 优先一手资料。
- 优先工程细节和失败经验。
- 产品发布只有在包含可迁移方法论时才收录。
- 二次解读必须能指向原文、论文或代码。
