# 热门 AI 技术文章阅读

这个板块不是 AI 新闻列表，而是把近期值得读的技术文章接到本仓库的学习路径上。

筛选文章时优先回答三个问题：

1. 这篇文章补充了仓库里的哪个主题。
2. 读者读完后能带走什么工程判断。
3. 这篇文章是否仍然值得在几个月后回看。

## 推荐阅读方式

如果你正在系统学习，建议先读对应的仓库章节，再读外部文章：

| 学习方向 | 仓库章节 | 阅读索引 |
| --- | --- | --- |
| Agent 基础 | [AI Agent](../llm/agent.md) | [Agent](topics/agent.md) |
| 编程智能体 | [AI 编程智能体](../llm/coding-agents.md) | [Coding Agents](topics/coding-agents.md) |
| Harness / Loop | [Harness Engineering](../llm/harness-engineering.md)、[Loop Engineering](../llm/loop-engineering.md) | [LLM 应用工程](topics/llm-engineering.md) |
| RAG | [RAG](../llm/rag.md)、[RAG 评估](../llm/rag-evaluation.md) | [RAG](topics/rag.md) |
| 多模态 | [多模态总览](../multimodal/README.md) | [多模态](topics/multimodal.md) |
| 推理与部署 | [API 与模型选型](../llm/api-and-models.md) | [模型推理与基础设施](topics/infra.md) |

## 最新精选

- [2026 年 6 月精选](2026/06.md)

## 选文标准

优先收录：

- 官方技术博客、论文、工程复盘、开源项目设计文档。
- 能对应到本仓库学习路径的文章，例如 Agent、RAG、评估、MCP、推理部署、多模态。
- 有清晰工程细节的文章，例如架构、失败模式、评估方法、上线经验。
- 观点可验证、可复盘、不是纯营销发布的内容。

暂不收录：

- 单纯模型发布新闻。
- 只比较排行榜、没有方法论的文章。
- 没有一手信息来源的二次转述。
- 无法长期帮助学习者建立判断的热点争议。

## 条目格式

每条文章控制在 150-250 字中文摘要，格式见 [template.md](template.md)。

必须包含：

- 链接、来源、日期。
- 主题和难度。
- 推荐阅读前置章节。
- 推荐理由。
- 摘要。
- 读完应该带走什么。

## 更新节奏

建议采用两段式流程：

1. 每天由 GitHub Actions 抓取可信 RSS，生成候选 issue。
2. 人工筛选后，把 3-5 篇真正值得读的文章写入月度精选。

不要让自动化直接提交文章摘要到 `master`。自动化适合做候选收集，最终摘要和推荐理由仍然需要人工判断。

## GitHub Actions 还是 Codex Automations

当前建议：

- GitHub Actions：负责公开、可审计、可复现的候选链接收集。
- Codex Automations：适合做个人提醒、辅助阅读、草拟摘要、定期提示维护者处理候选 issue。

更具体地说：

| 方案 | 适合做 | 不适合做 |
| --- | --- | --- |
| GitHub Actions | 定时抓 RSS、生成候选 issue、跑链接检查、保证流程在仓库内可见 | 自动判断文章质量、直接写入主分支 |
| Codex Automations | 每天提醒维护者筛选、帮忙总结候选文章、把 issue 草拟成 PR 内容 | 作为唯一公开记录、替代仓库内 CI 流程 |

因此本仓库先采用 GitHub Actions 收集候选，再由人或 Codex 辅助筛选后提交。

## 可信来源

来源列表见 [sources.md](sources.md)。自动收集的 RSS 源配置见 [sources.json](sources.json)。

自动候选收集现在覆盖官方实验室、Agent/工程博客、论文源、深度个人博客和中文来源。为了避免单一来源刷屏，GitHub Actions 会限制每个来源每天最多进入 2 篇候选，并要求候选达到最低相关性分数；中文文章也会按“大模型、智能体、多模态、工程化、评测、安全”等关键词参与初筛。
