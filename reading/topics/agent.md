# Agent 阅读索引

这个主题关注能调用工具、处理多步骤任务、根据反馈推进的智能体系统。

## 必读

- [Building effective agents](../2026/06.md#building-effective-agents)：Agent 与 workflow 的基本边界。
- [A practical guide to building agents](../2026/06.md#a-practical-guide-to-building-agents)：Agent 设计、工具、编排和 guardrails。
- [How we built our multi-agent research system](../2026/06.md#how-we-built-our-multi-agent-research-system)：多 Agent 研究系统的一手复盘。

## 阅读前置

- [AI Agent](../../llm/agent.md)
- [Agent 工具调用](../../llm/agent-tools.md)
- [Agent 工作流模式](../../llm/agent-workflows.md)

## 读文章时重点看什么

- 任务是不是必须用 Agent，而不是普通工作流。
- 工具接口是否清晰、可测试、可恢复。
- 是否有明确停止条件。
- 是否能用评估集证明改动有效。
- 是否有权限、沙箱、审计和人工接管机制。
