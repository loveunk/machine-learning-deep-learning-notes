# Coding Agents 阅读索引

这个主题关注 Codex、Claude Code、Cursor、Devin、Open SWE 等编程智能体，以及它们背后的代码库协作模式。

## 必读

- [Harness design for long-running application development](../2026/06.md#harness-design-for-long-running-application-development)：长周期应用开发 Agent 的 harness、handoff 和 evaluator 设计。
- [Scaling Managed Agents: Decoupling the brain from the hands](../2026/06.md#scaling-managed-agents-decoupling-the-brain-from-the-hands)：长任务 Agent 运行时抽象。
- [Securing the future of AI agents](../2026/06.md#securing-the-future-of-ai-agents)：高权限 coding agent 的安全监控和控制。

## 阅读前置

- [AI 编程智能体](../../llm/coding-agents.md)
- [Harness Engineering](../../llm/harness-engineering.md)
- [Loop Engineering](../../llm/loop-engineering.md)

## 读文章时重点看什么

- Agent 如何理解代码库规则。
- 如何拆解任务、写补丁、运行测试、回滚失败尝试。
- 是否有 repo instructions，例如 `AGENTS.md`、`CLAUDE.md` 或类似机制。
- 如何控制文件系统、shell、网络和凭证权限。
- 如何处理长上下文、上下文压缩和跨会话 handoff。
