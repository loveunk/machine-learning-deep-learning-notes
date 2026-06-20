# LLM 应用工程阅读索引

这个主题关注把大模型接入真实系统时需要的工程结构：上下文、工具、harness、loop、评估、观测和容错。

## 必读

- [How to Build a Custom Agent Harness](../2026/06.md#how-to-build-a-custom-agent-harness)：理解 model + harness 的基本结构。
- [The Art of Loop Engineering](../2026/06.md#the-art-of-loop-engineering)：理解反馈循环如何决定 Agent 可靠性。
- [Fault Tolerance in LangGraph](../2026/06.md#fault-tolerance-in-langgraph-retries-timeouts-and-error-handlers)：重试、超时和错误处理。

## 阅读前置

- [Prompt 与上下文工程](../../llm/prompting.md)
- [Harness Engineering](../../llm/harness-engineering.md)
- [Loop Engineering](../../llm/loop-engineering.md)
- [Agent 生产化](../../llm/agent-production.md)

## 读文章时重点看什么

- 文章是否给出了可迁移的系统结构，而不是只展示某个框架 API。
- 是否解释了失败模式。
- 是否包含评估、观测、日志或追踪。
- 是否讨论成本、延迟、权限和安全边界。
