# 模型推理与基础设施阅读索引

这个主题关注模型调用、推理优化、部署、成本、延迟、缓存、限流和可观测性。

## 当前关注方向

- 推理性能：KV cache、batching、speculative decoding、量化。
- 成本控制：模型路由、缓存、prompt 压缩、分层模型。
- 服务可靠性：限流、重试、熔断、降级、监控。
- Agent 基础设施：sandbox、session、event stream、artifact、审计日志。

## 阅读前置

- [API 与模型选型](../../llm/api-and-models.md)
- [Agent 生产化](../../llm/agent-production.md)
- [MCP：Agent 的工具与上下文协议](../../llm/mcp.md)

## 待补充

后续月度精选中，优先补充：

- 推理服务架构。
- Agent sandbox。
- 模型路由和成本优化。
- 观测、追踪和 eval pipeline。
