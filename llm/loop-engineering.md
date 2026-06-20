# Loop Engineering

Loop Engineering 是设计 Agent 如何反复观察、行动、验证、修正和停止的能力。Harness 让 Agent 能做事；Loop 决定 Agent 如何持续推进任务。

一个简单循环：

```text
observe -> decide -> act -> verify -> update state -> stop or continue
```

编程 Agent、RAG Agent、数据分析 Agent、运维 Agent 都需要 loop。没有 loop 的 Agent 只是一次模型调用；没有停止条件的 loop 会浪费成本甚至造成风险。

## Loop 和 Workflow 的区别

| 概念 | 重点 |
| --- | --- |
| Workflow | 固定步骤和状态转移 |
| Loop | 基于反馈反复迭代，直到满足条件 |
| Harness | 支撑 loop 运行的工具、权限、上下文和反馈系统 |

Workflow 更像流程图，Loop 更像控制系统。

## 典型 Agent Loop

### 1. ReAct Loop

```text
Thought -> Action -> Observation -> Thought -> ...
```

适合：

- 搜索和问答。
- 少量工具调用。
- 任务边界清楚。

风险：

- 容易反复搜索。
- 缺少全局计划。
- 中间 reasoning 不一定可靠。

控制：

- 最大步数。
- 工具调用预算。
- 重复动作检测。

### 2. Plan-Act-Verify Loop

```text
plan -> implement step -> run checks -> revise plan -> continue
```

适合：

- 修 bug。
- 重构。
- 数据分析。
- 文档迁移。

关键是 `verify` 必须外部化，不能只让模型自评。

### 3. Test-Driven Agent Loop

```text
write failing test -> implement -> run test -> fix -> pass -> refactor
```

适合：

- 代码任务。
- 规则明确的功能。
- bug regression。

优点：

- 验收标准明确。
- 更容易防止无意义改动。

### 4. Research Loop

```text
question -> search -> read -> synthesize -> gap analysis -> search again
```

适合：

- 文献调研。
- 竞品分析。
- API 选型。

控制：

- 限制来源范围。
- 记录引用。
- 明确何时资料足够。

### 5. Improvement Loop

```text
collect failures -> classify -> patch prompt/tool/code -> eval -> deploy
```

适合：

- RAG 系统优化。
- Agent 线上质量提升。
- 客服机器人。

这是把 Agent 从 demo 推向生产的关键 loop。

## Loop 的组成

| 组件 | 问题 |
| --- | --- |
| Goal | 目标是否可验证 |
| State | 当前进度存在哪里 |
| Action | 下一步允许做什么 |
| Feedback | 如何知道动作是否有效 |
| Budget | 最多花多少时间、token、工具调用 |
| Stop | 何时停止 |
| Recovery | 失败后重试、回滚还是转人工 |

## Stop Condition

Loop 必须有停止条件：

- 验收测试通过。
- 找不到更多相关信息。
- 连续两次改动没有改善。
- 达到最大步数或预算。
- 需要人工确认。
- 检测到高风险动作。
- 外部系统不可用。

没有停止条件的 Agent 容易陷入“再试一下”的成本黑洞。

## Verification

好的 loop 依赖强验证。

| 任务 | 验证方式 |
| --- | --- |
| 代码修复 | 单元测试、集成测试、lint、typecheck |
| RAG 回答 | 引用支持、faithfulness、人工样例 |
| 数据分析 | SQL 可复现、口径一致、异常检查 |
| 文档整理 | 链接有效、事实引用、结构检查 |
| 外部动作 | dry-run、人工确认、审计日志 |

模型自评只能作为辅助，不应作为唯一验证。

## Loop 设计模板

```text
目标：
输入：
允许动作：
禁止动作：
状态字段：
每轮步骤：
1. 观察当前状态
2. 选择下一步动作
3. 执行动作
4. 验证结果
5. 更新状态
停止条件：
失败处理：
人工介入点：
```

## 编程 Agent Loop 示例

```text
目标：修复登录接口 bug。

每轮：
1. 读取错误和相关代码。
2. 修改一个最小 diff。
3. 运行相关测试。
4. 如果失败，分类失败原因。
5. 最多重试 3 次。

停止：
- 测试通过并 diff 可解释。
- 连续两次失败原因相同。
- 需要修改数据库或生产配置。
```

## Loop Engineering 常见错误

### 只设计 happy path

修复：显式写出失败、超时、权限拒绝、信息不足时怎么办。

### 让模型决定是否成功

修复：外部验证优先，模型自评只做辅助说明。

### 无限重试

修复：最大步数、最大成本、重复动作检测。

### 没有状态

修复：保存任务 ID、步骤、工具结果、失败原因、人工确认。

### 每轮上下文越来越长

修复：压缩历史，保留状态摘要和关键证据，丢弃无关细节。

## 和 Harness 的关系

Harness 提供工具和环境，Loop 使用这些工具持续推进任务。

```text
Harness: shell, git, tests, docs, permissions, logs
Loop: edit -> test -> inspect -> revise -> stop
```

强 loop 通常需要强 harness。没有测试和日志，loop 很难知道自己是否变好。

## 下一步

- 学 [Harness Engineering](harness-engineering.md)。
- 学 [Agent 生产化](agent-production.md)。
- 学 [RAG 评估](rag-evaluation.md)。

## 参考资料

- OpenAI Cookbook: Build iterative repair loops with Codex: https://developers.openai.com/cookbook/examples/codex/build_iterative_repair_loops_with_codex
- OpenAI Cookbook: Build an Agent Improvement Loop with Traces, Evals, and Codex: https://developers.openai.com/cookbook/examples/agents_sdk/agent_improvement_loop
- Claude Code Overview: https://docs.anthropic.com/en/docs/claude-code/overview
