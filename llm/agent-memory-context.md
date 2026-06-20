# Agent 记忆与上下文管理

Agent 的质量很大程度取决于上下文。模型本身不会自动知道你的项目规范、历史决策、测试命令、业务边界和个人偏好；这些信息需要被有意识地放进上下文或持久化成记忆。

## 上下文和记忆的区别

| 概念 | 说明 | 例子 |
| --- | --- | --- |
| Context | 本轮任务可见的信息 | 当前 prompt、检索片段、打开的文件、工具结果 |
| Memory | 跨任务复用的信息 | 项目规则、常用命令、架构约束、用户偏好 |
| State | 当前任务进度 | todo、已执行步骤、错误、审批状态 |

不要把三者混在一起。Context 解决“当前要知道什么”，Memory 解决“长期要记住什么”，State 解决“现在做到哪里”。

## Agent 为什么会忘

常见原因：

- 上下文窗口有限。
- 多轮对话不断压缩。
- 文件没有被读取。
- 规则只写在历史聊天里。
- 检索没有召回相关文档。
- 子 Agent 没继承完整上下文。

因此，重要规则要放在稳定位置，而不是期待模型凭印象记住。

## 稳定记忆的载体

| 工具 | 载体 |
| --- | --- |
| Codex | `AGENTS.md`、rules、skills、`.codex/config.toml`、memories |
| Claude Code | `CLAUDE.md`、auto memory、skills、hooks、settings |
| 通用 Agent | system prompt、配置文件、RAG 知识库、数据库 |

仓库级规则应该进仓库；个人偏好应该进用户级记忆；任务进度应该进 state。

## 应该写进项目记忆的内容

- 项目目标和核心模块。
- 构建、测试、lint、格式化命令。
- 代码风格和架构边界。
- 哪些目录不能改。
- 数据库迁移、API 兼容、安全要求。
- 常见失败和排错方法。
- PR 和 commit 规范。
- 评估和验收标准。

示例：

```md
# Agent Instructions

- 修改后运行 `npm test` 和 `npm run lint`。
- 不要直接修改 `generated/` 目录。
- API response 必须保持向后兼容。
- 新增业务逻辑必须补单元测试。
- 涉及认证、权限、支付时先给计划，等确认后再改。
```

## 不应该写进记忆的内容

- API key、token、密码。
- 一次性临时信息。
- 未确认的猜测。
- 用户隐私和敏感数据。
- 会频繁变化的线上状态。

记忆不是垃圾桶。错误或过期记忆会持续污染 Agent 行为。

## 上下文压缩

长任务中需要压缩上下文。压缩时保留：

- 用户目标。
- 当前计划。
- 已完成步骤。
- 关键文件和 diff。
- 测试结果。
- 阻塞点。
- 决策和理由。

丢弃：

- 重复日志。
- 无关探索。
- 已被否定的方案细节。
- 大段原始输出，除非后续还要分析。

压缩模板：

```text
目标：
当前状态：
已修改：
验证结果：
未解决问题：
下一步：
禁止事项：
```

## 子 Agent 的上下文

子 Agent 不应该默认拿到全部上下文。给子 Agent 的输入应包含：

- 子任务目标。
- 允许读取/修改范围。
- 必要背景。
- 输出格式。
- 停止条件。

示例：

```text
你只负责检查当前 PR 的测试风险。
不要修改代码。
读取 diff 和 tests/ 目录，输出：
1. 可能缺失的测试
2. 现有测试是否覆盖改动
3. 建议新增的测试文件
```

这样能减少 context pollution，也能让并行 Agent 更独立。

## RAG 作为长期记忆

如果知识量很大，不能都写进 `AGENTS.md` 或 `CLAUDE.md`，应使用 RAG：

- 设计文档。
- 历史 ADR。
- API 文档。
- 会议纪要。
- 业务规则。
- 常见问题。

但 RAG 记忆要可追溯，回答要带来源。

## 记忆维护

记忆需要版本管理：

- 新规则经过 review 后写入。
- 过期规则定期删除。
- 失败案例沉淀为具体规则或测试。
- 个人偏好不要污染团队规则。

把记忆当成代码维护，而不是随手记录。

## 常见错误

### 把所有东西塞进 prompt

修复：稳定规则持久化，大量知识走 RAG，任务进度走 state。

### 记忆没有来源

修复：重要规则注明来源或决策背景。

### 子 Agent 上下文过宽

修复：给子 Agent 最小任务包。

### 过期记忆不删除

修复：定期 review `AGENTS.md`、`CLAUDE.md`、rules 和知识库。

## 下一步

- 学 [AI 编程智能体](coding-agents.md)。
- 学 [Harness Engineering](harness-engineering.md)。
- 学 [Loop Engineering](loop-engineering.md)。

## 参考资料

- OpenAI Codex AGENTS.md: https://developers.openai.com/codex/guides/agents-md
- OpenAI Codex Skills: https://developers.openai.com/codex/skills
- Claude Code Memory: https://docs.anthropic.com/en/docs/claude-code/memory
- Claude Code Overview: https://docs.anthropic.com/en/docs/claude-code/overview
