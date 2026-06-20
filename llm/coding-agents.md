# AI 编程智能体：Codex、Claude Code 与协作模式

AI 编程智能体不是单纯的代码补全工具。它们能读取代码库、编辑多个文件、运行命令、检查 diff、生成提交说明，甚至把任务交给云端或子 Agent 执行。学习 Agent 工程时，Codex、Claude Code、GitHub Copilot Coding Agent、Cursor 等工具是最容易落地的观察对象。

本章重点不是“哪个工具最强”，而是理解这些工具背后的工程范式：上下文、工具、权限、反馈环、评估和人类审查。

## 编程智能体解决什么问题

| 场景 | 传统方式 | 编程智能体方式 |
| --- | --- | --- |
| 熟悉陌生代码库 | 人读目录、搜文件、跑测试 | Agent 先探索，再给结构化说明 |
| 修 bug | 人定位、改代码、跑测试 | Agent 复现、定位、修改、验证 |
| 补测试 | 人找边界条件 | Agent 读实现和历史测试，补覆盖 |
| 重构 | 人维护大量上下文 | Agent 分步骤修改并保持 diff 可审 |
| 文档同步 | 人手动更新 | Agent 根据代码变更补 README/API 文档 |
| PR review | 人逐文件看 diff | Agent 做第一轮风险扫描和建议 |

编程智能体最适合处理“上下文大、步骤多、需要验证”的软件工程任务。

## 代表工具

### Codex

Codex 是 OpenAI 的编程智能体。官方文档把 Codex CLI 定义为可以在终端本地运行、读取/修改/运行当前目录代码的 coding agent；Codex IDE extension 则在 IDE 侧边栏工作，也可以把任务委托给 Codex Cloud。

Codex 的重要能力：

- CLI、IDE extension、Codex app、Web/Cloud 等多种表面。
- 能读写本地代码、运行命令、做 code review。
- 支持 MCP，让 Codex 接入第三方文档和开发工具。
- 支持 Skills，把可复用工作流打包成说明、脚本和资源。
- 支持 Subagents，把高度并行的探索或 review 任务拆给多个专门 Agent。
- 通过 AGENTS.md、规则、权限、沙箱和配置文件约束行为。

适合：

- 终端或 IDE 中的端到端代码任务。
- 代码库探索、bug fix、重构、测试、PR review。
- 需要本地命令、浏览器、MCP、skills 或子 Agent 协作的任务。

### Claude Code

Claude Code 是 Anthropic 的 agentic coding tool。官方文档描述它可以读取代码库、编辑文件、运行命令，并集成到终端、IDE、桌面、浏览器等环境中。

Claude Code 的重要能力：

- CLI、VS Code、JetBrains、Desktop、Web 等多种入口。
- 能直接处理代码库任务，如写测试、修 lint、解决 merge conflict、更新依赖、写 release notes。
- 支持 CLAUDE.md、auto memory、skills、hooks、subagents、MCP。
- 支持 hooks，在生命周期关键点执行确定性 shell 命令，例如编辑后格式化、执行前阻断危险命令、任务完成后通知。
- 支持多 Agent、background agents、Agent SDK 和 scheduled tasks。

适合：

- 终端驱动的代码任务。
- 需要 hooks、记忆、定时任务、CI/PR 自动化的团队工作流。
- 希望把 Claude Code 能力嵌进自定义 Agent SDK 工作流的场景。

### GitHub Copilot Coding Agent

适合从 GitHub Issue、PR 或 IDE 里触发代码修改。它的强项是贴近 GitHub 工作流：issue 到 branch、PR、review、CI。

### Cursor / Windsurf / IDE Agent

适合强交互开发：你在编辑器里保持上下文，Agent 辅助理解、改代码、跑局部任务。

## 怎么选工具

| 需求 | 优先考虑 |
| --- | --- |
| 终端里完成本地代码任务 | Codex CLI、Claude Code CLI |
| IDE 内边写边改 | Codex IDE、Claude Code IDE、Cursor |
| 多任务并行和云端执行 | Codex Cloud、Claude Code Web/Desktop/background agents |
| GitHub issue 到 PR | GitHub Copilot Coding Agent、Codex GitHub 集成、Claude Code CI/CD |
| 强规则和自动化钩子 | Claude Code hooks、Codex hooks/permissions |
| 可复用工作流 | Codex Skills、Claude Skills、插件 |
| 外部工具和私有系统 | MCP |

## 编程 Agent 的使用模式

### 1. Exploration mode

让 Agent 先读代码库，不改文件。

```text
请先探索这个仓库的结构，说明主要模块、测试命令、构建命令，以及修改用户认证逻辑可能涉及哪些文件。不要修改代码。
```

价值：

- 快速建立代码地图。
- 发现测试入口和风险区域。
- 降低直接修改造成的偏差。

### 2. Plan-first mode

先要求计划，再批准执行。

```text
目标：修复用户登录失败时错误提示不准确的问题。
请先给出修改计划、涉及文件、测试方案和风险点，等我确认后再改代码。
```

适合：

- 影响面不确定的改动。
- 需要和人类对齐设计的任务。

### 3. Patch-and-test mode

让 Agent 修改后必须运行验证。

```text
请实现这个修复，并运行相关单元测试。如果测试失败，先解释失败原因，再决定是否继续修改。
```

关键是把“完成”的定义写清楚：代码改了不算完，验证通过才算。

### 4. Review mode

让 Agent 扮演 reviewer，而不是 author。

```text
请 review 当前分支相对 master 的 diff。重点找 bug、回归风险、安全问题和缺失测试。不要重写代码。
```

适合：

- PR 前自检。
- 人类 review 前减轻低级错误。

### 5. Multi-agent mode

把可以并行的问题拆开。

```text
请为当前 PR 启动多个独立检查：安全、测试覆盖、性能、可维护性。每个检查独立分析，最后汇总结论。
```

注意：多 Agent 会增加 token 和工具成本，适合复杂任务，不适合简单修改。

## 编程 Agent 的关键输入

| 输入 | 作用 |
| --- | --- |
| 目标 | 做什么，不做什么 |
| 验收标准 | 什么结果算完成 |
| 代码范围 | 哪些目录/文件可以改 |
| 测试命令 | 怎么验证 |
| 风格约束 | 命名、架构、依赖、错误处理 |
| 安全边界 | 不能执行哪些命令，不能访问哪些数据 |
| 提交规范 | commit message、PR 模板、review 要点 |

这些输入最好沉淀到 `AGENTS.md`、`CLAUDE.md`、规则文件、skills 或 hooks 中，而不是每次临时口头补充。

## 常见错误

### 让 Agent 直接“大改一下”

修复：先要求探索和计划，限制文件范围，给出测试命令。

### 没有验证闭环

修复：每个任务都要绑定 lint、test、typecheck、build 或人工验收。

### 规则只写在聊天里

修复：把稳定规则写入仓库文件，例如 `AGENTS.md`、`CLAUDE.md`、项目 README、CI 脚本。

### 忽略权限

修复：高风险命令、生产数据、密钥、部署动作必须人工确认。

### 把 Agent 输出当作最终答案

修复：保留 diff review、测试日志、trace 和 reviewer 责任。

## 和 Agent 工程的关系

编程 Agent 是最典型的 Agent 工程实验场：

- 文件系统、shell、git、浏览器、issue tracker 都是工具。
- `AGENTS.md` / `CLAUDE.md` 是长期上下文。
- hooks / CI / tests 是反馈环。
- subagents 是并行任务分解。
- PR review 是人类监督。

因此，学会使用编程 Agent，本质上就是在学习 Agent harness 和 loop 的设计。

## 下一步

- 学 [Harness Engineering](harness-engineering.md)：设计 Agent 外部系统。
- 学 [Loop Engineering](loop-engineering.md)：设计迭代、验证和停止条件。
- 学 [MCP](mcp.md)：把 Agent 接入工具和数据源。
- 学 [Agent 记忆与上下文](agent-memory-context.md)：把项目规则持久化。

## 参考资料

- OpenAI Codex CLI: https://developers.openai.com/codex/cli
- OpenAI Codex IDE extension: https://developers.openai.com/codex/ide
- OpenAI Codex MCP: https://developers.openai.com/codex/mcp
- OpenAI Codex Skills: https://developers.openai.com/codex/skills
- OpenAI Codex Subagents: https://developers.openai.com/codex/subagents
- Claude Code Overview: https://docs.anthropic.com/en/docs/claude-code/overview
- Claude Code Hooks: https://docs.anthropic.com/en/docs/claude-code/hooks-guide
- Claude Code MCP: https://docs.anthropic.com/en/docs/claude-code/mcp
- Claude Code Subagents: https://docs.anthropic.com/en/docs/claude-code/sub-agents
