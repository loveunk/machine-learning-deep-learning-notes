# Harness Engineering

Harness Engineering 是围绕模型构建执行环境、工具、约束、反馈和评估的能力。一个裸 LLM 只是模型；当它拥有工具、状态、权限、上下文、验证和人类监督后，才变成能做事的 Agent。

可以把 Agent 写成：

```text
Agent = Model + Harness
```

这里的 harness 指模型之外的一切：prompt、上下文、工具、文件系统、shell、沙箱、规则、CI、评估集、日志、权限、人类确认、任务队列、子 Agent 编排。

## 为什么 Harness Engineering 重要

模型能力越强，瓶颈越不在“会不会生成答案”，而在：

- 是否拿到正确上下文。
- 是否调用了正确工具。
- 是否按权限执行动作。
- 是否能验证结果。
- 是否能从失败中恢复。
- 是否能把人类注意力用在关键节点。

强模型 + 弱 harness 的结果通常是：demo 很惊艳，生产很脆弱。

## Harness 的组成

| 组件 | 作用 | 例子 |
| --- | --- | --- |
| Context | 给模型足够但不过量的信息 | 代码片段、需求、文档、issue、历史决策 |
| Tools | 让模型访问外部世界 | search、shell、database、browser、MCP |
| State | 保存任务进度和中间结果 | todo、trace、checkpoint、session |
| Policy | 约束能做什么 | 权限、沙箱、审批、禁止命令 |
| Feedback | 告诉模型结果如何 | 测试、lint、typecheck、用户反馈 |
| Evaluation | 衡量是否可靠 | eval dataset、judge、人工 review |
| Recovery | 失败后怎么处理 | 重试、回滚、转人工、降级 |
| Interface | 人如何监督 Agent | diff、PR、日志、dashboard |

## 编程 Agent 的 Harness

以 Codex 或 Claude Code 为例，常见 harness 包括：

- 仓库文件系统。
- Git diff 和 commit history。
- shell 命令。
- 测试和 CI。
- `AGENTS.md` / `CLAUDE.md` / rules。
- MCP 工具。
- hooks。
- skills / custom commands。
- subagents。
- 人类 review。

模型本身负责推理和生成，harness 负责让推理落到真实工程系统中。

## Harness 设计流程

### 1. 定义任务边界

先回答：

- Agent 要解决什么任务？
- 哪些任务必须人类做？
- 允许读哪些信息？
- 允许写哪些文件或系统？
- 成功标准是什么？

示例：

```text
任务：修复后端 API 的一个 bug。
允许：读取仓库、修改 src/ 和 tests/、运行 pytest。
禁止：修改数据库迁移、访问生产环境、删除文件。
完成：相关测试通过，diff 小且有说明。
```

### 2. 设计工具集合

工具越多不一定越好。优先给 Agent 最小可用工具：

- 搜索代码。
- 读取文件。
- 编辑文件。
- 运行测试。
- 查看 git diff。

高风险工具单独审批：

- 删除文件。
- 运行部署脚本。
- 修改权限。
- 访问生产数据库。

### 3. 持久化规则

把稳定规则写到项目文件：

- `AGENTS.md`：Codex 可读的仓库级规则。
- `CLAUDE.md`：Claude Code 可读的项目说明。
- `.codex/config.toml`：Codex 项目配置。
- `.claude/settings.json`：Claude Code 设置和 hooks。
- CI 配置：不可绕过的质量门。

不要每次都靠临时 prompt 记规则。

### 4. 加反馈环

反馈环应该尽量自动化：

```text
edit -> format -> lint -> test -> diff review -> human approval
```

常见反馈源：

- 单元测试。
- 类型检查。
- lint。
- snapshot 测试。
- benchmark。
- 安全扫描。
- reviewer comment。

### 5. 加观测

至少记录：

- Agent 收到的目标。
- 修改了哪些文件。
- 执行了哪些命令。
- 命令输出。
- 测试结果。
- 最终 diff。
- 人类批准/拒绝点。

没有观测，就无法改进 harness。

## Harness 成熟度

| 阶段 | 特征 |
| --- | --- |
| 手动 prompt | 人每次粘贴上下文，Agent 只回答 |
| 仓库规则 | 有 AGENTS.md/CLAUDE.md/README 指引 |
| 工具约束 | 工具有 schema、权限、白名单 |
| 自动反馈 | hooks/CI/test 自动执行 |
| 可观测 | 有 trace、日志、diff、成本和失败类型 |
| 可复用 | skills、plugins、subagents、workflow 模板 |

## 常见错误

### 把 harness 当作 prompt

Prompt 只是 harness 的一部分。真正可靠的 harness 还需要工具、权限、状态、评估和反馈。

### 工具过宽

给 Agent 一个无限制 shell，再要求它“谨慎”，不是工程设计。

### 缺少质量门

没有测试、lint、review 的 Agent 只是自动写代码，不是可靠工程师。

### 规则不可见

规则只存在于人的脑子里，Agent 就会反复犯同样的错。

### 没有失败数据

不记录失败样例，就无法知道该改模型、prompt、工具还是流程。

## Harness 设计清单

- [ ] 任务范围清楚。
- [ ] 工具最小化且有权限层。
- [ ] 稳定规则写入仓库。
- [ ] 每个写操作有验证方式。
- [ ] 高风险动作需要人工确认。
- [ ] 所有命令和 diff 可追踪。
- [ ] 失败能停止、恢复或转人工。
- [ ] 评估集能覆盖常见任务。

## 下一步

- 学 [Loop Engineering](loop-engineering.md)：让 harness 形成迭代闭环。
- 学 [Agent 工具调用](agent-tools.md)：设计工具契约。
- 学 [Agent 生产化](agent-production.md)：上线前的权限和观测。

## 参考资料

- OpenAI: Harness engineering: https://openai.com/index/harness-engineering/
- Martin Fowler: Harness engineering for coding agent users: https://martinfowler.com/articles/harness-engineering.html
- LangChain: The Anatomy of an Agent Harness: https://www.langchain.com/blog/the-anatomy-of-an-agent-harness
- Claude Code Hooks: https://docs.anthropic.com/en/docs/claude-code/hooks-guide
- OpenAI Codex Best Practices: https://developers.openai.com/codex/learn/best-practices
