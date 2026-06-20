# MCP：Agent 的工具与上下文协议

MCP（Model Context Protocol）是把模型连接到外部工具和上下文的协议。对 Agent 工程来说，MCP 的价值是把“工具接入”从一次性的手写适配，变成可复用、可配置、可审计的能力。

## MCP 解决什么问题

没有 MCP 时，每个 Agent 系统都要自己写：

- 如何连接 Jira、GitHub、Slack、Google Drive、数据库。
- 如何描述工具 schema。
- 如何传认证信息。
- 如何把工具结果返回给模型。
- 如何管理权限和日志。

MCP 的目标是标准化这些连接方式，让 Agent 可以接入不同工具和数据源。

## 基本模型

```text
Agent Client
  -> MCP Server
     -> tools / resources / prompts
        -> external system
```

例子：

- Codex 通过 MCP 访问开发文档、浏览器或 Figma。
- Claude Code 通过 MCP 连接 Jira、Slack、Google Drive、数据库或自定义工具。
- 企业内部 Agent 通过 MCP 暴露只读知识库、工单系统或监控系统。

## MCP 提供什么

| 能力 | 说明 |
| --- | --- |
| Tools | Agent 可调用的函数，例如查 issue、读文件、跑查询 |
| Resources | Agent 可读取的上下文，例如文档、日志、配置 |
| Prompts | 可复用的任务模板 |
| Instructions | 服务器级别的使用约束和说明 |
| Transport | 本地 stdio 或远程 HTTP 等连接方式 |
| Auth | Bearer token、OAuth 等认证机制 |

不同客户端支持的细节会不同，实际使用前要看客户端文档。

## 在 Codex 中的 MCP

OpenAI 官方文档说明，MCP 可以让 Codex 访问第三方工具和上下文；Codex 在 CLI 和 IDE extension 中支持 MCP server。Codex 的 MCP 配置存放在 `config.toml` 中，可以是用户级 `~/.codex/config.toml`，也可以在受信任项目里使用 `.codex/config.toml`。

典型用途：

- 查第三方框架文档。
- 操作浏览器或 Figma。
- 访问团队内部工具。
- 给 Codex 添加跨工具 workflow 的 server instructions。

## 在 Claude Code 中的 MCP

Claude Code 官方文档说明，MCP 可以把 Claude Code 连接到外部工具和数据源，例如 issue tracker、监控 dashboard、数据库、Google Drive 或 Slack。适合替代反复复制粘贴外部系统信息的工作流。

典型用途：

- 从 Jira/GitHub issue 读取需求。
- 从 Google Drive/Notion 读取设计文档。
- 查询日志和监控。
- 把结果写回工单或聊天工具。

## MCP Server 设计原则

### 1. 工具要小而清晰

差：

```text
manage_jira()
```

好：

```text
get_issue(issue_id)
search_issues(query)
add_comment(issue_id, body)
create_ticket(title, body, priority)
```

### 2. 默认只读

先暴露只读工具，确认 Agent 行为稳定后再开放写工具。

写工具要有：

- 权限检查。
- 人工确认。
- 幂等键。
- 审计日志。
- 回滚方案。

### 3. 返回结构化结果

工具结果不要只返回一大段自然语言。

```json
{
  "ok": true,
  "issue_id": "PROJ-123",
  "title": "Login error",
  "status": "Open",
  "url": "https://..."
}
```

### 4. 写好 server instructions

Server instructions 应说明：

- 什么时候使用这个 server。
- 工具调用限制。
- 速率限制。
- 敏感字段处理。
- 写操作审批规则。

### 5. 不泄露密钥

认证信息应该由 MCP server 或客户端配置管理，不要让模型看到 token、cookie 或数据库密码。

## 什么时候不用 MCP

| 场景 | 更简单方案 |
| --- | --- |
| 一次性脚本 | 直接写 Python/CLI |
| 单个内部函数 | 普通 tool calling |
| 静态文档问答 | RAG |
| 高风险生产写操作 | 人工流程或受控后端 API |

MCP 适合可复用、跨工具、多人共享的集成；不是所有函数都需要包装成 MCP。

## 常见错误

### 工具权限太大

修复：按读/写/危险操作分层。

### 把业务规则交给模型判断

修复：业务规则放在 MCP server 里做确定性校验。

### 返回上下文过多

修复：返回摘要、ID、URL、关键字段，让 Agent 必要时再取详情。

### 没有审计日志

修复：记录每次工具调用、参数、调用者、结果和时间。

## 学习路径

1. 先学 [Agent 工具调用](agent-tools.md)。
2. 再理解 [Harness Engineering](harness-engineering.md)。
3. 最后把稳定工具抽象成 MCP server。

## 参考资料

- OpenAI Codex MCP: https://developers.openai.com/codex/mcp
- Claude Code MCP: https://docs.anthropic.com/en/docs/claude-code/mcp
- Model Context Protocol: https://modelcontextprotocol.io/
