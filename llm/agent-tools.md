# Agent 工具调用与结构化输出

Agent 的第一步不是“自动规划”，而是让模型可靠地调用工具。工具调用的关键是把外部能力写成清晰、受限、可验证的接口。

## 工具调用解决什么问题

LLM 本身不适合做这些事：

- 查询实时数据。
- 执行精确计算。
- 写数据库。
- 调用业务系统。
- 搜索内部知识库。
- 发送邮件或创建工单。

工具调用把这些动作交给确定性的程序，模型只负责决定何时调用、传什么参数、如何解释结果。

## 工具的最小契约

一个工具至少要定义：

| 字段 | 说明 |
| --- | --- |
| name | 工具名，清晰表达用途 |
| description | 什么时候用，什么时候不用 |
| input schema | 参数类型、必填字段、枚举值、范围 |
| output schema | 返回字段和错误格式 |
| permissions | 是否需要人工确认或权限 |
| side effects | 是否会修改外部系统 |

## 示例：文档搜索工具

```python
def search_docs(query: str, top_k: int = 5) -> list[dict]:
    """Search internal documents and return relevant chunks."""
    ...
```

工具 schema：

```json
{
  "name": "search_docs",
  "description": "Search internal learning notes. Use this when the user asks about repository content or concepts covered by the notes.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query in Chinese or English."
      },
      "top_k": {
        "type": "integer",
        "minimum": 1,
        "maximum": 10,
        "default": 5
      }
    },
    "required": ["query"]
  }
}
```

## 工具设计原则

### 1. 工具要小

差：

```text
manage_customer()
```

好：

```text
get_customer_profile()
list_recent_orders()
create_refund_ticket()
```

工具越大，模型越难选择正确动作，权限也越难控制。

### 2. 参数要强约束

使用枚举、范围和必填字段，少让模型自由编字符串。

```json
{
  "priority": {
    "type": "string",
    "enum": ["low", "medium", "high"]
  }
}
```

### 3. 有副作用的工具要隔离

读操作和写操作分开。

- 读工具：搜索、查询、计算。
- 写工具：发邮件、下单、删除、转账、改权限。

写工具通常需要：

- 人工确认。
- 幂等键。
- 审计日志。
- 回滚方案。

### 4. 错误要结构化

不要只返回自然语言错误。

```json
{
  "ok": false,
  "error_code": "NOT_FOUND",
  "message": "No matching document found.",
  "retryable": false
}
```

## 结构化输出 vs 工具调用

| 需求 | 优先方案 |
| --- | --- |
| 把文本抽取成 JSON | 结构化输出 |
| 分类、打标签 | 结构化输出 |
| 查询外部数据 | 工具调用 |
| 修改外部系统 | 工具调用 + 人工确认 |
| 多步骤任务 | 工具调用 + workflow |

结构化输出是“模型返回可解析结果”。工具调用是“模型请求程序执行动作”。

## 工具调用循环

```text
user request
  -> model decides tool call
  -> program validates arguments
  -> program executes tool
  -> tool returns observation
  -> model uses observation
  -> final answer or next tool call
```

程序必须掌握最终执行权。不要让模型绕过参数校验。

## 安全边界

高风险工具要默认关闭或人工确认：

- 删除数据。
- 修改权限。
- 发送外部消息。
- 付款和退款。
- 执行 shell。
- 访问敏感个人数据。

权限策略示例：

| 工具 | 权限 |
| --- | --- |
| search_docs | 自动执行 |
| get_order_status | 自动执行，但记录日志 |
| create_refund_ticket | 人工确认 |
| issue_refund | 双人审批 |
| run_shell_command | 默认禁用 |

## 常见错误

### 工具描述太模糊

模型不知道什么时候该用工具。

修复：在 description 中写清适用场景和反例。

### 工具返回太长

返回整篇文档会污染上下文。

修复：返回摘要、top chunks、metadata、score。

### 缺少参数校验

模型可能传空字符串、超大 top_k、错误日期。

修复：schema 校验 + 程序二次校验。

### 让模型直接决定高风险动作

修复：加入人工确认和权限层，不把安全寄托在 prompt 上。

## 下一步

- 需要多步骤编排时看 [Agent 工作流](agent-workflows.md)。
- 需要上线能力时看 [Agent 生产化](agent-production.md)。

## 参考资料

- OpenAI Using tools: https://developers.openai.com/api/docs/guides/tools
- OpenAI Function calling: https://developers.openai.com/api/docs/guides/function-calling
- LangChain tools: https://docs.langchain.com/oss/python/langchain/tools
