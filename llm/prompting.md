# Prompt 与上下文工程

Prompt Engineering 的核心不是写“神奇咒语”，而是把任务信息组织成模型容易遵守的输入。更准确地说，现代 LLM 应用做的是 Context Engineering：选择什么上下文、以什么结构放进去、如何约束输出、如何验证结果。

## 一个好 Prompt 的结构

最小结构：

```text
角色：你是谁。
任务：你要完成什么。
上下文：你能依据哪些信息。
约束：不能做什么，必须遵守什么。
输出：用什么格式返回。
验收标准：什么结果算正确。
```

示例：

```text
你是机器学习课程助教。

任务：用初学者能理解的话解释“过拟合”。

上下文：读者已经知道训练集和测试集，但不了解正则化。

约束：
- 不使用复杂公式。
- 不超过 120 字。
- 必须给一个生活类比。

输出格式：
1. 一句话定义
2. 一个类比
3. 一个判断方法
```

## Prompt 的五个层次

### 1. 指令

说明模型要完成什么任务。

差：

```text
总结一下。
```

好：

```text
请把下面会议记录总结成“决策、待办、风险”三类，每类最多 5 条。
```

### 2. 上下文

告诉模型可以依据哪些信息，避免它自由发挥。

```text
仅依据 <context> 中的信息回答。如果 context 中没有答案，返回“资料不足”。
```

### 3. 示例

few-shot 示例能稳定格式和风格。

```text
输入：用户要求退款，但订单已超过 30 天。
输出：
{
  "intent": "refund_request",
  "risk": "policy_violation",
  "next_action": "explain_policy"
}
```

### 4. 输出格式

能用结构化输出就不要只靠自然语言约定。

```json
{
  "answer": "string",
  "citations": ["string"],
  "confidence": "low | medium | high"
}
```

### 5. 校验

把“再检查一遍”变成明确规则。

```text
返回前检查：
- 每个结论是否能在引用中找到依据。
- 是否出现了上下文之外的人名、数字或日期。
- JSON 是否能被解析。
```

## Prompt 设计流程

1. 写出输入和输出样例。
2. 先用最短 Prompt 跑 10 个样例。
3. 记录失败类型，而不是直接加长 Prompt。
4. 针对失败补约束或示例。
5. 对高风险字段加结构化输出和程序校验。
6. 固化评估集，比较改动前后结果。

## 常见失败和修复

| 失败 | 原因 | 修复 |
| --- | --- | --- |
| 答案太发散 | 任务边界不清 | 加目标、受众、长度、输出格式 |
| 编造事实 | 上下文不足或未限制来源 | 要求仅依据 context，缺失时返回资料不足 |
| 格式不稳定 | 只用自然语言描述格式 | 使用 JSON schema 或强校验 |
| 过度解释 | 没有限制粒度 | 明确字数、条数、面向对象 |
| 忽略约束 | 约束太多且冲突 | 减少约束，按优先级排序 |
| 多轮对话跑偏 | 历史消息太长 | 摘要历史、重建状态、压缩上下文 |

## RAG 场景 Prompt 模板

```text
你是严谨的知识库问答助手。

规则：
1. 仅依据 <context> 回答。
2. 如果资料不足，回答“资料不足”，并说明缺少什么信息。
3. 每个关键结论都要给出引用编号。
4. 不要引用与结论无关的片段。

<context>
[1] ...
[2] ...
</context>

问题：...

输出：
- 答案：
- 引用：
- 资料缺口：
```

## 信息抽取模板

```text
从文本中抽取字段，返回严格 JSON。

字段：
- company: 公司名，找不到则为 null
- amount: 金额数字，保留原币种
- date: 日期，使用 YYYY-MM-DD；无法确定则为 null
- risk_flags: 风险标签数组

文本：
...
```

## 代码生成模板

```text
你是资深 Python 工程师。

任务：实现一个函数。

要求：
- 只使用标准库。
- 给出类型标注。
- 包含 3 个 pytest 测试。
- 说明时间复杂度。

输入输出：
...
```

## 不要做的事

- 不要把 Prompt 当作唯一安全措施。
- 不要在 Prompt 中放密钥、隐私数据或无关历史。
- 不要用“请一定不要出错”代替程序校验。
- 不要为了少数失败不断堆规则，先看是否需要检索、工具或评估。

## 下一步

- 如果问题是缺知识，学习 [RAG](rag.md)。
- 如果问题是格式不稳定，学习 [Agent 工具调用](agent-tools.md)。
- 如果问题是多步骤执行，学习 [Agent 工作流](agent-workflows.md)。

## 参考资料

- OpenAI Prompting: https://developers.openai.com/api/docs/guides/prompting
- OpenAI Structured Outputs: https://developers.openai.com/api/docs/guides/structured-outputs
- Anthropic Prompt Engineering: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview
