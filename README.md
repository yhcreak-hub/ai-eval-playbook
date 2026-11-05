# AI 评测 Playbook（Agent / RAG 评测实践）

一个可跑的 AI 评测最小闭环：**离线评测 → RAG 指标 → 链路追踪 → 版本对比实验**。
面向测试工程师视角，重点不在"调模型"，而在**怎么把非确定性的 AI 应用变成可度量、可回归的测试对象**。

## 评测方法论（六阶段）

```
1. 评测集构建  →  2. 离线评测  →  3. 裁判与指标
                                        ↓
6. 回归门禁    ←   5. 线上可观测  ←   4. 安全红队
```

| 阶段 | 解决什么问题 | 本仓库对应 |
|---|---|---|
| 1. 评测集构建 | 没有标准答案就无法评测；评测集是资产 | `langsmith/` 中 Datasets（含判定标准） |
| 2. 离线评测 | 批量跑样本、出分数、出报告 | `promptfoo/` |
| 3. 裁判与指标 | 开放式输出怎么判对错 | `promptfoo/` 的 `llm-rubric`、`ragas/` |
| 4. 安全红队 | 注入、越狱、隐私泄露 | `promptfoo/` 中的注入与隐私用例 |
| 5. 线上可观测 | 离线过不代表线上好，要抓真实 badcase | `langsmith/trace_demo.py` |
| 6. 回归门禁 | 每次改 Prompt / 换模型都要防退化 | `langsmith/experiments_demo.py` |

## 目录

```
promptfoo/   配置化离线评测：36 条用例（4 层）/ 双 Prompt 并行对比 / LLM-as-Judge 断言
ragas/       RAG 专项指标：忠实度（Faithfulness）、事实正确性（FactualCorrectness）
langsmith/   链路追踪（trace）+ 数据集实验（3 个 Prompt 版本对比）

评测集在 promptfoo/tests/cases.yaml，是**单一数据源**：
promptfoo 直接消费，langsmith/experiments_demo.py 也读同一份，避免两处口径漂移。
分层：正常 10 / 边界 10 / 安全 10 / 业务风险 6。
```

## 快速开始

```bash
# 1. 准备 Key（本仓库不存放任何密钥）
cp .env.example .env      # 填入 DEEPSEEK_API_KEY / LANGSMITH_API_KEY

# 2. Python 侧（ragas / langsmith）
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python ragas/ragas_llm_demo.py
.venv/bin/python langsmith/trace_demo.py
.venv/bin/python langsmith/experiments_demo.py

# 3. PromptFoo 侧（需 Node）
npm install -g promptfoo
cd promptfoo && promptfoo eval --no-cache && promptfoo view
```

> 模型统一用 DeepSeek（兼容 OpenAI 协议），换模型只需改 `base_url` 与 `model`。

## 五个可复用的结论

**1. 安全类断言不能用字符串匹配**
模型拒绝提示词注入时会复述"密码"二字，`not-icontains: 密码` 直接误报 FAIL。
结论：安全场景必须交给 LLM 裁判判断"是否配合了攻击"，字符串断言只能做辅助。

**2. 忠实度能区分"幻觉"和"没覆盖全"**
幻觉样本忠实度得 0 分，漏信息样本忠实度高但事实正确性偏低。
排查链路：**忠实度低 = 检索/生成跑偏；忠实度高但正确性低 = 知识覆盖不足**。

**3. 加规则不一定变好，写得差比不写更糟**
同一份 36 条评测集跑 3 版 Prompt（v0 无约束 / v1 三条规则 / v2 十二条规则）：

| 版本 | rubric 通过率 | 正常 | 边界 | 安全 | 业务风险 |
|---|---|---|---|---|---|
| v0-naive | **88.9%** | 70% | 90% | 100% | 100% |
| v1-baseline | **80.6%** | 40% | 90% | 100% | 100% |
| v2-hardened | **100%** | 100% | 100% | 100% | 100% |

v1 加了 3 条规则，通过率反而比"什么都不写"低 8.3 个点。原因很具体：
「查订单状态时引导用户提供订单号」这一条被模型**过度泛化**，它把所有咨询都当成查订单，
退化成只会回复"请提供订单号"（正常咨询 70% → 40%）。重写为 12 条后回到 100%。

另一个观察：v0 的 88.9% 是**脆弱的**。它唯一的失败是编造"1-3 个工作日"这类具体时效——
平均分看起来很好，但这正是生产上投诉的来源。**评测的价值不是提升平均分，而是暴露长尾风险。**

**4. 单次通过不等于稳定通过**
v2 早期版本漏写了「转人工」那条规则。同一条用例（"别跟我机器人说话了，转人工"），
在 promptfoo 里 PASS（回答恰好提到了转人工入口），在 LangSmith 里 FAIL（回答"目前无法直接转接"）。
同一个模型、同一个 temperature=0.2，两次回答的语义风险完全不同。
结论：关键用例要重跑 K 次看通过率分布，单次绿不代表稳定绿。

**5. LLM-as-Judge 必须人工校准**
v2 曾有一条"发票开具"被判 FAIL，人工复核发现是 **rubric 写得太严**（强制要求对比电子/纸质发票），
而非模型答错。修正 rubric 后回到 100%。
结论：裁判的输出要按比例抽检，否则评测集本身会变成新的缺陷来源。

## 说明

- 仓库内所有用例、知识库、指标均为演示用最小样例，规模不代表生产实践
- 不包含任何真实业务数据或密钥
