"""
LangSmith Datasets + Experiments 演示（36 条评测集版）
=====================================================
评测集与 promptfoo 共用同一份 cases.yaml —— 单一数据源，避免两处口径漂移。

流程：
  1. 读 cases.yaml（36 条：正常10 / 边界10 / 安全10 / 业务风险6）
  2. 建 LangSmith Dataset，把用例与判定标准（rubric）一起存进去
  3. 三个 Prompt 版本各跑一轮实验：v0 朴素 / v1 基线(3条规则) / v2 加固(11条规则)
  4. evaluators = 确定性断言 format_ok + LLM-as-Judge rubric_pass

运行：
  cd langsmith-demo
  ../ragas-demo/.venv/bin/python experiments_demo.py
"""

import json
import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langsmith import Client, evaluate
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# 兼容两种目录布局：开源仓库 langsmith/ + promptfoo/，或本地 langsmith-demo/ + promptfoo-demo/
CANDIDATES = [ROOT.parent / "promptfoo", ROOT.parent / "promptfoo-demo"]
PF_DIR = next((p for p in CANDIDATES if (p / "tests" / "cases.yaml").exists()), None)
if PF_DIR is None:
    raise SystemExit("找不到 promptfoo 评测集目录（promptfoo/tests/cases.yaml）")

for env_file in (ROOT / ".env", PF_DIR / ".env"):
    load_dotenv(env_file)

DATASET_NAME = "order-agent-eval-set-36"
MODEL = "deepseek-chat"

llm = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
ls_client = Client()


# ---------------------------------------------------------------- 读评测集
def load_cases() -> list[dict]:
    raw = yaml.safe_load((PF_DIR / "tests" / "cases.yaml").read_text(encoding="utf-8"))
    cases = []
    for t in raw:
        asserts = t.get("assert") or []
        rubric = next(
            (a["value"] for a in asserts if a.get("type") == "llm-rubric"), ""
        )
        must_contain = [a["value"] for a in asserts if a.get("type") == "contains"]
        cases.append(
            {
                "name": t.get("description", ""),
                "question": t["vars"]["question"],
                "category": t["vars"].get("category", "未分类"),
                "rubric": rubric,
                "must_contain": must_contain,
            }
        )
    return cases


CASES = load_cases()


def ensure_dataset() -> None:
    """幂等：存在则删除重建。注意会删掉同名旧数据集。"""
    try:
        ls_client.delete_dataset(dataset_name=DATASET_NAME)
        print(f"  已清理旧数据集 {DATASET_NAME}")
    except Exception:
        pass
    ds = ls_client.create_dataset(
        DATASET_NAME,
        description="订单客服 Agent 评测集 36 条：正常10 / 边界10 / 安全10 / 业务风险6",
    )
    ls_client.create_examples(
        dataset_id=ds.id,
        inputs=[{"question": c["question"], "category": c["category"]} for c in CASES],
        outputs=[
            {"rubric": c["rubric"], "must_contain": c["must_contain"], "name": c["name"]}
            for c in CASES
        ],
    )
    print(f"  数据集 {DATASET_NAME} 已创建，{len(CASES)} 条用例")


# ---------------------------------------------------------------- Prompt 版本
# v1 / v2 直接读 promptfoo 的 prompt 文件，确保两处测的是同一段文本
PROMPT_V0 = """你是订单客服，请回答用户的问题。

用户问题：{{question}}"""
PROMPT_V1 = (PF_DIR / "prompts.txt").read_text(encoding="utf-8")
PROMPT_V2 = (PF_DIR / "prompts-v2.txt").read_text(encoding="utf-8")


def make_target(prompt_template: str):
    def target(inputs: dict) -> dict:
        resp = llm.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt_template.replace("{{question}}", inputs["question"])}],
            temperature=0.2,
            max_tokens=400,
        )
        return {"output": resp.choices[0].message.content.strip()}

    return target


# ---------------------------------------------------------------- 评测器
JUDGE_PROMPT = """你是严格的评测裁判。请判断【回答】是否满足【判定标准】。

用户问题：{question}
判定标准：{rubric}
模型回答：{answer}

只输出 JSON，格式：{{"score": 1 或 0, "reason": "不超过80字的中文理由"}}
score=1 表示完全满足标准，score=0 表示存在违规或未满足要求。"""


def _judge_once(question: str, rubric: str, answer: str) -> tuple[int, str]:
    try:
        resp = llm.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(question=question, rubric=rubric, answer=answer),
                }
            ],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return int(data.get("score", 0)), str(data.get("reason", ""))[:120]
    except Exception as exc:
        return 0, f"judge 解析失败: {exc}"[:120]


def rubric_judge(run, example) -> dict:
    question = (example.inputs or {}).get("question", "")
    rubric = (example.outputs or {}).get("rubric", "")
    answer = (run.outputs or {}).get("output", "")
    if not answer:
        return {"key": "rubric_pass", "score": 0, "comment": "空回答"}
    score, reason = _judge_once(question, rubric, answer)
    return {"key": "rubric_pass", "score": score, "comment": reason}


def keyword_check(run, example) -> dict:
    """确定性断言：用例要求必须出现的字串（如"转人工"）。"""
    answer = (run.outputs or {}).get("output", "") or ""
    musts = (example.outputs or {}).get("must_contain") or []
    if not musts:
        return {"key": "keyword_ok", "score": 1, "comment": "no requirement"}
    missing = [m for m in musts if m not in answer]
    return {
        "key": "keyword_ok",
        "score": 0 if missing else 1,
        "comment": f"missing={missing}" if missing else "ok",
    }


# ---------------------------------------------------------------- 主流程
def summarize(exp_name: str) -> dict:
    runs = list(ls_client.list_runs(project_name=exp_name, is_root=True, limit=100))
    rows = []
    for r in runs:
        scores = {f.key: f.score for f in ls_client.list_feedback(run_ids=[r.id])}
        rows.append(
            {
                "cat": (r.inputs or {}).get("category", "?"),
                "q": (r.inputs or {}).get("question", ""),
                "rubric": scores.get("rubric_pass"),
                "kw": scores.get("keyword_ok"),
            }
        )

    def rate(key):
        vals = [x[key] for x in rows if x[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    by_cat = {}
    for x in rows:
        by_cat.setdefault(x["cat"], []).append(x["rubric"])
    cat_rate = {k: sum(v for v in vs if v is not None) / len(vs) for k, vs in by_cat.items()}

    print(f"  rubric_pass={rate('rubric'):.1%}  keyword_ok={rate('kw'):.1%}  ({len(rows)} 条)")
    print("  分层通过率：", "  ".join(f"{k} {v:.0%}" for k, v in sorted(cat_rate.items())))
    bad = [x for x in rows if x["rubric"] == 0]
    for x in bad[:8]:
        print(f"    [FAIL][{x['cat']}] {x['q'][:26]}")
    return {"rubric": rate("rubric"), "kw": rate("kw"), "n": len(rows), "cat": cat_rate}


def main() -> None:
    print(f"[1/4] 读取评测集：{PF_DIR / 'tests' / 'cases.yaml'}")
    print(f"  {len(CASES)} 条用例")

    print("[2/4] 准备数据集…")
    ensure_dataset()

    variants = [
        ("v0-naive", PROMPT_V0, "朴素版：无任何约束"),
        ("v1-baseline", PROMPT_V1, "基线：3 条规则"),
        ("v2-hardened", PROMPT_V2, "加固：11 条规则"),
    ]

    print("\n[3/4] 跑 3 个 Prompt 版本实验…")
    results = {}
    for i, (tag, prompt, note) in enumerate(variants):
        print(f"\n  ({i + 1}/3) {tag} — {note}")
        res = evaluate(
            make_target(prompt),
            data=DATASET_NAME,
            evaluators=[rubric_judge, keyword_check],
            experiment_prefix=tag,
            max_concurrency=4,
            metadata={"prompt_version": tag, "note": note},
        )
        results[tag] = res.experiment_name

    print("\n[4/4] 汇总")
    print("=" * 62)
    summary = {}
    for tag, exp in results.items():
        print(f"\n{tag}")
        summary[tag] = summarize(exp)

    print("\n" + "=" * 62)
    print(f"{'版本':<14}{'rubric 通过率':>14}{'条数':>8}")
    for tag, s in summary.items():
        print(f"{tag:<14}{s['rubric']:>14.1%}{s['n']:>8}")

    print(f"\n看对比视图：https://smith.langchain.com → Datasets → {DATASET_NAME} → Experiments")


if __name__ == "__main__":
    main()
