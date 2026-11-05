# -*- coding: utf-8 -*-
"""
LangSmith 追踪 demo：一条最小的 RAG 链路（检索 → 生成）

演示"线上可观测"长什么样——每个环节的输入输出、耗时、token 都被记录成 trace 树，
从而能定位"回答慢，是慢在检索还是慢在生成"。

运行（仅本地输出，不上报）：
    .venv/bin/python langsmith/trace_demo.py

运行（开启追踪，需 LangSmith API Key，免费注册）：
    在 .env 里配置 LANGSMITH_TRACING=true / LANGSMITH_API_KEY / LANGSMITH_PROJECT
    .venv/bin/python langsmith/trace_demo.py

运行后到 https://smith.langchain.com 对应项目里查看 trace 树。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import traceable

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------- 被测对象：一个极简 RAG ----------
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    temperature=0,
)

# 模拟知识库（真实场景是向量库检索）
KNOWLEDGE = {
    "高血压": "高血压患者应低盐低脂饮食，每日食盐摄入不超过5克；建议每周3-5次中等强度有氧运动。",
    "糖尿病": "糖尿病典型症状为三多一少：多饮、多食、多尿、体重减少。",
    "阿司匹林": "阿司匹林具有解热镇痛、抗血小板聚集的作用，长期服用需关注胃肠道反应。",
}


@traceable(name="retrieve", run_type="retriever")
def retrieve(question: str) -> list[str]:
    """关键词检索（真实场景替换为向量检索）"""
    hits = [v for k, v in KNOWLEDGE.items() if k in question]
    return hits or []


@traceable(name="generate", run_type="llm")
def generate(question: str, contexts: list[str]) -> str:
    """基于上下文生成答案"""
    ctx = "\n".join(contexts) if contexts else "（无检索结果）"
    resp = llm.invoke(
        [
            {"role": "system", "content": "你是医疗问答助手，只依据给定资料回答，资料不足时明确说明。"},
            {"role": "user", "content": f"资料：\n{ctx}\n\n问题：{question}"},
        ]
    )
    return resp.content


@traceable(name="rag_pipeline", run_type="chain")
def rag_pipeline(question: str) -> dict:
    """完整链路：检索 → 生成（在 LangSmith 里呈现为一棵 trace 树）"""
    contexts = retrieve(question)
    answer = generate(question, contexts)
    return {"question": question, "contexts": contexts, "answer": answer}


if __name__ == "__main__":
    questions = [
        "高血压患者平时要注意什么？",
        "阿司匹林有什么作用？",
        "骨质疏松怎么预防？",  # 知识库无覆盖，考察"资料不足"处理
    ]

    tracing_on = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    print(f"LangSmith 追踪：{'已开启' if tracing_on else '未开启（仅本地输出）'}\n")

    for q in questions:
        r = rag_pipeline(q)
        print(f"Q: {r['question']}")
        print(f"  检索命中: {len(r['contexts'])} 条")
        print(f"  A: {r['answer'][:80]}...")
        print()

    if tracing_on:
        print("已上报到 LangSmith，打开 https://smith.langchain.com 查看 trace 树")
    else:
        print("提示：在 .env 设置 LANGSMITH_TRACING=true 与 LANGSMITH_API_KEY 后可上报追踪")
