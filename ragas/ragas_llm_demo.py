# -*- coding: utf-8 -*-
"""
Ragas RAG 评测 demo —— 用 DeepSeek 当评测 LLM，跑两个指标

  1. Faithfulness        忠实度：答案是否忠于检索上下文（幻觉检测，0~1）
  2. FactualCorrectness  事实正确性：答案与标准答案的事实一致程度（0~1）

运行：
  .venv/bin/python ragas/ragas_llm_demo.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.metrics._faithfulness import Faithfulness

# ===== 1. 从环境变量读取 Key（不硬编码、不落盘） =====
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# DeepSeek 兼容 OpenAI 协议，用 ChatOpenAI 指向 DeepSeek 的 endpoint
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    temperature=0,
)

# ===== 2. 评测集：3 条医疗问诊 RAG 样本（好 / 中 / 差） =====
samples = [
    # 样本1：答案全部来自上下文 —— 忠实度应该高
    SingleTurnSample(
        user_input="高血压患者日常需要注意什么？",
        retrieved_contexts=[
            "高血压患者应低盐低脂饮食，每日食盐摄入不超过5克。",
            "建议每周进行3-5次中等强度有氧运动，如快走、游泳。",
        ],
        response="高血压患者需要低盐饮食，每日食盐不超过5克，并保持每周3-5次中等强度运动。",
        reference="高血压患者应低盐饮食（每日不超过5克），并坚持每周3-5次中等强度有氧运动。",
    ),
    # 样本2：答案部分来自上下文，但漏掉"多食、体重减少" —— 忠实度高但事实正确性中等
    SingleTurnSample(
        user_input="糖尿病的典型症状是什么？",
        retrieved_contexts=[
            "典型症状为'三多一少'：多饮、多食、多尿、体重减少。",
        ],
        response="糖尿病的典型症状是多饮、多尿。",
        reference="糖尿病典型症状为三多一少：多饮、多食、多尿、体重减少。",
    ),
    # 样本3：答案与上下文无关（幻觉）—— 忠实度应该很低
    SingleTurnSample(
        user_input="阿司匹林的主要作用是什么？",
        retrieved_contexts=[
            "布洛芬属于非甾体抗炎药，主要用于缓解疼痛和退热。",
        ],
        response="阿司匹林主要作用是补充维生素，改善睡眠质量。",
        reference="阿司匹林具有解热镇痛、抗血小板聚集的作用。",
    ),
]

dataset = EvaluationDataset(samples)

# ===== 3. 执行 LLM 评测 =====
metrics = [
    Faithfulness(),
    FactualCorrectness(),
]

result = evaluate(dataset, metrics=metrics, llm=llm)

# ===== 4. 输出报告 =====
df = result.to_pandas()
cols = [c for c in df.columns if c.startswith(("user_input", "faithfulness", "factual"))]
print("\n" + "=" * 70)
print("RAG 评测结果（DeepSeek 作为评测 LLM，1 分最好）")
print("=" * 70)
print(df[cols].to_string(index=False, max_colwidth=16))

print("\n指标均值:")
print(df.select_dtypes("number").mean().round(3).to_string())

print("""
================================================================
【结果解读要点】
  - 忠实度(Faithfulness)：只看"答案是否由上下文支撑"，不判断对错
      样本1 高 / 样本3 低 —— 幻觉被检测出来
  - 事实正确性(FactualCorrectness)：对照标准答案做事实层面比对
      样本2 忠实度不低但正确性中等 —— "没说错话，但没说全"
  - 排查链路：忠实度低 = 检索/生成跑偏；忠实度高但正确性低 = 知识覆盖不足
================================================================
""")
