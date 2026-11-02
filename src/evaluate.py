import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chain import CustomerServiceChain


PROMPT_FAITHFULNESS = """你是一个评估专家。请判断以下AI回答是否完全基于提供的知识库内容，有没有编造不存在的信息。

知识库内容:
{context}

AI回答:
{answer}

请评分(0-1):
- 1.0: 所有陈述都能在知识库中找到依据
- 0.5: 部分陈述有依据，部分无法验证
- 0.0: 大部分是编造的

只返回分数，不要解释。"""

PROMPT_RELEVANCY = """你是一个评估专家。请判断以下AI回答是否与用户问题相关。

用户问题: {question}
AI回答: {answer}

请评分(0-1):
- 1.0: 回答完全针对问题，没有偏题
- 0.5: 部分相关，但有冗余或不相关内容
- 0.0: 答非所问

只返回分数，不要解释。"""

PROMPT_PRECISION = """你是一个评估专家。请判断检索到的知识库段落是否与问题相关，是否包含有用的信息。

用户问题: {question}

检索到的段落:
{context}

请评分(0-1):
- 1.0: 大部分检索段落都直接相关且有帮助
- 0.5: 部分相关，但夹杂了无关段落
- 0.0: 检索段落与问题完全无关

只返回分数，不要解释。"""


def score_with_llm(client: OpenAI, prompt: str) -> float:
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        text = resp.choices[0].message.content.strip()
        return float(text)
    except Exception:
        return 0.0


def run_evaluation():
    print("=" * 60)
    print("  RAG 评测 — 手动 DeepSeek 打分")
    print("=" * 60)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    path = PROJECT_ROOT / "data" / "eval_questions.json"
    with open(path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"\n[加载] {len(test_data)} 条测试用例")

    print("[初始化] 加载知识库和模型...")
    chain = CustomerServiceChain()

    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []
    details = []

    for i, item in enumerate(test_data):
        question = item["question"]
        results = chain.retrieve(question, top_k=3)
        contexts = [text for _, text in results]
        answer = chain.answer(question)

        ctx_text = "\n---\n".join(contexts[:3])

        faith_score = score_with_llm(
            client,
            PROMPT_FAITHFULNESS.format(context=ctx_text[:1500], answer=answer[:800]),
        )
        rel_score = score_with_llm(
            client,
            PROMPT_RELEVANCY.format(question=question, answer=answer[:800]),
        )
        prec_score = score_with_llm(
            client,
            PROMPT_PRECISION.format(question=question, context=ctx_text[:1500]),
        )

        faithfulness_scores.append(faith_score)
        relevancy_scores.append(rel_score)
        precision_scores.append(prec_score)

        details.append({
            "question": question,
            "faithfulness": faith_score,
            "relevancy": rel_score,
            "precision": prec_score,
        })

        avg = (faith_score + rel_score + prec_score) / 3
        status = "✓" if avg >= 0.7 else ("△" if avg >= 0.4 else "✗")
        print(f"  {status} [{i+1:2d}/{len(test_data)}] {question[:35]:<35s} "
              f"F:{faith_score:.2f} R:{rel_score:.2f} P:{prec_score:.2f}")

    avg_f = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_r = sum(relevancy_scores) / len(relevancy_scores)
    avg_p = sum(precision_scores) / len(precision_scores)
    overall = (avg_f + avg_r + avg_p) / 3

    print("\n" + "=" * 60)
    print("  评测结果")
    print("=" * 60)
    for name, val in [("Faithfulness", avg_f), ("Answer Relevancy", avg_r), ("Context Precision", avg_p)]:
        bar = "█" * int(val * 25) + "░" * (25 - int(val * 25))
        print(f"  {name:<20s}: {val:.4f}  {bar}")
    print(f"  {'Overall':<20s}: {overall:.4f}")
    print(f"\n  测试用例数: {len(test_data)}")
    print(f"  通过率(>0.7): {sum(1 for d in details if (d['faithfulness']+d['relevancy']+d['precision'])/3 >= 0.7)}/{len(test_data)}")

    out_path = PROJECT_ROOT / "data" / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {"faithfulness": avg_f, "relevancy": avg_r, "precision": avg_p, "overall": overall},
            "details": details,
        }, f, ensure_ascii=False, indent=2)
    print(f"  详细结果已保存: {out_path}")

    return overall


if __name__ == "__main__":
    run_evaluation()
