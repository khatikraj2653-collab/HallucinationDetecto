import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import sys
sys.path.append(os.path.dirname(__file__))

import json
import random
import requests
import time
from transformers import pipeline

from rag_utils import build_vectorstore, check_claim_grounding
from llm_judge import multi_llm_judge, multi_consistency_check
from refusal_utils import is_refusal
from ablation_utils import score_nli_only, score_nli_rag, score_nli_rag_judges, score_full_ensemble

from dotenv import load_dotenv
load_dotenv()

print("Downloading HaluEval QA data...")
url = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
response = requests.get(url)
lines = response.text.strip().split("\n")
data = [json.loads(line) for line in lines]
print(f"Loaded {len(data)} total examples.")

random.seed(42)
sample = random.sample(data, 20)

nli_model = pipeline("text-classification", model="facebook/bart-large-mnli", device=-1)

def check_claim(context, claim):
    result = nli_model(f"{context}</s></s>{claim}")[0]
    return result["label"], round(result["score"], 3)

results = []

for idx, example in enumerate(sample):
    knowledge = example["knowledge"]
    question = example["question"]
    right_answer = example["right_answer"]
    hallucinated_answer = example["hallucinated_answer"]

    print(f"\n[{idx+1}/{len(sample)}] {question[:60]}...")

    vectorstore = build_vectorstore(knowledge)

    for label, answer_text in [("faithful", right_answer), ("hallucinated", hallucinated_answer)]:
        try:
            if is_refusal(answer_text):
                continue

            fake_answers = [answer_text, answer_text, answer_text]
            consistency_results = multi_consistency_check(fake_answers[0], fake_answers[1], fake_answers[2])
            consistent_votes = sum(1 for v in consistency_results.values() if "ERROR" not in v.upper() and "CONSISTENT" in v.upper() and "INCONSISTENT" not in v.upper())
            total_votes = sum(1 for v in consistency_results.values() if "ERROR" not in v.upper())
            is_consistent = (consistent_votes >= total_votes / 2) if total_votes > 0 else True

            nli_label, nli_score = check_claim(knowledge, answer_text)
            is_grounded, rag_score, best_chunk = check_claim_grounding(vectorstore, answer_text)
            judge_results = multi_llm_judge(knowledge, answer_text)

            actual_hallucination = (label == "hallucinated")

            pred_nli = score_nli_only(nli_label)
            pred_nli_rag = score_nli_rag(nli_label, is_grounded)
            pred_nli_rag_judges = score_nli_rag_judges(nli_label, is_grounded, judge_results)
            pred_full = score_full_ensemble(nli_label, is_grounded, judge_results, is_consistent)

            results.append({
                "question": question,
                "answer_type": label,
                "actual_hallucination": actual_hallucination,
                "pred_nli_only": pred_nli,
                "pred_nli_rag": pred_nli_rag,
                "pred_nli_rag_judges": pred_nli_rag_judges,
                "pred_full_ensemble": pred_full
            })

            print(f"  {label}: actual={actual_hallucination} | NLI={pred_nli} | +RAG={pred_nli_rag} | +Judges={pred_nli_rag_judges} | Full={pred_full}")

        except Exception as e:
            print(f"  Error on {label}: {e}")

    time.sleep(1)

def calc_accuracy(results, pred_key):
    correct = sum(1 for r in results if r[pred_key] == r["actual_hallucination"])
    total = len(results)
    return round((correct / total) * 100, 1) if total > 0 else 0.0

print(f"\n\n=== ABLATION RESULTS (n={len(results)}) ===")
print(f"NLI only:              {calc_accuracy(results, 'pred_nli_only')}%")
print(f"NLI + RAG:             {calc_accuracy(results, 'pred_nli_rag')}%")
print(f"NLI + RAG + Judges:    {calc_accuracy(results, 'pred_nli_rag_judges')}%")
print(f"Full ensemble (6):     {calc_accuracy(results, 'pred_full_ensemble')}%")

with open("ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved detailed results to ablation_results.json")