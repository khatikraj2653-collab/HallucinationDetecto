import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import sys
sys.path.append(os.path.dirname(__file__))

import json
import random
import time
from transformers import pipeline
from datasets import load_dataset

from rag_utils import build_vectorstore, check_claim_grounding
from llm_judge import multi_llm_judge, multi_consistency_check
from refusal_utils import is_refusal
from scoring_utils import compute_final_verdict

from dotenv import load_dotenv
load_dotenv()

print("Downloading TruthfulQA data...")
dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
print(f"Loaded {len(dataset)} total examples.")

random.seed(99)
indices = random.sample(range(len(dataset)), 20)
sample = [dataset[i] for i in indices]

nli_model = pipeline("text-classification", model="facebook/bart-large-mnli", device=-1)

def check_claim(context, claim):
    result = nli_model(f"{context}</s></s>{claim}")[0]
    return result["label"], round(result["score"], 3)

def evaluate_claim(context, claim, is_consistent, vectorstore):
    if is_refusal(claim):
        return None
    nli_label, nli_score = check_claim(context, claim)
    is_grounded, rag_score, best_chunk = check_claim_grounding(vectorstore, claim)
    judge_results = multi_llm_judge(context, claim)
    verdict, reason = compute_final_verdict(nli_label, is_grounded, judge_results, is_consistent)
    return "Hallucination" in verdict

results = []

for idx, example in enumerate(sample):
    question = example["question"]
    best_answer = example["best_answer"]
    incorrect_answers = example["incorrect_answers"]

    if not incorrect_answers:
        continue

    incorrect_answer = incorrect_answers[0]
    knowledge = best_answer

    print(f"\n[{idx+1}/{len(sample)}] {question[:60]}...")

    vectorstore = build_vectorstore(knowledge)

    for label, answer_text in [("correct", best_answer), ("incorrect", incorrect_answer)]:
        try:
            fake_answers = [answer_text, answer_text, answer_text]
            consistency_results = multi_consistency_check(fake_answers[0], fake_answers[1], fake_answers[2])
            consistent_votes = sum(1 for v in consistency_results.values() if "ERROR" not in v.upper() and "CONSISTENT" in v.upper() and "INCONSISTENT" not in v.upper())
            total_votes = sum(1 for v in consistency_results.values() if "ERROR" not in v.upper())
            is_consistent = (consistent_votes >= total_votes / 2) if total_votes > 0 else True

            predicted_hallucination = evaluate_claim(knowledge, answer_text, is_consistent, vectorstore)

            if predicted_hallucination is None:
                continue

            actual_hallucination = (label == "incorrect")
            correct = (predicted_hallucination == actual_hallucination)

            results.append({
                "question": question,
                "answer_type": label,
                "actual_hallucination": actual_hallucination,
                "predicted_hallucination": predicted_hallucination,
                "correct": correct
            })

            print(f"  {label}: actual={actual_hallucination}, predicted={predicted_hallucination}, correct={correct}")

        except Exception as e:
            print(f"  Error on {label}: {e}")

    time.sleep(1)

total = len(results)
correct = sum(1 for r in results if r["correct"])
accuracy = round((correct / total) * 100, 1) if total > 0 else 0.0

print(f"\n\n=== RESULTS ===")
print(f"Total evaluated: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy}%")

with open("truthfulqa_results_batch2.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved detailed results to truthfulqa_results_batch2.json")