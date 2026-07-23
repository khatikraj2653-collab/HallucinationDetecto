import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import sys
sys.path.append(os.path.dirname(__file__))

import json
import time
from transformers import pipeline

from rag_utils import build_vectorstore, check_claim_grounding
from llm_judge import multi_llm_judge, multi_consistency_check
from refusal_utils import is_refusal
from scoring_utils import compute_final_verdict

from dotenv import load_dotenv
load_dotenv()

with open("hand_labeled_dataset.json") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} hand-labeled examples.")

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

for idx, example in enumerate(dataset):
    domain = example["domain"]
    context = example["context"]
    question = example["question"]
    correct_answer = example["correct_answer"]
    wrong_answer = example["wrong_answer"]

    print(f"\n[{idx+1}/{len(dataset)}] ({domain}) {question[:60]}...")

    vectorstore = build_vectorstore(context)

    for label, answer_text in [("correct", correct_answer), ("wrong", wrong_answer)]:
        try:
            fake_answers = [answer_text, answer_text, answer_text]
            consistency_results = multi_consistency_check(fake_answers[0], fake_answers[1], fake_answers[2])
            consistent_votes = sum(1 for v in consistency_results.values() if "ERROR" not in v.upper() and "CONSISTENT" in v.upper() and "INCONSISTENT" not in v.upper())
            total_votes = sum(1 for v in consistency_results.values() if "ERROR" not in v.upper())
            is_consistent = (consistent_votes >= total_votes / 2) if total_votes > 0 else True

            predicted_hallucination = evaluate_claim(context, answer_text, is_consistent, vectorstore)

            if predicted_hallucination is None:
                continue

            actual_hallucination = (label == "wrong")
            correct = (predicted_hallucination == actual_hallucination)

            results.append({
                "domain": domain,
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
correct_count = sum(1 for r in results if r["correct"])
accuracy = round((correct_count / total) * 100, 1) if total > 0 else 0.0

print(f"\n\n=== HAND-LABELED DATASET RESULTS ===")
print(f"Total evaluated: {total}")
print(f"Correct: {correct_count}")
print(f"Accuracy: {accuracy}%")

print("\n=== By Domain ===")
domains = set(r["domain"] for r in results)
for d in domains:
    domain_results = [r for r in results if r["domain"] == d]
    domain_correct = sum(1 for r in domain_results if r["correct"])
    domain_total = len(domain_results)
    domain_acc = round((domain_correct / domain_total) * 100, 1) if domain_total > 0 else 0.0
    print(f"{d}: {domain_correct}/{domain_total} = {domain_acc}%")

with open("handlabeled_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved detailed results to handlabeled_results.json")