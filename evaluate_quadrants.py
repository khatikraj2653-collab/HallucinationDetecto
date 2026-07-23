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
from consistency_utils import generate_multiple_answers

from dotenv import load_dotenv
load_dotenv()

print("Downloading HaluEval QA data...")
url = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
response = requests.get(url)
lines = response.text.strip().split("\n")
data = [json.loads(line) for line in lines]

random.seed(42)
sample = random.sample(data, 20)

nli_model = pipeline("text-classification", model="facebook/bart-large-mnli", device=-1)

def check_claim(context, claim):
    result = nli_model(f"{context}</s></s>{claim}")[0]
    return result["label"], round(result["score"], 3)

def get_quadrant(is_flagged, is_consistent):
    if is_flagged and not is_consistent:
        return "Flagged+Inconsistent"
    elif is_flagged and is_consistent:
        return "Flagged+Consistent"
    elif not is_flagged and not is_consistent:
        return "NotFlagged+Inconsistent"
    else:
        return "NotFlagged+Consistent"

results = []

for idx, example in enumerate(sample):
    knowledge = example["knowledge"]
    right_answer = example["right_answer"]
    hallucinated_answer = example["hallucinated_answer"]

    print(f"[{idx+1}/{len(sample)}]")

    vectorstore = build_vectorstore(knowledge)

    for label, answer_text in [("faithful", right_answer), ("hallucinated", hallucinated_answer)]:
        try:
            if is_refusal(answer_text):
                continue

            question_for_generation = example["question"]
            real_samples = generate_multiple_answers(knowledge, question_for_generation, n=3)
            consistency_results = multi_consistency_check(real_samples[0], real_samples[1], real_samples[2])
            consistent_votes = sum(1 for v in consistency_results.values() if "ERROR" not in v.upper() and "CONSISTENT" in v.upper() and "INCONSISTENT" not in v.upper())
            total_votes = sum(1 for v in consistency_results.values() if "ERROR" not in v.upper())
            is_consistent = (consistent_votes >= total_votes / 2) if total_votes > 0 else True

            nli_label, nli_score = check_claim(knowledge, answer_text)
            is_grounded, rag_score, best_chunk = check_claim_grounding(vectorstore, answer_text)
            judge_results = multi_llm_judge(knowledge, answer_text)

            bad_votes = 0
            total = 0
            if nli_label.upper() == "CONTRADICTION":
                bad_votes += 1
            total += 1
            if not is_grounded:
                bad_votes += 1
            total += 1
            for v in judge_results.values():
                if "ERROR" not in v.upper():
                    total += 1
                    if "UNSUPPORTED" in v.upper():
                        bad_votes += 1

            is_flagged = (bad_votes / total) >= 0.5 if total > 0 else False

            quadrant = get_quadrant(is_flagged, is_consistent)
            actual_hallucination = (label == "hallucinated")

            results.append({
                "answer_type": label,
                "actual_hallucination": actual_hallucination,
                "is_flagged": is_flagged,
                "is_consistent": is_consistent,
                "quadrant": quadrant
            })

            print(f"  {label}: quadrant={quadrant}")

        except Exception as e:
            print(f"  Error: {e}")

    time.sleep(1)

print(f"\n\n=== QUADRANT BREAKDOWN (n={len(results)}) ===")

quadrant_names = ["Flagged+Inconsistent", "Flagged+Consistent", "NotFlagged+Inconsistent", "NotFlagged+Consistent"]

for q in quadrant_names:
    all_in_q = [r for r in results if r["quadrant"] == q]
    hallucinations_in_q = [r for r in all_in_q if r["actual_hallucination"]]
    print(f"\n{q}: {len(all_in_q)} total claims")
    if all_in_q:
        pct_hallucination = round((len(hallucinations_in_q) / len(all_in_q)) * 100, 1)
        print(f"  Of these, {len(hallucinations_in_q)} are real hallucinations ({pct_hallucination}%)")

total_hallucinations = sum(1 for r in results if r["actual_hallucination"])
print(f"\n\nTotal real hallucinations in sample: {total_hallucinations}")
for q in quadrant_names:
    count = sum(1 for r in results if r["quadrant"] == q and r["actual_hallucination"])
    pct = round((count / total_hallucinations) * 100, 1) if total_hallucinations > 0 else 0
    print(f"  {pct}% of all real hallucinations fell into {q}")

with open("quadrant_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved to quadrant_results.json")