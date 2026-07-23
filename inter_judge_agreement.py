import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import sys
sys.path.append(os.path.dirname(__file__))

import json
import random
import requests
import time

from rag_utils import build_vectorstore
from llm_judge import multi_llm_judge
from refusal_utils import is_refusal

from dotenv import load_dotenv
load_dotenv()

print("Downloading HaluEval QA data...")
url = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
response = requests.get(url)
lines = response.text.strip().split("\n")
data = [json.loads(line) for line in lines]

random.seed(7)
sample = random.sample(data, 20)

judge_records = []

for idx, example in enumerate(sample):
    knowledge = example["knowledge"]
    right_answer = example["right_answer"]
    hallucinated_answer = example["hallucinated_answer"]

    print(f"[{idx+1}/{len(sample)}]")

    for answer_text in [right_answer, hallucinated_answer]:
        if is_refusal(answer_text):
            continue
        try:
            judge_results = multi_llm_judge(knowledge, answer_text)
            valid = {k: v for k, v in judge_results.items() if "ERROR" not in v.upper()}
            if len(valid) < 2:
                continue

            def normalize(v):
                if "UNSUPPORTED" in v.upper():
                    return "UNSUPPORTED"
                elif "SUPPORTED" in v.upper():
                    return "SUPPORTED"
                else:
                    return "UNCLEAR"

            record = {k: normalize(v) for k, v in valid.items()}
            judge_records.append(record)

        except Exception as e:
            print(f"  Error: {e}")

    time.sleep(1)

pairs = [("OpenAI", "Groq"), ("OpenAI", "Gemini"), ("Groq", "Gemini")]
print(f"\n=== Inter-Judge Agreement (n={len(judge_records)} claims) ===")

for a, b in pairs:
    valid_records = [r for r in judge_records if a in r and b in r]
    if not valid_records:
        print(f"{a} vs {b}: no overlapping data")
        continue
    agree = sum(1 for r in valid_records if r[a] == r[b])
    total = len(valid_records)
    rate = round((agree / total) * 100, 1) if total > 0 else 0.0
    print(f"{a} vs {b}: {agree}/{total} agree = {rate}%")

with open("inter_judge_agreement.json", "w") as f:
    json.dump(judge_records, f, indent=2)

print("\nSaved to inter_judge_agreement.json")