import json
import math

def wilson_score_interval(successes, total, confidence=0.95):
    if total == 0:
        return 0.0, 0.0, 0.0

    z = 1.96 if confidence == 0.95 else 1.645
    p_hat = successes / total

    denominator = 1 + (z**2 / total)
    center = (p_hat + (z**2) / (2 * total)) / denominator
    margin = (z / denominator) * math.sqrt((p_hat * (1 - p_hat) / total) + (z**2 / (4 * total**2)))

    lower = max(0, center - margin) * 100
    upper = min(1, center + margin) * 100
    point = p_hat * 100

    return round(point, 1), round(lower, 1), round(upper, 1)


def analyze_with_ci(filepaths, dataset_name):
    results = []
    for filepath in filepaths:
        with open(filepath) as f:
            results.extend(json.load(f))

    total = len(results)
    correct = sum(1 for r in results if r["predicted_hallucination"] == r["actual_hallucination"])

    point, lower, upper = wilson_score_interval(correct, total)

    print(f"\n=== {dataset_name} ===")
    print(f"n = {total}")
    print(f"Accuracy: {point}% (95% CI: {lower}% - {upper}%)")

    return {"dataset": dataset_name, "n": total, "accuracy": point, "ci_lower": lower, "ci_upper": upper}


halueval_ci = analyze_with_ci(["halueval_results.json", "halueval_results_batch2.json"], "HaluEval")
truthfulqa_ci = analyze_with_ci(["truthfulqa_results.json", "truthfulqa_results_batch2.json"], "TruthfulQA")

def analyze_ablation_ci(filepath):
    with open(filepath) as f:
        results = json.load(f)

    total = len(results)
    configs = {
        "NLI only": "pred_nli_only",
        "NLI + RAG": "pred_nli_rag",
        "NLI + RAG + Judges": "pred_nli_rag_judges",
        "Full ensemble (6)": "pred_full_ensemble"
    }

    print(f"\n=== Ablation Study Confidence Intervals (n={total}) ===")
    ablation_ci_results = {}
    for name, key in configs.items():
        correct = sum(1 for r in results if r[key] == r["actual_hallucination"])
        point, lower, upper = wilson_score_interval(correct, total)
        print(f"{name}: {point}% (95% CI: {lower}% - {upper}%)")
        ablation_ci_results[name] = {"accuracy": point, "ci_lower": lower, "ci_upper": upper}

    return ablation_ci_results

ablation_ci = analyze_ablation_ci("ablation_results.json")

with open("confidence_intervals.json", "w") as f:
    json.dump({"halueval": halueval_ci, "truthfulqa": truthfulqa_ci, "ablation": ablation_ci}, f, indent=2)

print("\nSaved to confidence_intervals.json")