import json

def calculate_metrics(filepaths, dataset_name):
    results = []
    for filepath in filepaths:
        with open(filepath) as f:
            results.extend(json.load(f))

    true_positives = sum(1 for r in results if r["actual_hallucination"] and r["predicted_hallucination"])
    false_positives = sum(1 for r in results if not r["actual_hallucination"] and r["predicted_hallucination"])
    true_negatives = sum(1 for r in results if not r["actual_hallucination"] and not r["predicted_hallucination"])
    false_negatives = sum(1 for r in results if r["actual_hallucination"] and not r["predicted_hallucination"])

    total = len(results)
    accuracy = round(((true_positives + true_negatives) / total) * 100, 1) if total > 0 else 0.0

    precision = round((true_positives / (true_positives + false_positives)) * 100, 1) if (true_positives + false_positives) > 0 else 0.0
    recall = round((true_positives / (true_positives + false_negatives)) * 100, 1) if (true_positives + false_negatives) > 0 else 0.0
    f1 = round((2 * precision * recall) / (precision + recall), 1) if (precision + recall) > 0 else 0.0

    print(f"\n=== {dataset_name} ===")
    print(f"Total examples: {total}")
    print(f"True Positives (correctly caught hallucinations): {true_positives}")
    print(f"False Positives (false alarms on true answers): {false_positives}")
    print(f"True Negatives (correctly passed true answers): {true_negatives}")
    print(f"False Negatives (missed real hallucinations): {false_negatives}")
    print(f"Accuracy: {accuracy}%")
    print(f"Precision: {precision}%")
    print(f"Recall: {recall}%")
    print(f"F1 Score: {f1}%")

    return {
        "dataset": dataset_name, "total": total, "tp": true_positives, "fp": false_positives,
        "tn": true_negatives, "fn": false_negatives, "accuracy": accuracy,
        "precision": precision, "recall": recall, "f1": f1
    }

halueval_metrics = calculate_metrics(["halueval_results.json", "halueval_results_batch2.json"], "HaluEval (combined)")
truthfulqa_metrics = calculate_metrics(["truthfulqa_results.json", "truthfulqa_results_batch2.json"], "TruthfulQA (combined)")

with open("metrics_summary.json", "w") as f:
    json.dump({"halueval": halueval_metrics, "truthfulqa": truthfulqa_metrics}, f, indent=2)

print("\n\nSaved summary to metrics_summary.json")