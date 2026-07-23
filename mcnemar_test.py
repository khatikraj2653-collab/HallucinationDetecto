import json
from scipy.stats import binomtest

def mcnemar_exact(results, key_a, key_b, label_a, label_b):
    b_correct_a_wrong = 0
    a_correct_b_wrong = 0

    for r in results:
        actual = r["actual_hallucination"]
        a_correct = (r[key_a] == actual)
        b_correct = (r[key_b] == actual)

        if a_correct and not b_correct:
            a_correct_b_wrong += 1
        elif b_correct and not a_correct:
            b_correct_a_wrong += 1

    n_discordant = a_correct_b_wrong + b_correct_a_wrong

    if n_discordant == 0:
        print(f"\n{label_a} vs {label_b}: No discordant pairs — predictions identical on every example.")
        return

    result = binomtest(min(a_correct_b_wrong, b_correct_a_wrong), n_discordant, 0.5)
    p_value = result.pvalue

    print(f"\n{label_a} vs {label_b}:")
    print(f"  {label_a} correct, {label_b} wrong: {a_correct_b_wrong}")
    print(f"  {label_b} correct, {label_a} wrong: {b_correct_a_wrong}")
    print(f"  McNemar's exact test p-value: {round(p_value, 4)}")
    if p_value < 0.05:
        print(f"  → Statistically significant difference (p < 0.05)")
    else:
        print(f"  → No statistically significant difference (p >= 0.05)")


with open("ablation_results.json") as f:
    results = json.load(f)

print(f"=== McNemar's Exact Test (n={len(results)}) ===")

mcnemar_exact(results, "pred_nli_rag_judges", "pred_full_ensemble", "NLI+RAG+Judges", "Full Ensemble (6)")
mcnemar_exact(results, "pred_nli_only", "pred_nli_rag", "NLI only", "NLI+RAG")
mcnemar_exact(results, "pred_nli_only", "pred_nli_rag_judges", "NLI only", "NLI+RAG+Judges")