def compute_final_verdict(nli_label, is_grounded, judge_results, is_consistent):
    """
    Equal-weight scoring: 6 signals total, each worth 1 point if flagged.
    - NLI (1 point if CONTRADICTION)
    - RAG-grounding (1 point if Not Grounded)
    - OpenAI judge (1 point if UNSUPPORTED)
    - Groq judge (1 point if UNSUPPORTED)
    - Gemini judge (1 point if UNSUPPORTED)
    - Cross-sample consistency (1 point if Inconsistent)

    hallucination_% = (points / 6) * 100

    >= 67%  -> High-Confidence Hallucination
    34-66%  -> Possible Hallucination
    < 34%   -> Faithful
    """
    score = 0
    total = 6

    if nli_label.upper() == "CONTRADICTION":
        score += 1

    if not is_grounded:
        score += 1

    for verdict in judge_results.values():
        if "ERROR" not in verdict.upper() and "UNSUPPORTED" in verdict.upper():
            score += 1
        elif "ERROR" in verdict.upper():
            total -= 1

    if not is_consistent:
        score += 1

    hallucination_pct = round((score / total) * 100, 1) if total > 0 else 0.0

    if hallucination_pct >= 67:
        verdict = "🚩 High-Confidence Hallucination"
    elif hallucination_pct >= 34:
        verdict = "🟠 Possible Hallucination"
    else:
        verdict = "✅ Faithful"

    reason = f"{score}/{total} signals flagged ({hallucination_pct}%)"
    return verdict, reason