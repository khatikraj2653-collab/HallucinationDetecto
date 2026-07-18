def is_flagged_by_detection(nli_label, is_grounded, judge_results):
    """Returns True if 2 or more detection signals flag this claim as unsupported."""
    flags = 0
    total = 0

    total += 1
    if nli_label.upper() == "CONTRADICTION":
        flags += 1

    total += 1
    if not is_grounded:
        flags += 1

    for verdict in judge_results.values():
        if "ERROR" not in verdict.upper():
            total += 1
            if "UNSUPPORTED" in verdict.upper():
                flags += 1

    return flags >= 2, flags, total


def compute_final_verdict(nli_label, is_grounded, judge_results, is_consistent):
    """
    Combines detection signals + cross-sample consistency into one final verdict,
    using the 4-quadrant logic:
    - Flagged + Inconsistent  -> High-Confidence Hallucination
    - Flagged + Consistent    -> Possible Hallucination (systematic bias)
    - Not Flagged + Consistent -> Faithful
    - Not Flagged + Inconsistent -> Possible Hallucination (consistency signal only)
    """
    flagged, flag_count, total_checks = is_flagged_by_detection(nli_label, is_grounded, judge_results)

    if flagged and not is_consistent:
        return "🚩 High-Confidence Hallucination", f"Flagged by {flag_count}/{total_checks} detectors AND unstable across samples"
    elif flagged and is_consistent:
        return "🟠 Possible Hallucination (systematic bias)", f"Flagged by {flag_count}/{total_checks} detectors, but stable across samples"
    elif not flagged and not is_consistent:
        return "🟡 Possible Hallucination (consistency signal only)", "Not flagged by detectors, but unstable across samples"
    else:
        return "✅ Faithful", f"Not flagged by any detector ({flag_count}/{total_checks}), consistent across samples"