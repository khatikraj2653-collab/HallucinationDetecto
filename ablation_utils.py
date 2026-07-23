def score_nli_only(nli_label):
    score = 1 if nli_label.upper() == "CONTRADICTION" else 0
    total = 1
    return (score / total) * 100 >= 50

def score_nli_rag(nli_label, is_grounded):
    score = 0
    total = 2
    if nli_label.upper() == "CONTRADICTION":
        score += 1
    if not is_grounded:
        score += 1
    return (score / total) * 100 >= 50

def score_nli_rag_judges(nli_label, is_grounded, judge_results):
    score = 0
    total = 0
    if nli_label.upper() == "CONTRADICTION":
        score += 1
    total += 1
    if not is_grounded:
        score += 1
    total += 1
    for v in judge_results.values():
        if "ERROR" not in v.upper():
            total += 1
            if "UNSUPPORTED" in v.upper():
                score += 1
    return (score / total) * 100 >= 50 if total > 0 else False

def score_full_ensemble(nli_label, is_grounded, judge_results, is_consistent):
    score = 0
    total = 0
    if nli_label.upper() == "CONTRADICTION":
        score += 1
    total += 1
    if not is_grounded:
        score += 1
    total += 1
    for v in judge_results.values():
        if "ERROR" not in v.upper():
            total += 1
            if "UNSUPPORTED" in v.upper():
                score += 1
    if not is_consistent:
        score += 1
    total += 1
    return (score / total) * 100 >= 34 if total > 0 else False