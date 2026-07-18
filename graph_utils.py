import os
from concurrent.futures import ThreadPoolExecutor
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any

from rag_utils import check_claim_grounding
from refusal_utils import is_refusal
from llm_judge import multi_llm_judge


class ClaimState(TypedDict):
    context: str
    claim: str
    vectorstore: Any
    nli_fn: Any
    nli_label: str
    nli_score: float
    is_grounded: bool
    rag_score: float
    best_chunk: str
    judge_results: Dict[str, str]
    is_refusal: bool


def node_check_refusal(state: ClaimState) -> ClaimState:
    state["is_refusal"] = is_refusal(state["claim"])
    return state


def node_run_detectors_parallel(state: ClaimState) -> ClaimState:
    if state["is_refusal"]:
        return state

    def run_nli():
        return state["nli_fn"](state["context"], state["claim"])

    def run_rag():
        return check_claim_grounding(state["vectorstore"], state["claim"])

    def run_judges():
        return multi_llm_judge(state["context"], state["claim"])

    with ThreadPoolExecutor(max_workers=3) as executor:
        nli_future = executor.submit(run_nli)
        rag_future = executor.submit(run_rag)
        judges_future = executor.submit(run_judges)

        nli_label, nli_score = nli_future.result()
        is_grounded, rag_score, best_chunk = rag_future.result()
        judge_results = judges_future.result()

    state["nli_label"] = nli_label
    state["nli_score"] = nli_score
    state["is_grounded"] = is_grounded
    state["rag_score"] = rag_score
    state["best_chunk"] = best_chunk
    state["judge_results"] = judge_results
    return state


def build_claim_subgraph():
    graph = StateGraph(ClaimState)
    graph.add_node("check_refusal", node_check_refusal)
    graph.add_node("run_detectors", node_run_detectors_parallel)
    graph.set_entry_point("check_refusal")
    graph.add_edge("check_refusal", "run_detectors")
    graph.add_edge("run_detectors", END)
    return graph.compile()


claim_subgraph = build_claim_subgraph()


def analyze_claim(context, claim, vectorstore, nli_fn):
    result = claim_subgraph.invoke({
        "context": context,
        "claim": claim,
        "vectorstore": vectorstore,
        "nli_fn": nli_fn,
        "nli_label": "", "nli_score": 0.0,
        "is_grounded": False, "rag_score": 0.0, "best_chunk": "",
        "judge_results": {}, "is_refusal": False
    })
    return result


def analyze_answers_parallel(context, answers, vectorstore, nli_fn, split_claims_fn):
    def analyze_one_answer(answer):
        claims = split_claims_fn(answer)
        claim_results = []
        for claim in claims:
            result = analyze_claim(context, claim, vectorstore, nli_fn)
            claim_results.append(result)
        return claim_results

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(analyze_one_answer, ans) for ans in answers]
        all_results = [f.result() for f in futures]

    return all_results