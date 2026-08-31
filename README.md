# Hallucination Detector

A hallucination detection and evaluation system for LLM outputs, with an evaluation harness across HaluEval, TruthfulQA, and a hand-labeled dataset, plus ablation studies and confidence-interval analysis.

**Live app:** https://hallucinationdetecto.streamlit.app/

## Demo
![HallucinationDetector demo](hallucinationdetector-demo.gif)

## Stack
- Streamlit (frontend)
- LangChain (OpenAI, Groq, Google Generative AI backends)
- FAISS-based retrieval for consistency checking
- SQLite (logging)

## Architecture

```mermaid
flowchart TD
    A["User input: Context + Question<br/>(frontend/app.py)"] --> B["generate_multiple_answers()<br/>consistency_utils.py<br/>GPT-4o, temp=0.9, n=3"]
    B --> C["multi_consistency_check()<br/>llm_judge.py<br/>cross-sample consistency vote"]
    B --> D["build_vectorstore(context)<br/>rag_utils.py — FAISS index"]
    C --> E
    D --> E["analyze_answers_parallel()<br/>graph_utils.py<br/>ThreadPoolExecutor per answer"]

    E --> F["split_claims_fn()<br/>sentence-split each answer"]
    F --> G["analyze_claim() per claim<br/>invokes claim_subgraph (LangGraph)"]

    subgraph SG["claim_subgraph (StateGraph)"]
        G1["node_check_refusal<br/>is_refusal() — refusal_utils.py"] --> G2["node_run_detectors_parallel<br/>ThreadPoolExecutor(3 workers)"]
        G2 --> N["NLI check<br/>facebook/bart-large-mnli"]
        G2 --> R["check_claim_grounding()<br/>rag_utils.py — FAISS cosine sim"]
        G2 --> J["multi_llm_judge()<br/>llm_judge.py — GPT-4o, Groq Llama-3.3, Gemini"]
    end

    G --> G1
    N --> H["Per-claim signal votes:<br/>NLI + RAG + 3 judges + consistency"]
    R --> H
    J --> H

    H --> I["Answer score = bad_votes / total_signals<br/>(computed in frontend/app.py)"]
    I --> K["Verdict badges rendered in Streamlit UI<br/>(green/orange/red)"]

    H --> L["log_claim()<br/>db_utils.py — SQLite hallucination_logs.db"]
    A --> M["log_event('visit')<br/>log_client.py"]
```

## Evaluation
- `evaluate_halueval.py`, `evaluate_truthfulqa.py`, `evaluate_handlabeled.py` — benchmark evaluation runs
- `evaluate_ablation.py` / `analyze_results.py` — ablation studies
- `calculate_confidence_intervals.py` — statistical confidence intervals on results

## Run locally
```bash
pip install -r requirements.txt
streamlit run frontend/app.py
```
