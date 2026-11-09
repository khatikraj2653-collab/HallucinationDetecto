import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_multiple_answers(context, question, n=3):
    answers = []
    for _ in range(n):
        response = client.chat.completions.create(
            model="gpt-6-luna",
            messages=[
                {"role": "system", "content": "Answer the question using only the given context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ],
            temperature=0.9,
            extra_body={"reasoning_effort": "none"}
        )
        answers.append(response.choices[0].message.content)
    return answers

CONSISTENCY_PROMPT = """You are comparing 3 independently generated answers to the same question, based on the same source context.

Answer 1: {a1}

Answer 2: {a2}

Answer 3: {a3}

Your task: identify SPECIFIC factual disagreements between the answers — only names, numbers, dates, or explicit claims that CONTRADICT each other across answers.

IMPORTANT rules:
- Trivial differences in wording, phrasing, length, greeting style, capitalization, or punctuation are NOT inconsistencies. Respond CONSISTENT.
- Different politeness ("Hello!" vs "hello" vs "Hi there!") is NOT an inconsistency. Respond CONSISTENT.
- An answer being shorter / more concise than another is NOT an inconsistency.
- Only flag genuine factual conflicts: different names for the same entity, contradicting numbers, contradicting dates, one answer asserting X and another asserting NOT X.
- If none of the answers contain extractable facts (e.g. they're all generic greetings, refusals, or "I don't know" style replies), respond CONSISTENT.

If there are NO genuine factual contradictions, respond with exactly: CONSISTENT

Otherwise, list the inconsistent facts, one per line."""

def check_consistency(judge_fn, a1, a2, a3):
    # Shortcut: if all three answers are identical (case/whitespace normalised),
    # they're trivially consistent — skip the LLM call entirely.
    norm = [(x or "").strip().lower() for x in (a1, a2, a3)]
    if norm[0] == norm[1] == norm[2]:
        return "CONSISTENT"
    prompt = CONSISTENCY_PROMPT.format(a1=a1, a2=a2, a3=a3)
    return judge_fn(prompt)