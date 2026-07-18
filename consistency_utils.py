import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_multiple_answers(context, question, n=3):
    answers = []
    for _ in range(n):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Answer the question using only the given context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ],
            temperature=0.9
        )
        answers.append(response.choices[0].message.content)
    return answers

CONSISTENCY_PROMPT = """You are comparing 3 independently generated answers to the same question, based on the same source context.

Answer 1: {a1}

Answer 2: {a2}

Answer 3: {a3}

List any specific facts (names, numbers, dates) that appear in only 1 or 2 of the answers, but not all 3. These inconsistent facts are likely unreliable.

If all 3 answers agree on all facts, respond with exactly: CONSISTENT

Otherwise, list the inconsistent facts, one per line."""

def check_consistency(judge_fn, a1, a2, a3):
    prompt = CONSISTENCY_PROMPT.format(a1=a1, a2=a2, a3=a3)
    return judge_fn(prompt)