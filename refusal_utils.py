import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

REFUSAL_CHECK_PROMPT = """Is the following sentence a refusal or disclaimer (the model saying it doesn't know, can't confirm, lacks access to information, or suggesting the user check elsewhere), or is it a factual claim/statement?

Sentence: {claim}

Respond with exactly one word: REFUSAL or CLAIM."""

def is_refusal(claim):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": REFUSAL_CHECK_PROMPT.format(claim=claim)}],
            temperature=0
        )
        verdict = response.choices[0].message.content.strip().upper()
        return "REFUSAL" in verdict
    except Exception:
        return False