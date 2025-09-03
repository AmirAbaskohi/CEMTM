import openai
import time
import os

# Set your API key securely (not hardcoded here)
# e.g., export OPENAI_API_KEY=your_key
openai.api_key = os.getenv("OPENAI_API_KEY")

def rate_topic_with_gpt(topic_words: list[str], model="gpt-4") -> float:
    """
    Sends a topic word list to OpenAI GPT and gets a coherence score between 1–3.
    Returns a float (e.g., 2.5).
    """

    prompt = (
        "Rate the coherence of the following list of topic words on a scale from 1 (incoherent) "
        "to 3 (very coherent). Just return the number.\n\n"
        f"{', '.join(topic_words)}"
    )

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.2,
    )
    content = response["choices"][0]["message"]["content"]
    return float(content.strip())

