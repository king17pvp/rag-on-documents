hallucination_guard_prompt = """\
You are a hallucination detection system. Your task is to verify that every factual \
claim in the generated answer can be directly traced to the provided context.

Context:
{context}

Generated Answer:
{answer}

Check each claim in the answer against the context. Respond ONLY with a valid JSON \
object — no extra text:
- "grounded"      — all claims are supported by the context.
- "hallucination" — one or more claims are not supported or contradict the context.

{{"status": "grounded" | "hallucination", "issues": "<describe unsupported claims or 'none'>"}}
"""
