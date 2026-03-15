context_relevance_filter_prompt = """\
You are a context relevance evaluator. Your task is to assess how well the retrieved \
document chunks can support answering the given question.

Question:
{question}

Retrieved Context:
{context}

Evaluate the context and respond ONLY with a valid JSON object — no extra text:
- "strong"     — the context directly and fully answers the question.
- "weak"       — the context is partially relevant but may not fully answer the question.
- "no_context" — the context is irrelevant or completely insufficient.

{{"relevance": "strong" | "weak" | "no_context", "reasoning": "<one sentence>"}}
"""
