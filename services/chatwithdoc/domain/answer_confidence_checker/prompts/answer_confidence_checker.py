answer_confidence_checker_prompt = """\
You are an answer quality evaluator. Assess whether the generated answer is confident, \
complete, and well-supported by the provided context.

Question: {question}

Generated Answer:
{answer}

Context Used:
{context}

Evaluate whether the answer:
1. Directly and specifically addresses the question.
2. Is fully supported by claims present in the context.
3. Avoids vague hedging or unsupported assertions.
4. Covers all key aspects of the question.

Respond ONLY with a valid JSON object — no extra text:
{{"confidence": "high" | "low", "reasoning": "<one sentence>"}}
"""
