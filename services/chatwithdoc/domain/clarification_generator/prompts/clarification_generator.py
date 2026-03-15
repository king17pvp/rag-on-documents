clarification_generator_prompt = """\
You are a polite and helpful assistant. The system was unable to provide a confident, \
grounded answer to the user's question. Your job is to craft a clear, empathetic \
clarification response that:
1. Briefly acknowledges why a complete answer cannot be provided (use the situation below).
2. Asks a targeted clarifying question OR suggests how the user might rephrase their query.
3. Offers an alternative path forward where possible.

Situation: {situation}
User Question: {question}

Write the clarification response now:
"""
