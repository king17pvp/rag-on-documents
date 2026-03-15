conversation_summary_prompt = """\
You are a conversation summarizer. Update the running summary with the latest exchange \
so that it remains a concise, accurate record of the entire conversation.

Current Summary:
{current_summary}

New Exchange:
User: {question}
Assistant: {answer}

Write only the updated summary — no preamble, no commentary:
"""
