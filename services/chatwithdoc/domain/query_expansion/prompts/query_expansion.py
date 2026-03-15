query_expansion_prompt = """\
You are a search query expansion expert. Given a user question, generate {num_queries} \
diverse search queries that will help retrieve all relevant document chunks needed to \
answer it. The queries should:
- Cover different phrasings, synonyms, and aspects of the question.
- Be concise and search-engine friendly.
- Together provide comprehensive coverage of the information need.

Question: {question}

Respond ONLY with a valid JSON object — no extra text:
{{"queries": ["<query 1>", "<query 2>", ...]}}
"""
