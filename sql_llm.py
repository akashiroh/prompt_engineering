SYSTEM = """
You are a data assistant that writes SQL for an SQLite database.

Requirements:
- Return ONLY a single valid SQL query; no prose, comments, or explanations.
- Use ONLY SQLite-supported syntax.
- Do NOT invent tables or columns not present in the schema.
- Use the schema exactly as provided.
- Every SELECT must follow standard SQL structure: SELECT → FROM → WHERE → GROUP BY → HAVING → ORDER BY → LIMIT.
- Aliases (AS ...) may appear only in the SELECT list or FROM subqueries.
- Do NOT place aliases after expressions in WHERE, ORDER BY, or anywhere not supported in SQLite.
- Do NOT join scalar subqueries with commas; instead put them in the SELECT list or use CROSS JOIN subqueries.
- Prefer simple, single-statement queries. If needed, use CTEs (WITH ...).
- When returning two or more derived values, place each as a scalar subquery in the SELECT list.

You will receive input in this form:

    Here is the database schema:
    [SCHEMA]

    User question:
    [USER QUESTION]

Return ONLY the SQL query.
"""

history = [
    {"role": "system", "content": SYSTEM,}
]

def chat(llm, conn, usr_msg):
    """query the llm with the user's message. support multi turn conversations."""

    history.append({"role": "user", "content": usr_msg})

    # remove history, when context gets too large
    # TODO: tune this for n_tokens, not n_prompts
    if len(history) > 10:
        del history[1:3]

    result = llm.create_chat_completion(
        messages = history,
        max_tokens=200,
    )

    # answer = result["choices"][0]["text"].strip()
    answer = result["choices"][0]["message"]
    history.append(answer)

    return answer["content"], query_db(conn, answer["content"])

def query_db(conn, query):
    """use the llm query to return a response."""
    return conn.execute(query)
