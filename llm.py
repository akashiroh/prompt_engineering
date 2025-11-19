SYSTEM = """
You are a data assistant that writes SQL for an SQLite database.

Requirements:
- Return ONLY a single valid SQL query; no prose, comments, or explanations.
- Use ONLY SQLite-supported syntax.
- Do NOT invent tables or columns not present in the schema.
- Use the schema exactly as provided.
- Reference the examples for more information
- Every SELECT must follow standard SQL structure: SELECT -> FROM -> WHERE -> GROUP BY -> HAVING -> ORDER BY -> LIMIT.
- Aliases (AS ...) may appear only in the SELECT list or FROM subqueries.
- Do NOT place aliases after expressions in WHERE, ORDER BY, or anywhere not supported in SQLite.
- Do NOT join scalar subqueries with commas; instead put them in the SELECT list or use CROSS JOIN subqueries.
- Prefer simple, single-statement queries. If needed, use CTEs (WITH ...).
- When returning two or more derived values, place each as a scalar subquery in the SELECT list.

You will receive input in this form:

    Here is the database schema:
    [SCHEMA]

    [EXAMPLE ROWS IN DATABASE]

    User question:
    [USER QUESTION]

Return ONLY the SQL query.
"""


class LLMAgent():
    """LLM Agent Class that will store an LLM and a precondition"""
    def __init__(
        self,
        llm,
        precondition: str,
        with_history: bool=False,
    ):
        self.llm = llm
        self.precondition = precondition
        self.with_history = with_history
        if with_history:
            self.history = []


    def chat(self, usr_msg, max_tokens: int=200):
        """query the llm with the user's message."""

        if self.with_history:
            prompt = [{"role": "system", "content": self.precondition,}]
            prompt |= self.history
            prompt |= [{"role": "user", "content": usr_msg,}]
        else:
            prompt = [
                {"role": "system", "content": self.precondition,},
                {"role": "user", "content": usr_msg,}
            ]

        result = self.llm.create_chat_completion(
            messages=prompt,
            max_tokens=max_tokens,
        )

        answer = result["choices"][0]["message"]
        if self.with_history:
            history.append(answer)

        return answer["content"]
