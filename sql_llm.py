from llama_cpp import Llama
from pathlib import Path
import sqlite3
import pandas as pd

def initialize_model(
    model_name: str="Qwen/Qwen2-7B-Instruct-GGUF",
):
    """return a pretrained llm from huggingface."""
    llm = Llama.from_pretrained(
        repo_id=model_name,
        filename="*q4_k_m.gguf",
        n_ctx=2048,
        n_gpu_layers=-1,
        n_threads=4,
        verbose=False,
    )
    return llm


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

def chat(llm, usr_msg):
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

    return answer["content"]


def load_db(db_name, table_name: str="yelp"):
    """load sqlite db and return connection and schema."""
    conn = sqlite3.connect(db_name)

    schema = ""
    pragma = pd.read_sql_query(f"PRAGMA table_info({table_name});", conn)
    schema = f"CREATE TABLE {table_name} (\n"
    for _, r in pragma.iterrows():
        schema += f"  {r['name']} {r['type']},\n"
    schema = schema.rstrip(",\n") + "\n);"

    return conn, schema

def query_db(conn, query):
    """use the llm query to return a response."""
    return conn.execute(query)


def main():
    """conversation loop."""

    conn, schema = load_db(
        db_name="sqlite_db/yelp_business.sqlite",
        table_name="yelp",
    )

    llm = initialize_model()

    print("Query the LLM.")
    usr_msg = None

    while usr_msg != " ":
        usr_msg = input()
        if usr_msg == "DEBUG":
            import ipdb
            ipdb.set_trace()
            continue
        response = chat(
            llm,
            f"Here is the database schema: \n {schema} \n Here is the user message: " + usr_msg,
        )
        print("\n[ASSISTANT]:", response)
        response = query_db(conn, response)
        for row in response:
            print(row)
        print()



if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    main()
