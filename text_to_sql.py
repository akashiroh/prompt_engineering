from llama_cpp import Llama
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
You are a data assistant.
You will create a SQL query for an SQLite database.

You will recieve prompts that look like the follow:

    Here is the database schema:
    [SCHEMA]

    User question:
    "[USER QUESTION]"

Return ONLY the SQL query.
"""

history = [
    {"role": "system", "content": SYSTEM,}
]

def chat(llm, usr_msg):
    """query the llm with the user's message. support multi turn conversations."""

    history.append({"role": "user", "content": usr_msg})

    result = llm.create_chat_completion(
        messages = history,
        max_tokens=200,
    )

    # answer = result["choices"][0]["text"].strip()
    answer = result["choices"][0]["message"]
    history.append(answer)

    return answer["content"]


def load_db():
    """load csv into sqlite and return (connection, schema_for_llm)."""

    df = pd.read_csv("data/whr_2023_dataset.csv")

    # clean column names
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(":", "")
    )

    conn = sqlite3.connect("mydb.sqlite")

    df.to_sql("whr2023", conn, if_exists="replace", index=False)

    # LLM-friendly schema
    schema = ""
    pragma = pd.read_sql_query("PRAGMA table_info(whr2023);", conn)
    schema = "CREATE TABLE whr2023 (\n"
    for _, r in pragma.iterrows():
        schema += f"  {r['name']} {r['type']},\n"
    schema = schema.rstrip(",\n") + "\n);"

    return conn, schema


def main():
    """conversation loop."""

    conn, schema = load_db()

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
        print()



if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    main()
