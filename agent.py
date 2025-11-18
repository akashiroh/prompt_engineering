from llama_cpp import Llama
import sqlite3
import pandas as pd

from sql_llm import chat as sql_llm_chat

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


        sql_query, sql_response = sql_llm_chat(
            llm,
            conn,
            f"Here is the database schema: \n {schema} \n Here is the user message: " + usr_msg,
        )

        print("\n[ASSISTANT]:", sql_query)
        for row in sql_response:
            print(row)

        print()



if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    main()
