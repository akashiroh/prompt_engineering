from llama_cpp import Llama
import sqlite3
import pandas as pd
import yaml

from llms.llm import LLMAgent

def initialize_model(
    model_name: str="Qwen/Qwen2-7B-Instruct-GGUF",
):
    """return a pretrained llm from huggingface."""

    try:
        path = hf_hub_download(
            repo_id=model_name,
            filename="qwen2-7b-instruct-q4_k_m.gguf",
        )
        llm = Llama(model_path=path)
        print(f"Loaded model [{model_name}] from cache")
    except:
        llm = Llama.from_pretrained(
            repo_id=model_name,
            filename="*q4_k_m.gguf",
            n_ctx=2048,
            n_gpu_layers=-1,
            verbose=False,
        )
        print(f"Downloaded model [{model_name}] from huggingface hub")
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

def query_db(conn, query):
    """use the llm query to return a response."""
    return conn.execute(query)


def main():
    """conversation loop."""
    llm = initialize_model()

    agents = {}
    with open("preconditions.yaml") as f:
        preconditions = yaml.safe_load(f)
        agents["sql-llm"] = LLMAgent(llm, preconditions["sql-llm"])

    # load database
    conn, schema = load_db(
        db_name="sqlite_db/yelp_business.sqlite",
        table_name="yelp",
    )

    # samples from database
    db_samples = conn.execute(
        "SELECT * from yelp WHERE categories LIKE '%restaurant%' LIMIT 2"
    )
    examples = "\n".join([
        " ".join(map(str, sample)) for sample in db_samples
    ])

    print("Query the LLM.")
    usr_msg = None

    while usr_msg != " ":
        usr_msg = input()

        sql_query = agents["sql-llm"].chat(
            f"Here is the database schema: \n [{schema}] [{examples}] \n Here is the user message: " + usr_msg,
        )
        sql_response = query_db(conn, sql_query)

        print("\n[ASSISTANT]:", sql_query)
        for row in sql_response:
            print(row)

        print()



if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    main()
