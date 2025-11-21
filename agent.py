from llama_cpp import Llama
from pathlib import Path
import pandas as pd
import argparse
import sqlite3
import yaml
import time

from llm import LLMAgent

def initialize_model(
    model_name: str="Qwen/Qwen2-7B-Instruct-GGUF",
    n_ctx: int=2048,
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
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            verbose=False,
        )
        print(f"Downloaded model [{model_name}] from huggingface hub")
    return llm

def load_db(db_name):
    """load sqlite db and return connection and schema."""
    conn = sqlite3.connect(db_name)

    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    schemas = {k[0]: "" for k in tables}

    for table_name in schemas.keys():
        pragma = pd.read_sql_query(f"PRAGMA table_info({table_name});", conn)
        schema = f"CREATE TABLE {table_name} (\n"
        for _, r in pragma.iterrows():
            schema += f"  {r['name']} {r['type']},\n"
        schema = schema.rstrip(",\n") + "\n);"

        schemas[table_name] = schema

    schema_str =  "\n\n".join(schemas.values())
    return conn, schemas, schema_str

def query_db(conn, query):
    """use the llm query to return a response."""
    return conn.execute(query)


def main(args):
    """conversation loop."""

    # load database(s)
    conn, schemas, schema_str = load_db(
        db_name=args.database,
    )

    examples = ""
    for name in schemas.keys():
        response = conn.execute("SELECT * FROM business LIMIT 3;")
        examples += f"TABLE {name} \n"
        examples += f"Examples: \n {'\n'.join(map(str, response))} \n"

    llm = initialize_model(args.model)

    agents = {}
    with open("preconditions.yaml") as f:
        preconditions = yaml.safe_load(f)
        agents["sql-llm"] = LLMAgent(llm, preconditions["sql-llm"])

    print("Query the LLM.")
    usr_msg = None

    while usr_msg != " ":
        usr_msg = input()

        sql_query = agents["sql-llm"].chat(
            f"Here are the database schemas: \n [{schema_str}] \n Here is the user message: " + usr_msg,
        )
        print("\n[ASSISTANT]:", sql_query)

        start_time = time.time()
        sql_response = query_db(conn, sql_query)
        end_time = time.time()

        response = " | ".join([r[0] for r in sql_response])
        print(response)
        print(f"Query took {end_time - start_time:.3f} seconds")

        print()


def parse_args():
    """parse dem args."""
    parser = argparse.ArgumentParser(description="Create a SQLite DB from a CSV or JSON.")

    parser.add_argument(
        "-m", "--model", type=str,
        default="Qwen/Qwen2-7B-Instruct-GGUF",
        help="HuggingFace Model",
    )
    parser.add_argument(
        "-d", "--database", type=Path,
        default="/research/hutchinson/workspace/holmesa8/data/sqlite/yelp.sqlite",
        help="/path/to/*sqlite",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
