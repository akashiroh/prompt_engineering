from llama_cpp import Llama
from pathlib import Path
import pandas as pd
import argparse
import sqlite3
import yaml
import time

from llm import LLMAgent

def initialize_model(
    model_dir: str,
    model_name: str,
    n_ctx: int=4096,
    cache_dir: Path=None,
):
    """return a pretrained llm from huggingface."""

    llm = Llama.from_pretrained(
        repo_id=model_dir,
        filename=model_name,
        cache_dir=cache_dir,
        n_ctx=n_ctx,
        n_gpu_layers=-1,
        verbose=False,
    )
    print(f"Downloaded model [{model_name}] from huggingface hub [{model_dir}]")

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

    examples = ""
    for name in schemas.keys():
        response = conn.execute("SELECT * FROM business LIMIT 3;")
        examples += f"TABLE {name} \n"
        examples += f"Examples: \n {'\n'.join(map(str, response))} \n"

    return conn, schemas, schema_str, examples

def query_db(conn, query):
    """use the llm query to return a response."""
    return conn.execute(query)


def main(args):
    """conversation loop."""

    # load database(s)
    conn, schemas, schema_str, examples = load_db(
        db_name=args.database,
    )
    vprint(f"Connecting to SQLITE database: [{args.database.name}]", args.verbose)

    llm = initialize_model(
        model_dir=args.model_dir,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
    )

    agents = {}
    with open("preconditions.yaml") as f:
        preconditions = yaml.safe_load(f)
        agents["sql-llm"] = LLMAgent(llm, preconditions["sql-llm"])
        agents["toolkit-llm"] = LLMAgent(llm, preconditions["toolkit-llm"], with_history=True)
        agents["assistant-llm"] = LLMAgent(llm, preconditions["assistant-llm"], with_history=True)

    print("Query the LLM.")
    usr_msg = None

    while usr_msg != " ":
        usr_msg = input()

        sql_required = agents["toolkit-llm"].chat(
            f"Here are the database schemas: \n [{schema_str}] \n Here is the user message: " + usr_msg,
        )
        vprint(f"\n[TOOKIT]: \n{sql_required}", args.verbose)

        if '"needs_sql": true' in sql_required:
            vprint(f"SQL required to answer: {usr_msg}", args.verbose)

            sql_query = agents["sql-llm"].chat(
                f"Database Schemas\n{schema_str}\n\nExamples\n{examples}\n\nSQL JSON\n{sql_required}\n\nUser Message\n{usr_msg}"
            )
            vprint(f"\n[SQL]: \n{sql_query}", args.verbose)

            start_time = time.time()
            sql_response = query_db(conn, sql_query)
            sql_response_str = "\n".join(
                [" ".join(map(str, row)) for row in sql_response]
            )
            end_time = time.time()
            vprint(f"\n[DB RESULTS]: \n{sql_response_str}", args.verbose)
            vprint(f"SQLITE query took {end_time - start_time:.2f} seconds.", args.verbose)

            natural_answer = agents["assistant-llm"].chat(
                f"User Question\n{usr_msg}\n\nQuery Output\n{sql_response_str}"
            )
            print(f"\n[ASSISTANT]: {natural_answer}")

        else:
            vprint(f"SQL NOT required to answer: {usr_msg}", args.verbose)
            natural_answer = agents["assistant-llm"].chat(
                f"User Question\n{usr_msg}\nSQL was not required.\n"
            )
            print(f"\n[ASSISTANT]: {natural_answer}")

        print()


def parse_args():
    """parse dem args."""
    parser = argparse.ArgumentParser(description="Create a SQLite DB from a CSV or JSON.")

    parser.add_argument(
        "-m", "--model-dir", type=str,
        default="Qwen/Qwen2-7B-Instruct-GGUF",
        help="HuggingFace Model Dir",
    )
    parser.add_argument(
        "-n", "--model-name", type=str,
        default="qwen2-7b-instruct-q4_k_m.gguf",
        help="HuggingFace Model Name",
    )
    parser.add_argument(
        "-d", "--database", type=Path,
        default="/research/hutchinson/workspace/holmesa8/data/sqlite/yelp.sqlite",
        help="/path/to/*sqlite",
    )
    parser.add_argument(
        "-c", "--cache_dir", type=Path,
        default="/research/hutchinson/workspace/holmesa8/data/.cache/huggingface",
        help="/path/to/.cache/huggingface",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="verbose mode: print intermediate queries and steps.",
    )

    args = parser.parse_args()
    return args


def vprint(msg, verbose=False):
    if verbose:
        print(msg)


if __name__ == "__main__":
    args = parse_args()
    main(args)
