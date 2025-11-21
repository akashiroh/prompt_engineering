import pandas as pd
from pathlib import Path
import argparse
import sqlite3

import json

def generate_db(args):
    """save a .sqlite db from a csv or json"""

    conn = sqlite3.connect(args.save_to)

    for (file_name, table_name) in zip(args.files, args.names):
        suffix = file_name.suffix

        if suffix == ".csv":
            df = pd.read_csv(file_name)
        elif suffix == ".json":
            df = pd.read_json(file_name, lines=True, dtype=False)
            obj_cols = df.select_dtypes(include=['object']).columns
            df[obj_cols] = df[obj_cols].astype('string')
        else:
            raise NotImplementedError(f"{suffix} data types are not implemented yet. sry")

        df.to_sql(table_name, conn, if_exists="replace", index=False)

        data_short_path = Path(*file_name.parts[-3:])
        save_short_path = Path(*args.save_to.parts[-3:])
        print(f"Saved {data_short_path} to {save_short_path} as {table_name}")


def parse_args():
    """parse dem args."""
    parser = argparse.ArgumentParser(description="Create a SQLite DB from a CSV or JSON.")

    parser.add_argument("-f", "--files", nargs="+", type=Path, help="/path/to/{dir, *.{csv, json}}")
    parser.add_argument("-o", "--save_to", type=Path, help="/path/to/save/*.sqlite")
    parser.add_argument("-n", "--names", nargs="+", type=str , help="name of table in database")

    args = parser.parse_args()

    if len(args.files) != len(args.names):
        parser.error(f"""
             Arguments --names and --files must have the same number of values.
             --names: {args.names}
             --files: {args.files}
        """)
    assert args.save_to.suffix == ".sqlite", f"Save path should be /path/to/*.sqlite"

    return args

if __name__ == "__main__":
    args = parse_args()
    generate_db(args)
