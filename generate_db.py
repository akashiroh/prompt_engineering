import pandas as pd
from pathlib import Path
import argparse
import sqlite3

import json

def generate_db(args):
    """save a .sqlite db from a csv or json"""
    suffix = args.data_path.suffix

    if suffix == ".csv":
        df = pd.read_csv(args.data_path)
    elif suffix == ".json":
        df = pd.read_json(args.data_path, lines=True, dtype=False)
        obj_cols = df.select_dtypes(include=['object']).columns
        df[obj_cols] = df[obj_cols].astype('string')
    else:
        raise NotImplementedError(f"{suffix} data types are not implemented yet. sry")

    assert args.save_path.suffix == ".sqlite", f"Save path should be /path/to/*.sqlite"

    conn = sqlite3.connect(args.save_path)
    df.to_sql(args.name, conn, if_exists="replace", index=False)

    data_short_path = Path(*args.data_path.parts[-3:])
    save_short_path = Path(*args.save_path.parts[-3:])
    print(f"Saved {data_short_path} to {save_short_path} as {args.name}")


def parse_args():
    """parse dem args."""
    parser = argparse.ArgumentParser(description="Create a SQLite DB from a CSV or JSON.")

    parser.add_argument("-d", "--data-path", type=Path, help="/path/to/*.{csv, json}")
    parser.add_argument("-o", "--save-path", type=Path, help="/path/to/save/*.sqlite")
    parser.add_argument("-n", "--name", type=str , help="name of table in database")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generate_db(args)
