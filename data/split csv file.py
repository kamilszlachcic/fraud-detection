import pandas as pd
import os
from pathlib import Path


def split_large_csv(file_path, output_dir, prefix, chunk_size=50000):
    """
    Splits a large CSV file into smaller chunks.

    Args:
        file_path (str): Path to the large CSV file.
        output_dir (str): Directory to store split files.
        prefix (str): Prefix for the output file names.
        chunk_size (int): Number of rows per chunk.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
        chunk_file = os.path.join(output_dir, f"part_{i}_{prefix}.csv")
        chunk.to_csv(chunk_file, index=False)
        print(f"Saved: {chunk_file} ({len(chunk)} rows)")



transaction_path = Path("raw") / "test_transaction.csv"
identity_path = Path("raw") / "test_identity.csv"
output_path = Path("raw") / "split"

split_large_csv(transaction_path, output_path, "test_transaction", chunk_size=10000)
split_large_csv(identity_path, output_path, "test_identity", chunk_size=10000)