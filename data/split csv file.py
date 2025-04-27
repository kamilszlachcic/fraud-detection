import pandas as pd
import os
from pathlib import Path


def split_transaction_and_identity(transaction_path, identity_path, output_dir, chunk_size=10000):
    os.makedirs(output_dir, exist_ok=True)

    # load CSV
    df_trans = pd.read_csv(transaction_path)
    df_id = pd.read_csv(identity_path)

    # Sortowanie opcjonalne, np. po TransactionID
    df_trans = df_trans.sort_values("TransactionID").reset_index(drop=True)
    df_id = df_id.set_index("TransactionID")

    num_chunks = (len(df_trans) + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df_trans))

        df_chunk_trans = df_trans.iloc[start_idx:end_idx]
        chunk_ids = df_chunk_trans["TransactionID"].values

        # Wybieramy pasujące ID z identity — może być pusty
        df_chunk_id = df_id.loc[df_id.index.intersection(chunk_ids)].reset_index()

        df_chunk_trans.to_csv(f"{output_dir}/part_{i}_test_transaction.csv", index=False)
        df_chunk_id.to_csv(f"{output_dir}/part_{i}_test_identity.csv", index=False)

        print(f"Saved chunk {i}: {len(df_chunk_trans)} transactions, {len(df_chunk_id)} identities")


# paths
transaction_path = Path("raw") / "test_transaction.csv"
identity_path = Path("raw") / "test_identity.csv"
output_path = Path("raw") / "split"


split_transaction_and_identity(transaction_path, identity_path, output_path, chunk_size=50000)
