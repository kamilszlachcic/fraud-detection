import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_and_merge(transaction_file: Path, identity_file: Path) -> pd.DataFrame:
    """
        Loads and merges transaction and identity CSV files based on 'TransactionID'.

        Args:
            transaction_file (str): Relative or absolute path to the transaction CSV.
            identity_file (str): Relative or absolute path to the identity CSV.

        Returns:
            pd.DataFrame: Merged dataset with TransactionID as index.
        """
    df1 = pd.read_csv(transaction_file)
    df2 = pd.read_csv(identity_file)
    df = df1.merge(df2, how='left')
    df.set_index("TransactionID", inplace=True)
    df.columns = df.columns.str.replace("-", "_")
    return df


def drop_v_columns(df):
    # Thanks to spectacular EDA for Columns V https://www.kaggle.com/code/cdeotte/eda-for-columns-v-and-id#V-Reduced
    # 211 V-Columns will be dropped
    # Define Selected all V-Columns

    all_v_columns = [col for col in df.columns if col.startswith("V")]
    v = [1, 3, 4, 6, 8, 11]
    v += [13, 14, 17, 20, 23, 26, 27, 30]
    v += [36, 37, 40, 41, 44, 47, 48]
    v += [54, 56, 59, 62, 65, 67, 68, 70]
    v += [76, 78, 80, 82, 86, 88, 89, 91]
    v += [96, 98, 99, 104]
    v += [107, 108, 111, 115, 117, 120, 121, 123]
    v += [124, 127, 129, 130, 136]
    v += [138, 139, 142, 147, 156, 162]
    v += [165, 160, 166]
    v += [178, 176, 173, 182]
    v += [187, 203, 205, 207, 215]
    v += [169, 171, 175, 180, 185, 188, 198, 210, 209]
    v += [218, 223, 224, 226, 228, 229, 235]
    v += [240, 258, 257, 253, 252, 260, 261]
    v += [264, 266, 267, 274, 277]
    v += [220, 221, 234, 238, 250, 271]
    v += [294, 284, 285, 286, 291, 297]
    v += [303, 305, 307, 309, 310, 320]
    v += [281, 283, 289, 296, 301, 314]
    v += [332, 325, 335, 338]

    # Convert to column names
    selected_v_columns = [f"V{i}" for i in v]

    # Determine unimportant V-columns (all V-columns - selected V-columns)
    not_important_v_columns = list(set(all_v_columns) - set(selected_v_columns))

    # Drop unimportant V-columns while keeping everything else
    df = df.drop(columns=not_important_v_columns)
    print(
        f'✅ Dropped {len(not_important_v_columns)} '
        f'unimportant V-columns. Remaining columns: {len(df)}')
    return df


def detect_columns_to_scale(df: pd.DataFrame, threshold: int = 10) -> list:
    """
    Detects numeric columns that should be scaled, excluding binary and categorical-like features.

    Args:
        df (pd.DataFrame): Input dataframe.
        threshold (int): Minimum number of unique values to qualify for scaling.

    Returns:
        list: List of column names to scale.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    to_scale = [
        col for col in numeric_cols
        if df[col].nunique() > threshold and not set(df[col].unique()).issubset({0, 1})
    ]
    return to_scale


def fit_scaler_and_scale(df: pd.DataFrame, columns_to_scale: list, save: bool = True):
    """
    Fits StandardScaler and scales the selected columns. Optionally saves the scaler and columns list.

    Args:
        df (pd.DataFrame): Input dataframe.
        columns_to_scale (list): Columns to scale.
        save (bool): Whether to save the scaler and columns list.

    Returns:
        pd.DataFrame: Dataframe with scaled columns.
        scaler: The fitted scaler.
    """
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])

    if save:
        joblib.dump(scaler, PROJECT_ROOT / "src/data_processing/standard_scaler.pkl")
        joblib.dump(columns_to_scale, PROJECT_ROOT / "src/data_processing/scaled_columns.pkl")
        print("✅ Scaler and column list saved.")

    return df_scaled, scaler


def load_scaler_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loads previously saved scaler and column list, and applies scaling.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Scaled dataframe.
    """
    scaler = joblib.load(PROJECT_ROOT / "src/data_processing/standard_scaler.pkl")
    columns_to_scale = joblib.load(PROJECT_ROOT / "src/data_processing/scaled_columns.pkl")

    df_scaled = df.copy()
    df_scaled[columns_to_scale] = scaler.transform(df_scaled[columns_to_scale])

    return df_scaled


from sklearn.model_selection import train_test_split


def split_train_test(df, target_column="isFraud", test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column, "UID"], errors="ignore")
    y = df[target_column]

    return train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)
