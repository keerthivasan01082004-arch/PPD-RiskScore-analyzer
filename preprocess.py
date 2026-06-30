import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import (
    DATA_PATH, FEATURE_COLS, TARGET_COL, DROP_COLS,
    ENCODING_MAP, TEST_SIZE, RANDOM_STATE
)

def load_raw():
    """Load dataset from configured path."""
    df = pd.read_csv(DATA_PATH)
    print("✅ Loaded dataset:", df.shape)
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop irrelevant columns, normalize text, and handle missing values."""
    # Drop irrelevant columns
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=col)

    # Normalize text columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.title()

    # Handle missing values
    for col in df.columns:
        if col == TARGET_COL:
            df[col] = df[col].fillna("Unknown")
        elif df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].mean())

    print("✅ After cleaning:", df.shape)
    return df

def encode_survey_values(df: pd.DataFrame) -> pd.DataFrame:
    """Encode survey responses and filter valid labels."""
    # Encode feature columns
    for col in FEATURE_COLS:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].map(ENCODING_MAP).fillna(0).astype(int)
        else:
            print(f"⚠️ Warning: Feature column '{col}' not found in dataset")

    # Normalize target labels
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.title()

    # Keep only valid labels
    valid_labels = {"Yes", "No", "Not Interested To Say"}
    df = df[df[TARGET_COL].isin(valid_labels)]

    print("✅ After encoding:", df.shape, "Labels:", df[TARGET_COL].unique())
    return df

def split(df: pd.DataFrame):
    """Split dataset into train/test sets."""
    if df.empty:
        raise ValueError("❌ Dataset is empty after preprocessing. Check column names and labels.")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("✅ Train/Test split:", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test

def preprocess_pipeline():
    """Full preprocessing pipeline."""
    df = load_raw()
    df = clean(df)
    df = encode_survey_values(df)
    X_train, X_test, y_train, y_test = split(df)
    print("✅ Final preprocessing complete.")
    return X_train, X_test, y_train, y_test
