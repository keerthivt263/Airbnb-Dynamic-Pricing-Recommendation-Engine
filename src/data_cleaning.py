# src/data_cleaning.py
import pandas as pd
import numpy as np
import os
import warnings

# === CONFIG ===
IN = os.path.join("C:/Users/DELL/Downloads/project/Elevate labs/airbnb_pricing/data/airbnb_raw_cleaned.csv")
OUT = os.path.join("C:/Users/DELL/Downloads/project/Elevate labs/airbnb_pricing/data/airbnb_processed.csv")

def _to_numeric_safe(series):
    """Coerce numeric-like strings to floats; keep NaN if not convertible."""
    return pd.to_numeric(series.astype(str).str.replace(",", "").str.replace("%", ""), errors="coerce")

def clean():
    # Load
    df = pd.read_csv(IN, low_memory=False)
    initial_shape = df.shape
    print(f"Loaded {IN}. Shape: {initial_shape}")

    # --- PRICE ---
    if "log_price" in df.columns and df["log_price"].notna().sum() > 0:
        df["price"] = np.exp(df["log_price"])
    elif "price" in df.columns:
        # remove currency symbols and commas then convert
        df["price"] = df["price"].astype(str).str.replace(r'[\$,]', '', regex=True)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # --- NUMERIC CONVERSIONS ---
    numeric_like = ["bathrooms", "beds", "bedrooms", "accommodates", "number_of_reviews"]
    for col in numeric_like:
        if col in df.columns:
            df[col] = _to_numeric_safe(df[col])

    # host_response_rate might be like "95%"
    if "host_response_rate" in df.columns:
        df["host_response_rate"] = _to_numeric_safe(df["host_response_rate"])

    # --- AMENITIES COUNT ---
    if "amenities" in df.columns:
        df["amenities_count"] = df["amenities"].fillna("").apply(lambda x: len([a for a in str(x).split(",") if a.strip()]) if x else 0)

    # --- CATEGORICAL CLEANING ---
    if "property_type" in df.columns:
        df["property_type"] = df["property_type"].fillna("Unknown").astype(str).str.strip()

    for host_col in ["host_since", "host_has_profile_pic", "host_identity_verified"]:
        if host_col in df.columns:
            df[host_col] = df[host_col].fillna("Unknown")

    # --- DATE CONVERSION / HOST AGE ---
    if "host_since" in df.columns:
        df["host_since"] = pd.to_datetime(df["host_since"], errors="coerce")
        df["host_age_days"] = (pd.Timestamp.now() - df["host_since"]).dt.days
        # replace negative/infinite with 0 and fillna
        df["host_age_days"] = df["host_age_days"].apply(lambda x: x if pd.notna(x) and x >= 0 else 0).fillna(0)

    # --- DUPLICATE REMOVAL ---
    # 1) Exact duplicate rows
    dup_rows = df.duplicated(keep="first").sum()
    if dup_rows > 0:
        print(f"Removing {dup_rows} exact duplicate rows.")
        df = df.drop_duplicates(keep="first")

    # 2) Duplicate by id (if id column exists) — keep first occurrence
    if "id" in df.columns:
        dup_id_count = df.duplicated(subset=["id"], keep="first").sum()
        if dup_id_count > 0:
            print(f"Removing {dup_id_count} duplicate rows based on 'id' column (keeping first).")
            df = df.drop_duplicates(subset=["id"], keep="first")

    # --- IMPUTE SMALL MISSING NUMERIC VALUES (median) ---
    num_impute_cols = ["bathrooms", "beds", "bedrooms"]
    for c in num_impute_cols:
        if c in df.columns:
            median_val = df[c].median()
            df[c] = df[c].fillna(median_val)

    # --- OUTLIER REMOVAL FOR PRICE ---
    if "price" in df.columns:
        before_price_filter = df.shape[0]
        # Keep reasonable range: price > 10 and below 99th percentile
        upper = df["price"].quantile(0.99)
        df = df[(df["price"] > 10) & (df["price"] < upper)].copy()
        removed = before_price_filter - df.shape[0]
        print(f"Price filtering removed {removed} rows (kept price between 10 and 99th pct: {upper:.2f}).")

    # --- FINAL TOUCHES ---
    df = df.reset_index(drop=True)

    final_shape = df.shape
    removed_total = initial_shape[0] - final_shape[0]
    if removed_total > 0:
        warnings.warn(f"Total rows removed during cleaning: {removed_total} (from {initial_shape} to {final_shape})", UserWarning)
    else:
        print("No rows removed during cleaning.")

    # --- SAVE ---
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"✅ Saved processed data to {OUT}. Final shape: {final_shape}")

if __name__ == "__main__":
    clean()