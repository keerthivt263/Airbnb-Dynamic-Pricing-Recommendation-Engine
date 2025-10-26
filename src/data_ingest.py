# src/data_cleaning.py
import pandas as pd
import numpy as np
import os

# Input and output file paths
IN = os.path.join("C:/Users/DELL/Downloads/project/Elevate labs/airbnb_pricing/data/airbnb_raw_cleaned.csv")
OUT = os.path.join("C:/Users/DELL/Downloads/project/Elevate labs/airbnb_pricing/data/airbnb_processed.csv")

def clean():
    # Load dataset
    df = pd.read_csv(IN, low_memory=False)

    # ----- PRICE -----
    if "log_price" in df.columns:
        df["price"] = np.exp(df["log_price"])
    elif "price" in df.columns:
        df["price"] = df["price"].replace('[\$,]', '', regex=True).astype(float)

    # ----- NUMERIC CONVERSIONS -----
    for col in ["bathrooms", "beds", "bedrooms", "accommodates", "number_of_reviews"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ----- IMPUTE MISSING VALUES -----
    num_cols = ["bathrooms", "beds", "bedrooms"]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    # ----- CATEGORICAL CLEANING -----
    if "property_type" in df.columns:
        df["property_type"] = df["property_type"].fillna("Unknown").str.strip()

    # ----- AMENITIES COUNT -----
    if "amenities" in df.columns:
        df["amenities_count"] = df["amenities"].fillna("").apply(
            lambda x: len(x.split(",")) if x else 0
        )

    # ----- HOST FEATURES -----
    for host_col in ["host_since", "host_has_profile_pic", "host_identity_verified", "host_response_rate"]:
        if host_col in df.columns:
            df[host_col] = df[host_col].fillna("Unknown")

    # ----- DATE CONVERSION -----
    if "host_since" in df.columns:
        df["host_since"] = pd.to_datetime(df["host_since"], errors="coerce")
        df["host_age_days"] = (pd.Timestamp.now() - df["host_since"]).dt.days.fillna(0)

    # ----- OUTLIER REMOVAL -----
    df = df[(df["price"] > 10) & (df["price"] < df["price"].quantile(0.99))].copy()

    # Save processed data
    df.to_csv(OUT, index=False)
    print(f"✅ Saved processed data to {OUT}. New shape: {df.shape}")

if __name__ == "__main__":
    clean()