"""Loading, cleaning and splitting the steel-plant load dataset."""

import numpy as np
import pandas as pd


def load_data(path):
    """Read the raw CSV and standardise the timestamp column name to 'Date'."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df.rename(columns={"Date_Time": "Date"})


def clean_data(df):
    """Parse dates, drop duplicate timestamps, fix corrupted NSM and invalid power factors."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y %H:%M", errors="coerce")
    df = (
        df.dropna(subset=["Date"])
        .drop_duplicates(subset="Date")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    num_cols = [c for c in df.columns if c not in ("Date", "Load_Type")]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # NSM is corrupted in the raw file: rebuild it from the timestamp
    df["NSM"] = df["Date"].dt.hour * 3600 + df["Date"].dt.minute * 60

    # A power factor must lie between 0 and 100
    for c in ["Lagging_Current_Power_Factor", "Leading_Current_Power_Factor"]:
        df.loc[(df[c] < 0) | (df[c] > 100), c] = np.nan
    return df


def time_split(df, cutoff="2018-12-01"):
    """Chronological split: everything before `cutoff` is train, the rest is test."""
    train = df[df["Date"] < cutoff]
    test = df[df["Date"] >= cutoff]
    return train, test
