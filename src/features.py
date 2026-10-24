"""Feature engineering for the load-type model."""

import numpy as np

WINTER_MONTHS = [11, 12, 1, 2]


def add_time_features(df):
    """Add calendar, season and cyclical time-of-day features from 'Date' and 'NSM'.

    The load type follows a time-of-use schedule, so the useful inputs are the time
    of day, weekday vs weekend, and the season (the 'Maximum' window is much longer
    in winter). A winter flag is used instead of the raw month number because the
    test month (December) never appears in training, but the winter *season* does
    (November, January, February). 'hour' is skipped because it duplicates NSM.
    """
    df = df.copy()
    df["day_of_week"] = df["Date"].dt.day_of_week
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_winter"] = df["Date"].dt.month.isin(WINTER_MONTHS).astype(int)
    df["nsm_sin"] = np.sin(2 * np.pi * df["NSM"] / 86400)
    df["nsm_cos"] = np.cos(2 * np.pi * df["NSM"] / 86400)
    return df
