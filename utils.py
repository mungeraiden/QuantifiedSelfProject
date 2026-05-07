# Aiden Munger
# DJ Desmet's influence on my Spotify listening habits
# 4/23/24
# https://pandas.pydata.org/docs/user_guide/index.html#user-guide

import json
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# DATA LOADING + CLEANING

def load_spotify_data(files):
    """Load and combine Spotify streaming history JSON files."""
    dfs = []

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            dfs.append(pd.DataFrame(json.load(f)))

    df = pd.concat(dfs, ignore_index=True)
    return clean_spotify_data(df)


def clean_spotify_data(df):
    """Clean Spotify dataframe and create helper columns."""
    df = df.copy()

    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
    df["minutes_played"] = df["ms_played"] / 60000

    return df

# DAILY AGGREGATION

def prepare_daily_listening(df, break_start=None, break_end=None):
    df = df.copy()

    df["ts"] = pd.to_datetime(df["ts"])

    daily = (
        df.set_index("ts")
          .resample("D")["minutes_played"]
          .sum()
    )

    daily = daily[daily > 0]
    daily = daily[daily.index.dayofweek < 5]

    if break_start and break_end:
        break_start = pd.to_datetime(break_start)
        break_end = pd.to_datetime(break_end)

        daily = daily[(daily.index < break_start) | (daily.index > break_end)]

    return daily


def get_daily_listening(df):
    """Create daily listening totals."""
    return (
        df.set_index("ts")
          .resample("D")["minutes_played"]
          .sum()
    )

# SPLITTING DATA

def split_series_before_after(series, split_date):
    """Split a time series into before/after groups."""
    split_date = pd.to_datetime(split_date)

    before = series[series.index < split_date]
    after = series[series.index >= split_date]

    return before, after


def split_df_before_after(df, split_date):
    """Split dataframe into before and after groups."""
    split_date = pd.to_datetime(split_date)

    before = df[df["ts"] < split_date]
    after = df[df["ts"] >= split_date]

    return before, after


# ARTIST ANALYSIS

def get_top_artists(df, n=10):
    """Return top artists by play count."""
    return (
        df["master_metadata_album_artist_name"]
        .value_counts()
        .head(n)
    )

def get_new_artists(before_df, after_df, n=10):
    """Find artists discovered after the split date."""
    before_artists = set(
        before_df["master_metadata_album_artist_name"].dropna()
    )

    after_artists = set(
        after_df["master_metadata_album_artist_name"].dropna()
    )

    new_artists = after_artists - before_artists

    new_artist_df = after_df[
        after_df["master_metadata_album_artist_name"].isin(new_artists)
    ]

    return (
        new_artist_df
        .groupby("master_metadata_album_artist_name")["minutes_played"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
    )

# STATISTICS

def run_ttest(before, after):
    """Run Welch's t-test."""
    t_stat, p_val = ttest_ind(before, after, equal_var=False)

    return {
        "before_mean": before.mean(),
        "after_mean": after.mean(),
        "t_stat": t_stat,
        "p_value": p_val,
    }

# CALENDAR MERGING

def merge_calendar(academic_daily, calendar_csv, dj_date):
    """Merge listening data with academic calendar."""
    cal = pd.read_csv(calendar_csv)

    cal["date"] = pd.to_datetime(cal["date"])
    academic_daily["date"] = pd.to_datetime(academic_daily["date"])

    model_df = academic_daily.merge(cal, on="date", how="left")

    dj_date = pd.to_datetime(dj_date)

    model_df["after_dj"] = (model_df["date"] >= dj_date).astype(int)

    return model_df


# PLOTTING

def plot_daily_listening(daily, split_date=None):
    plt.figure(figsize=(10, 5))
    daily.plot()

    if split_date:
        plt.axvline(pd.to_datetime(split_date), linestyle="--")

    plt.title("Listening Over Time")
    plt.ylabel("Minutes per Day")

    plt.show()
    plt.close()


def plot_before_after(before_mean, after_mean):
    plt.figure(figsize=(6, 4))

    plt.bar(["Before DJ", "After DJ"], [before_mean, after_mean])

    plt.ylabel("Average Minutes per Day")
    plt.title("Average Listening Comparison")

    plt.show()
    plt.close()


def plot_top_artists(series, title):
    plt.figure(figsize=(10, 5))

    series.plot(kind="bar")

    plt.title(title)
    plt.ylabel("Play Count")

    plt.show()
    plt.close()

# MACHINE LEARNING

def prepare_features(model_df):
    model_df = model_df.copy()

    median_minutes = model_df["minutes_played"].median()

    model_df["high_listening"] = (
        model_df["minutes_played"] > median_minutes
    ).astype(int)

    model_df["date"] = pd.to_datetime(model_df["date"])
    model_df["dayofweek"] = model_df["date"].dt.dayofweek
    model_df["month"] = model_df["date"].dt.month

    X = model_df[["dayofweek", "month", "after_dj"]]
    y = model_df["high_listening"]

    return X, y


def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled


def train_knn(X_train_scaled, y_train, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaled, y_train)
    return knn


def train_decision_tree(X_train, y_train, max_depth=4):
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42,
    )

    dt.fit(X_train, y_train)
    return dt


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)