

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")


def load():
    df = pd.read_csv("data/movies_clean.csv")
    df["year"]     = pd.to_numeric(df["year"],     errors="coerce")
    df["rating"]   = pd.to_numeric(df["rating"],   errors="coerce")
    df["runtime"]  = pd.to_numeric(df["runtime"],  errors="coerce")
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    print(f"Loaded {len(df)} movies for EDA.")
    return df


def plot_rating_distribution(df):
    ratings = df["rating"].dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(ratings, bins=30, kde=True, color="#4C72B0", ax=ax)
    ax.axvline(ratings.mean(),   color="red",    ls="--", lw=1.5,
               label=f"Mean: {ratings.mean():.2f}")
    ax.axvline(ratings.median(), color="orange", ls=":",  lw=1.5,
               label=f"Median: {ratings.median():.2f}")
    ax.set_title("Distribution of Movie Ratings", fontsize=14, fontweight="bold")
    ax.set_xlabel("Rating (out of 10)")
    ax.set_ylabel("Number of Movies")
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/1_rating_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved: outputs/1_rating_distribution.png")


def plot_top_genres(df):
    all_genres = []
    for val in df["genre"].dropna():
        all_genres.extend([g.strip() for g in str(val).split(",")])
    counts = Counter(all_genres)
    counts.pop("Unknown", None)
    top10 = pd.DataFrame(counts.most_common(10), columns=["Genre", "Count"])

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top10, x="Count", y="Genre", palette="viridis", ax=ax)
    ax.set_title("Top 10 Most Common Genres", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Movies")
    for bar in ax.patches:
        ax.text(bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width())}",
                va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig("outputs/2_top_genres.png", dpi=150)
    plt.close(fig)
    print("Saved: outputs/2_top_genres.png")


def plot_rating_by_year(df):
    yearly = (df.dropna(subset=["year", "rating"])
                .groupby("year")["rating"]
                .agg(["mean", "count"])
                .reset_index())
    yearly = yearly[
        (yearly["count"] >= 3) &
        (yearly["year"]  >= 1980) &
        (yearly["year"]  <= 2024)
    ]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(yearly["year"], yearly["mean"], color="#DD8452", lw=2)
    ax.fill_between(yearly["year"], yearly["mean"], alpha=0.15, color="#DD8452")
    ax.set_title("Average Movie Rating by Release Year",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Rating")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    fig.tight_layout()
    fig.savefig("outputs/3_rating_by_year.png", dpi=150)
    plt.close(fig)
    print("Saved: outputs/3_rating_by_year.png")


def plot_top_directors(df):
    dirs = (df["director"]
              .dropna()
              .loc[df["director"] != "Unknown"]
              .value_counts()
              .head(10)
              .reset_index())
    dirs.columns = ["Director", "Count"]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=dirs, x="Count", y="Director", palette="muted", ax=ax)
    ax.set_title("Top 10 Directors by Movie Count",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Movies")
    for bar in ax.patches:
        ax.text(bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width())}",
                va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig("outputs/4_top_directors.png", dpi=150)
    plt.close(fig)
    print("Saved: outputs/4_top_directors.png")


def plot_runtime_distribution(df):
    rt = df["runtime"].dropna()
    rt = rt[(rt > 30) & (rt < 300)]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(rt, bins=40, kde=True, color="#55A868", ax=ax)
    ax.axvline(rt.mean(), color="red", ls="--", lw=1.5,
               label=f"Mean: {rt.mean():.0f} min")
    ax.set_title("Movie Runtime Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Runtime (minutes)")
    ax.set_ylabel("Number of Movies")
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/5_runtime_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved: outputs/5_runtime_distribution.png")


def run_eda():
    df = load()
    print("\n--- Summary Statistics ---")
    print(df[["rating", "year", "runtime", "vote_count"]].describe().round(2))

    plot_rating_distribution(df)
    plot_top_genres(df)
    plot_rating_by_year(df)
    plot_top_directors(df)
    plot_runtime_distribution(df)
    print("\nEDA complete. All plots saved to outputs/")
    return df


if __name__ == "__main__":
    run_eda()
