

import ast
import os
import pandas as pd


def safe_parse(obj, key="name", limit=3):

    try:
        items = ast.literal_eval(obj)
        return ", ".join([i[key] for i in items[:limit]])
    except Exception:
        return "Unknown"


def get_director(crew_str):
    
    try:
        crew = ast.literal_eval(crew_str)
        for member in crew:
            if member.get("job") == "Director":
                return member["name"]
    except Exception:
        pass
    return "Unknown"


def load_and_clean():
    
    movies_path  = "data/tmdb_5000_movies.csv"
    credits_path = "data/tmdb_5000_credits.csv"

    
    for path in [movies_path, credits_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\nFile not found: {path}\n"
                "Please download the TMDB 5000 dataset from Kaggle:\n"
                "https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata\n"
                "and place both CSV files inside your MovieMate/data/ folder."
            )

    
    print("Loading CSV files...")
    movies  = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    print(f"  Movies:  {len(movies)} rows")
    print(f"  Credits: {len(credits)} rows")

    
    df = movies.merge(credits, on="title")
    print(f"  After merge: {len(df)} rows")

    
    print("Parsing genres, cast, crew...")
    df["genre"]    = df["genres"].apply(lambda x: safe_parse(x, "name"))
    df["cast"]     = df["cast"].apply(lambda x: safe_parse(x, "name", limit=3))
    df["director"] = df["crew"].apply(get_director)
    df["keywords"] = df["keywords"].apply(lambda x: safe_parse(x, "name", limit=5))

    
    df = df.rename(columns={
        "overview":       "summary",
        "vote_average":   "rating",
        "vote_count":     "vote_count",
        "release_date":   "release_date",
        "runtime":        "runtime",
        "popularity":     "popularity",
        "original_title": "original_title",
    })

    
    df["year"] = pd.to_datetime(
        df["release_date"], errors="coerce"
    ).dt.year.astype("Int64")

    
    keep = [
        "title", "summary", "genre", "year", "rating",
        "vote_count", "runtime", "director", "cast",
        "keywords", "popularity", "original_language"
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    
    df["rating"]     = pd.to_numeric(df["rating"],     errors="coerce")
    df["runtime"]    = pd.to_numeric(df["runtime"],    errors="coerce")
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce")

    
    df = df[df["title"].notna()   & (df["title"].str.strip()   != "")]
    df = df[df["summary"].notna() & (df["summary"].str.strip() != "")]

   
    df["genre"].fillna("Unknown",    inplace=True)
    df["director"].fillna("Unknown", inplace=True)
    df["cast"].fillna("Unknown",     inplace=True)
    df["runtime"].fillna(df["runtime"].median(), inplace=True)

    
    df = df.drop_duplicates(subset="title", keep="first")
    df = df.reset_index(drop=True)

    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/movies_clean.csv", index=False)

    print(f"\n{'='*50}")
    print(f"Clean dataset saved → data/movies_clean.csv")
    print(f"Total movies:  {len(df)}")
    print(f"Columns:       {list(df.columns)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nSample:\n{df[['title','year','rating','genre']].head(5).to_string()}")

    return df


if __name__ == "__main__":
    df = load_and_clean()
