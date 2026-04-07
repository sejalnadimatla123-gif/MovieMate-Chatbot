

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import faiss


warnings.filterwarnings("ignore")

MODEL_NAME = "all-MiniLM-L6-v2"

INDEX_PATH = "data/movies.index"
DF_PATH    = "data/movies_df.pkl"
TEXTS_PATH = "data/movies_texts.pkl"



def build_movie_text(row):
    return (
        f"Title: {row.get('title', '')}. "
        f"Genre: {row.get('genre', 'Unknown')}. "
        f"Director: {row.get('director', 'Unknown')}. "
        f"Cast: {row.get('cast', 'Unknown')}. "
        f"Year: {row.get('year', 'N/A')}. "
        f"Rating: {row.get('rating', 'N/A')} out of 10. "
        f"Runtime: {row.get('runtime', 'N/A')} minutes. "
        f"Keywords: {row.get('keywords', '')}. "
        f"Summary: {row.get('summary', '')}."
    )


def build_vectorstore():
    from sentence_transformers import SentenceTransformer
    
    
    if os.path.exists(INDEX_PATH):
        print("⚠️ Vectorstore already exists. Skipping rebuild.")
        print("👉 Delete 'data/' folder if you want to rebuild.")
        return load_vectorstore()

    print("🚀 Building vectorstore for the FIRST time...")

    df = pd.read_csv("data/movies_clean.csv").fillna("")
    print(f"📊 Total movies: {len(df)}")

    texts = [build_movie_text(row) for _, row in df.iterrows()]

    print("🧠 Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("⚡ Generating embeddings (this is the ONLY slow step)...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")

    
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    
    os.makedirs("data", exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    with open(DF_PATH, "wb") as f:
        pickle.dump(df, f)

    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(texts, f)

    print("\n✅ Vectorstore built successfully!")
    print("👉 Next time, use load_vectorstore() (instant load)")

    return index, df, texts



def load_vectorstore():

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            "❌ Vectorstore not found.\n"
            "👉 Run build_vectorstore() once first."
        )

    print("⚡ Loading vectorstore (instant)...")

    index = faiss.read_index(INDEX_PATH)

    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    with open(TEXTS_PATH, "rb") as f:
        texts = pickle.load(f)

    return index, df, texts


def retrieve(query, model, index, df, top_k=5):
   

    
    qvec = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(qvec)

    scores, idxs = index.search(qvec, top_k)

    results = df.iloc[idxs[0]].copy()
    results["similarity_score"] = scores[0]

    return results



if __name__ == "__main__":
    print("⚠️ Running vectorstore.py directly")

    index, df, texts = build_vectorstore()

    model = SentenceTransformer(MODEL_NAME)

    print("\n🔍 Test query:")
    results = retrieve("action movies after 2010", model, index, df)

    print(results[["title", "genre", "year", "rating"]].to_string(index=False))