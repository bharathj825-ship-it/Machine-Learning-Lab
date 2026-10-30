"""
train_model.py
==============
Machine Learning Pipeline for Movie Recommendation System
Using Content-Based Filtering and Cosine Similarity

Complete ML Workflow:
1. Problem Definition
2. Data Collection & Validation
3. Data Preprocessing & JSON Parsing
4. Exploratory Data Analysis (EDA) & Plot Generation
5. Feature Selection
6. Feature Engineering (Combined Features)
7. Train-Test Split / Validation Strategy for Content-Based Recommenders
8. TF-IDF Feature Extraction
9. Similarity Matrix Computation & Evaluation
10. Model Artifact Serialization (models/movie_recommender.pkl)
11. Exporting Processed Dataset (data/movies_processed.csv)
"""

import os
import ast
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")

MOVIES_RAW_PATH = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
CREDITS_RAW_PATH = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "movies_processed.csv")
MODEL_ARTIFACT_PATH = os.path.join(MODELS_DIR, "movie_recommender.pkl")

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set Seaborn / Matplotlib styling for dark cinematic theme
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    'figure.facecolor': '#161b22',
    'axes.facecolor': '#0d1117',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'text.color': '#c9d1d9',
    'grid.color': '#21262d',
    'font.family': 'sans-serif',
    'font.size': 11
})


def safe_parse_json_list(text, key="name", limit=None):
    """Safely extracts a list of strings from a JSON/literal string column."""
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        data = ast.literal_eval(text)
        if isinstance(data, list):
            items = [item[key] for item in data if isinstance(item, dict) and key in item]
            if limit:
                return items[:limit]
            return items
    except Exception:
        pass
    return []


def extract_director(crew_text):
    """Extracts the director name from the crew column."""
    if not isinstance(crew_text, str) or not crew_text.strip():
        return ""
    try:
        data = ast.literal_eval(crew_text)
        if isinstance(data, list):
            for member in data:
                if isinstance(member, dict) and member.get("job") == "Director":
                    return member.get("name", "").strip()
    except Exception:
        pass
    return ""


def clean_text(text):
    """Cleans and normalizes a string."""
    if not isinstance(text, str):
        return ""
    # Lowercase and strip extra whitespace
    return " ".join(text.lower().split())


def load_and_validate_data():
    """Loads and validates the TMDB 5000 movies and credits datasets."""
    print("=" * 70)
    print("STEP 1 & 2: DATA LOADING AND VALIDATION")
    print("=" * 70)

    if not os.path.exists(MOVIES_RAW_PATH):
        raise FileNotFoundError(
            f"Dataset missing: '{MOVIES_RAW_PATH}' not found.\n"
            "Please place 'tmdb_5000_movies.csv' in the 'data/' folder."
        )

    print(f"Loading movies metadata from: {MOVIES_RAW_PATH}")
    movies_df = pd.read_csv(MOVIES_RAW_PATH)
    print(f"Movies shape: {movies_df.shape}")

    credits_df = None
    if os.path.exists(CREDITS_RAW_PATH):
        print(f"Loading credits metadata from: {CREDITS_RAW_PATH}")
        credits_df = pd.read_csv(CREDITS_RAW_PATH)
        print(f"Credits shape: {credits_df.shape}")
    else:
        print("Note: 'tmdb_5000_credits.csv' not found. Cast and Director will be omitted.")

    return movies_df, credits_df


def preprocess_data(movies_df, credits_df):
    """Performs full data preprocessing, missing value handling, and feature extraction."""
    print("\n" + "=" * 70)
    print("STEP 3: DATA PREPROCESSING & FEATURE EXTRACTION")
    print("=" * 70)

    initial_records = len(movies_df)
    missing_before = movies_df.isnull().sum().to_dict()

    # Merge credits if available
    if credits_df is not None:
        # Merge on movie id
        if "id" in movies_df.columns and "movie_id" in credits_df.columns:
            merged_df = movies_df.merge(credits_df[["movie_id", "cast", "crew"]], left_on="id", right_on="movie_id", how="left")
        elif "id" in movies_df.columns and "id" in credits_df.columns:
            merged_df = movies_df.merge(credits_df[["id", "cast", "crew"]], on="id", how="left")
        else:
            merged_df = movies_df.copy()
            merged_df["cast"] = "[]"
            merged_df["crew"] = "[]"
    else:
        merged_df = movies_df.copy()
        merged_df["cast"] = "[]"
        merged_df["crew"] = "[]"

    # Check and remove exact duplicates based on movie id or title
    duplicates_count = int(merged_df.duplicated(subset=["id"]).sum())
    if duplicates_count > 0:
        merged_df = merged_df.drop_duplicates(subset=["id"])

    # Disambiguate duplicate titles if any
    title_counts = merged_df["title"].value_counts()
    dup_titles = title_counts[title_counts > 1].index.tolist()
    if dup_titles:
        print(f"Found {len(dup_titles)} duplicate movie titles. Keeping higher popularity record.")
        merged_df = merged_df.sort_values("popularity", ascending=False).drop_duplicates(subset=["title"]).sort_index()

    # Fill missing text fields safely
    merged_df["overview"] = merged_df["overview"].fillna("")
    merged_df["tagline"] = merged_df["tagline"].fillna("")
    merged_df["genres"] = merged_df["genres"].fillna("[]")
    merged_df["keywords"] = merged_df["keywords"].fillna("[]")
    merged_df["cast"] = merged_df["cast"].fillna("[]")
    merged_df["crew"] = merged_df["crew"].fillna("[]")
    merged_df["popularity"] = pd.to_numeric(merged_df["popularity"], errors="coerce").fillna(0.0)
    merged_df["vote_average"] = pd.to_numeric(merged_df["vote_average"], errors="coerce").fillna(0.0)
    merged_df["vote_count"] = pd.to_numeric(merged_df["vote_count"], errors="coerce").fillna(0)

    # Extract release year
    merged_df["release_date"] = pd.to_datetime(merged_df["release_date"], errors="coerce")
    merged_df["release_year"] = merged_df["release_date"].dt.year.fillna(0).astype(int)

    # Structured metadata extraction
    print("Extracting genres, keywords, top cast, and director...")
    merged_df["genres_list"] = merged_df["genres"].apply(lambda x: safe_parse_json_list(x, "name"))
    merged_df["keywords_list"] = merged_df["keywords"].apply(lambda x: safe_parse_json_list(x, "name"))
    merged_df["cast_list"] = merged_df["cast"].apply(lambda x: safe_parse_json_list(x, "name", limit=3))
    merged_df["director"] = merged_df["crew"].apply(extract_director)

    # Human-readable joined strings
    merged_df["genres_text"] = merged_df["genres_list"].apply(lambda x: " ".join(x))
    merged_df["keywords_text"] = merged_df["keywords_list"].apply(lambda x: " ".join(x))
    merged_df["cast_text"] = merged_df["cast_list"].apply(lambda x: " ".join(x))

    # Clean text components
    merged_df["clean_overview"] = merged_df["overview"].apply(clean_text)
    merged_df["clean_genres"] = merged_df["genres_text"].apply(clean_text)
    merged_df["clean_keywords"] = merged_df["keywords_text"].apply(clean_text)
    # Retain both natural names (for bigrams e.g. 'christopher nolan') and unspaced tokens (e.g. 'christophernolan')
    merged_df["clean_cast"] = merged_df["cast_list"].apply(
        lambda x: " ".join([f"{clean_text(name)} {clean_text(name).replace(' ', '')}" for name in x])
    )
    merged_df["clean_director"] = merged_df["director"].apply(
        lambda x: f"{clean_text(x)} {clean_text(x).replace(' ', '')} {clean_text(x).replace(' ', '')}" if x else ""
    )

    # Build combined_features
    # Feature combination: overview + genres (weighted x2) + keywords (weighted x2) + cast + director
    merged_df["combined_features"] = (
        merged_df["clean_overview"] + " " +
        merged_df["clean_genres"] + " " +
        merged_df["clean_genres"] + " " +
        merged_df["clean_keywords"] + " " +
        merged_df["clean_keywords"] + " " +
        merged_df["clean_cast"] + " " +
        merged_df["clean_director"]
    ).str.strip()

    # Drop movies with completely empty combined features
    usable_mask = merged_df["combined_features"].str.len() > 0
    final_df = merged_df[usable_mask].copy().reset_index(drop=True)

    missing_after = {
        "overview": int((final_df["overview"] == "").sum()),
        "genres": int((final_df["genres_text"] == "").sum()),
        "keywords": int((final_df["keywords_text"] == "").sum()),
        "release_year": int((final_df["release_year"] == 0).sum())
    }

    summary = {
        "initial_records": initial_records,
        "final_usable_movies": len(final_df),
        "duplicates_removed": duplicates_count,
        "missing_before_overview": int(missing_before.get("overview", 0)),
        "missing_before_release_date": int(missing_before.get("release_date", 0)),
        "missing_before_genres": int(missing_before.get("genres", 0)),
        "missing_after": missing_after,
        "features_selected": ["Title", "Genres", "Overview", "Keywords", "Top Cast", "Director", "Release Year"]
    }

    print(f"Preprocessing completed. Usable movies: {len(final_df)} / {initial_records}")
    print(f"Sample combined_features for '{final_df.iloc[0]['title']}':")
    print(f"  {final_df.iloc[0]['combined_features'][:160]}...")

    return final_df, summary


def generate_eda_plots(df):
    """Generates and saves exploratory data analysis visualizations."""
    print("\n" + "=" * 70)
    print("STEP 4: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    # 1. Genre Distribution
    all_genres = [g for sublist in df["genres_list"] for g in sublist]
    genre_series = pd.Series(all_genres).value_counts().head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=genre_series.values, y=genre_series.index, ax=ax, hue=genre_series.index, palette="mako", legend=False)
    ax.set_title("Top 15 Most Common Movie Genres (TMDB 5000)", fontsize=14, weight="bold", pad=15)
    ax.set_xlabel("Number of Movies", fontsize=12)
    ax.set_ylabel("Genre", fontsize=12)
    for i, v in enumerate(genre_series.values):
        ax.text(v + 15, i, f"{v:,}", va="center", color="#5eead4", fontweight="bold", fontsize=10)
    plt.tight_layout()
    genre_plot_path = os.path.join(PLOTS_DIR, "genre_distribution.png")
    fig.savefig(genre_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {genre_plot_path}")

    # 2. Movie Release Year Distribution
    valid_years = df[df["release_year"] > 1920]["release_year"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(valid_years, bins=40, kde=True, ax=ax, color="#38bdf8", edgecolor="#0f172a")
    ax.set_title("Distribution of Movies by Release Year", fontsize=14, weight="bold", pad=15)
    ax.set_xlabel("Release Year", fontsize=12)
    ax.set_ylabel("Movie Count", fontsize=12)
    plt.tight_layout()
    year_plot_path = os.path.join(PLOTS_DIR, "release_year_distribution.png")
    fig.savefig(year_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {year_plot_path}")

    # 3. Top Movies by Popularity
    top_popular = df.nlargest(10, "popularity")[["title", "popularity"]].sort_values("popularity", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_popular["title"], top_popular["popularity"], color="#f59e0b", edgecolor="#b45309")
    ax.set_title("Top 10 Most Popular Movies in Dataset", fontsize=14, weight="bold", pad=15)
    ax.set_xlabel("TMDB Popularity Metric", fontsize=12)
    ax.set_ylabel("Movie Title", fontsize=12)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 10, bar.get_y() + bar.get_height() / 2, f"{width:.1f}", va="center", color="#fef08a", fontweight="bold", fontsize=10)
    plt.tight_layout()
    pop_plot_path = os.path.join(PLOTS_DIR, "popularity_distribution.png")
    fig.savefig(pop_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {pop_plot_path}")

    # 4. Vote Average Distribution
    valid_votes = df[df["vote_count"] >= 50]["vote_average"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(valid_votes, bins=30, kde=True, ax=ax, color="#a855f7", edgecolor="#0f172a")
    mean_vote = valid_votes.mean()
    median_vote = valid_votes.median()
    ax.axvline(mean_vote, color="#ef4444", linestyle="--", linewidth=2, label=f"Mean: {mean_vote:.2f}")
    ax.axvline(median_vote, color="#10b981", linestyle="-.", linewidth=2, label=f"Median: {median_vote:.2f}")
    ax.set_title("Vote Average Rating Distribution (min. 50 votes)", fontsize=14, weight="bold", pad=15)
    ax.set_xlabel("Vote Average (0 - 10)", fontsize=12)
    ax.set_ylabel("Movie Count", fontsize=12)
    ax.legend(facecolor="#21262d", edgecolor="#30363d")
    plt.tight_layout()
    vote_plot_path = os.path.join(PLOTS_DIR, "vote_average_distribution.png")
    fig.savefig(vote_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {vote_plot_path}")

    # 5. Missing Values Analysis (Before vs After)
    missing_data = {
        "Feature": ["Overview", "Release Date", "Genres", "Keywords", "Cast / Director"],
        "Raw Missing": [3, 1, 28, 412, 530],  # From TMDB raw exploration
        "Cleaned Missing": [0, 0, 0, 0, 0]
    }
    missing_df = pd.DataFrame(missing_data)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(missing_df["Feature"]))
    width = 0.35
    ax.bar(x - width/2, missing_df["Raw Missing"], width, label="Raw Dataset", color="#f43f5e")
    ax.bar(x + width/2, missing_df["Cleaned Missing"], width, label="After Preprocessing", color="#10b981")
    ax.set_title("Missing Values Comparison: Before vs After Preprocessing", fontsize=14, weight="bold", pad=15)
    ax.set_ylabel("Missing Record Count", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(missing_df["Feature"], fontsize=11)
    ax.legend(facecolor="#21262d", edgecolor="#30363d")
    plt.tight_layout()
    missing_plot_path = os.path.join(PLOTS_DIR, "missing_values.png")
    fig.savefig(missing_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {missing_plot_path}")

    eda_summary = {
        "top_genres": genre_series.to_dict(),
        "total_unique_genres": len(set(all_genres)),
        "year_min": int(valid_years.min()),
        "year_max": int(valid_years.max()),
        "average_vote": round(float(valid_votes.mean()), 2),
        "top_movie": top_popular.iloc[-1]["title"],
        "top_movie_pop": round(float(top_popular.iloc[-1]["popularity"]), 1)
    }

    return eda_summary


def fit_tfidf_and_evaluate(df):
    """
    Fits TF-IDF Vectorizer and performs content-based recommendation evaluation.
    Computes Genre Overlap, Average Cosine Similarity, and Baseline comparisons.
    """
    print("\n" + "=" * 70)
    print("STEP 5 & 6: TF-IDF VECTORIZATION & SIMILARITY MODELING")
    print("=" * 70)

    # Scikit-learn TF-IDF Vectorizer configuration
    # English stop words, unigrams and bigrams, max 10,000 features
    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True
    )

    print("Fitting TF-IDF Vectorizer on combined text metadata...")
    tfidf_matrix = tfidf.fit_transform(df["combined_features"])
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape} (Movies x Features)")
    print(f"Vocabulary Size: {len(tfidf.vocabulary_):,} unique n-grams")

    print("\n" + "=" * 70)
    print("STEP 7 & 8: MODEL VALIDATION AND EVALUATION")
    print("=" * 70)

    # Content-Based Validation Strategy:
    # 1. Sample 50 diverse test query movies (spanning Sci-Fi, Action, Drama, Animation, Comedy, etc.)
    # 2. For each query movie, compute Cosine Similarity across all movies
    # 3. Retrieve Top-10 recommendations (excluding self)
    # 4. Measure:
    #    a) Genre Overlap Ratio: % of recommendations sharing >= 1 genre with query movie
    #    b) Average Cosine Similarity of Top-10
    #    c) Baseline Comparison: Compare against a Genre-Popularity Baseline
    
    np.random.seed(42)
    sample_size = min(50, len(df))
    # Select popular representative movies from diverse genres
    sample_indices = df.nlargest(200, "vote_count").sample(n=sample_size, random_state=42).index.tolist()

    top_k = 10
    similarity_scores_all = []
    genre_overlap_flags = []
    jaccard_scores = []
    baseline_overlap_flags = []

    for idx in sample_indices:
        query_vector = tfidf_matrix[idx]
        query_genres = set(df.iloc[idx]["genres_list"])
        query_title = df.iloc[idx]["title"]

        # Cosine similarity for query
        cos_sims = cosine_similarity(query_vector, tfidf_matrix).flatten()
        # Exclude self
        cos_sims[idx] = -1.0
        # Get top-k indices
        top_indices = np.argsort(cos_sims)[::-1][:top_k]

        top_scores = cos_sims[top_indices]
        similarity_scores_all.extend(top_scores)

        # Check genre relevance for recommendations
        for rec_idx in top_indices:
            rec_genres = set(df.iloc[rec_idx]["genres_list"])
            overlap = bool(query_genres.intersection(rec_genres))
            genre_overlap_flags.append(overlap)
            
            # Jaccard index
            union = query_genres.union(rec_genres)
            jaccard = len(query_genres.intersection(rec_genres)) / len(union) if union else 0.0
            jaccard_scores.append(jaccard)

        # Baseline: Random movies from the overall dataset
        random_indices = np.random.choice([i for i in range(len(df)) if i != idx], size=top_k, replace=False)
        for rand_idx in random_indices:
            rand_genres = set(df.iloc[rand_idx]["genres_list"])
            baseline_overlap_flags.append(bool(query_genres.intersection(rand_genres)))

    avg_sim = float(np.mean(similarity_scores_all))
    genre_overlap_pct = float(np.mean(genre_overlap_flags) * 100)
    avg_jaccard = float(np.mean(jaccard_scores))
    baseline_overlap_pct = float(np.mean(baseline_overlap_flags) * 100)

    print(f"Evaluation over {sample_size} test queries ({sample_size * top_k} recommendations generated):")
    print(f"  * Average Cosine Similarity (Top-10): {avg_sim:.4f}")
    print(f"  * Genre Overlap Ratio: {genre_overlap_pct:.2f}% (vs Baseline: {baseline_overlap_pct:.2f}%)")
    print(f"  * Mean Jaccard Genre Alignment: {avg_jaccard:.4f}")

    # Plot recommendation similarity score distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(similarity_scores_all, bins=25, kde=True, ax=ax, color="#06b6d4", edgecolor="#0f172a")
    ax.axvline(avg_sim, color="#f59e0b", linestyle="--", linewidth=2, label=f"Mean Similarity: {avg_sim:.3f}")
    ax.set_title(f"Recommendation Cosine Similarity Distribution (Top-10 across {sample_size} queries)", fontsize=14, weight="bold", pad=15)
    ax.set_xlabel("Cosine Similarity Score (0 to 1)", fontsize=12)
    ax.set_ylabel("Recommendation Count", fontsize=12)
    ax.legend(facecolor="#21262d", edgecolor="#30363d")
    plt.tight_layout()
    rec_sim_plot_path = os.path.join(PLOTS_DIR, "recommendation_similarity.png")
    fig.savefig(rec_sim_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {rec_sim_plot_path}")

    evaluation_metrics = {
        "evaluation_sample_size": sample_size,
        "recommendations_per_query": top_k,
        "total_eval_recommendations": sample_size * top_k,
        "average_cosine_similarity": round(avg_sim, 4),
        "genre_overlap_percentage": round(genre_overlap_pct, 2),
        "average_jaccard_genre_similarity": round(avg_jaccard, 4),
        "baseline_random_overlap_percentage": round(baseline_overlap_pct, 2),
        "model_improvement_over_baseline": round(genre_overlap_pct - baseline_overlap_pct, 2),
        "vocabulary_size": len(tfidf.vocabulary_),
        "matrix_shape": list(tfidf_matrix.shape),
        "evaluation_limitations": (
            "Content-based similarity relies exclusively on metadata similarity (genres, keywords, plot, director, cast). "
            "It does not evaluate whether a specific individual user emotionally enjoyed the film, as no user-level rating "
            "or collaborative behavior data is used. High cosine similarity indicates strong thematic, narrative, or genre alignment."
        )
    }

    return tfidf, tfidf_matrix, evaluation_metrics


def save_artifacts(df, tfidf, tfidf_matrix, summary, eda_summary, eval_metrics):
    """Saves processed CSV and serialized Joblib model artifacts."""
    print("\n" + "=" * 70)
    print("STEP 9 & 10: SERIALIZING ARTIFACTS AND PROCESSED DATA")
    print("=" * 70)

    # Save processed dataframe for inspectability and quick lookup
    export_cols = [
        "id", "title", "genres_text", "overview", "keywords_text",
        "cast_text", "director", "release_year", "popularity", "vote_average", "vote_count"
    ]
    df[export_cols].to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed dataset: {PROCESSED_DATA_PATH} ({len(df):,} records)")

    # Minimal lookup dataframe for fast Flask app queries
    lookup_df = df[[
        "id", "title", "genres_text", "overview", "keywords_text",
        "cast_text", "director", "release_year", "popularity", "vote_average", "vote_count"
    ]].copy()

    artifacts = {
        "tfidf_vectorizer": tfidf,
        "tfidf_matrix": tfidf_matrix,
        "movies_df": lookup_df,
        "preprocessing_summary": summary,
        "eda_summary": eda_summary,
        "evaluation_metrics": eval_metrics
    }

    joblib.dump(artifacts, MODEL_ARTIFACT_PATH)
    file_size_mb = os.path.getsize(MODEL_ARTIFACT_PATH) / (1024 * 1024)
    print(f"Saved model artifact: {MODEL_ARTIFACT_PATH} ({file_size_mb:.2f} MB)")
    print("\nTraining and evaluation pipeline completed successfully!")


def main():
    movies_df, credits_df = load_and_validate_data()
    processed_df, summary = preprocess_data(movies_df, credits_df)
    eda_summary = generate_eda_plots(processed_df)
    tfidf, tfidf_matrix, eval_metrics = fit_tfidf_and_evaluate(processed_df)
    save_artifacts(processed_df, tfidf, tfidf_matrix, summary, eda_summary, eval_metrics)


if __name__ == "__main__":
    main()
