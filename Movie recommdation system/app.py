"""
app.py
======
Flask Web Application for Movie Recommendation System
Using Content-Based Filtering and Cosine Similarity

Suitable for College ML Mini-Project Demonstrations
Runs 100% locally with zero cloud/external API dependencies.
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_ARTIFACT_PATH = os.path.join(MODELS_DIR, "movie_recommender.pkl")

# Global Cache for Model Artifacts
ARTIFACTS = None
MOVIES_DF = None
TFIDF_MATRIX = None
TFIDF_VECTORIZER = None
ALL_TITLES = []
ALL_GENRES = []
EVAL_METRICS = {}
PREPROCESS_SUMMARY = {}
EDA_SUMMARY = {}


def load_artifacts():
    """Loads the precomputed recommendation artifacts and metadata."""
    global ARTIFACTS, MOVIES_DF, TFIDF_MATRIX, TFIDF_VECTORIZER
    global ALL_TITLES, ALL_GENRES, EVAL_METRICS, PREPROCESS_SUMMARY, EDA_SUMMARY

    if not os.path.exists(MODEL_ARTIFACT_PATH):
        print(f"Warning: Model artifact not found at {MODEL_ARTIFACT_PATH}.")
        print("Please run `python train_model.py` to train and build the artifacts.")
        return False

    try:
        print(f"Loading recommendation model artifacts from {MODEL_ARTIFACT_PATH}...")
        ARTIFACTS = joblib.load(MODEL_ARTIFACT_PATH)
        MOVIES_DF = ARTIFACTS["movies_df"]
        TFIDF_MATRIX = ARTIFACTS["tfidf_matrix"]
        TFIDF_VECTORIZER = ARTIFACTS["tfidf_vectorizer"]
        PREPROCESS_SUMMARY = ARTIFACTS.get("preprocessing_summary", {})
        EDA_SUMMARY = ARTIFACTS.get("eda_summary", {})
        EVAL_METRICS = ARTIFACTS.get("evaluation_metrics", {})

        # Extract sorted unique movie titles
        ALL_TITLES = sorted(MOVIES_DF["title"].dropna().unique().tolist())

        # Extract sorted unique genres
        raw_genres = set()
        for text in MOVIES_DF["genres_text"].dropna():
            for g in text.split():
                if g:
                    raw_genres.add(g)
        ALL_GENRES = sorted(list(raw_genres))

        print(f"Loaded {len(MOVIES_DF):,} movies and {len(ALL_GENRES)} genres.")
        return True
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return False


# Attempt initial load
artifacts_loaded = load_artifacts()


def find_movie_by_title(query_title):
    """
    Finds a movie in MOVIES_DF by exact or case-insensitive match,
    or falls back to substring matches.
    """
    if MOVIES_DF is None or not query_title:
        return None, []

    query_clean = query_title.strip().lower()

    # 1. Exact case-insensitive match
    exact_match = MOVIES_DF[MOVIES_DF["title"].str.lower() == query_clean]
    if not exact_match.empty:
        return exact_match.index[0], []

    # 2. Substring match
    sub_matches = MOVIES_DF[MOVIES_DF["title"].str.lower().str.contains(query_clean, regex=False)]
    if len(sub_matches) == 1:
        return sub_matches.index[0], []
    elif len(sub_matches) > 1:
        # Multiple candidates found, return suggestions
        suggestions = sub_matches["title"].head(5).tolist()
        return None, suggestions

    # 3. No match found - suggest closest starting matches
    starts_with = MOVIES_DF[MOVIES_DF["title"].str.lower().str.startswith(query_clean[:3])]["title"].head(5).tolist()
    return None, starts_with


def get_recommendations(movie_idx, genre_filter="all", top_n=5):
    """
    Calculates cosine similarity of the selected movie against all other movies.
    Excludes the query movie, optionally filters by genre, and returns Top-N.
    """
    if MOVIES_DF is None or TFIDF_MATRIX is None:
        return []

    # Retrieve TF-IDF vector of query movie
    query_vector = TFIDF_MATRIX[movie_idx]

    # Calculate Cosine Similarity with all movies in corpus
    similarity_scores = cosine_similarity(query_vector, TFIDF_MATRIX).flatten()

    # Exclude the selected movie itself
    similarity_scores[movie_idx] = -1.0

    # Apply genre filter if specified
    valid_indices = np.arange(len(MOVIES_DF))
    if genre_filter and genre_filter.lower() != "all":
        genre_mask = MOVIES_DF["genres_text"].str.contains(genre_filter, case=False, na=False).values
        # Mask out non-matching movies
        similarity_scores = np.where(genre_mask, similarity_scores, -1.0)

    # Sort descending by similarity
    ranked_indices = np.argsort(similarity_scores)[::-1]

    recommendations = []
    for idx in ranked_indices:
        score = similarity_scores[idx]
        if score <= 0:
            break
        row = MOVIES_DF.iloc[idx]
        recommendations.append({
            "id": int(row["id"]),
            "title": str(row["title"]),
            "release_year": int(row["release_year"]) if row["release_year"] > 0 else "N/A",
            "genres": [g for g in str(row["genres_text"]).split() if g],
            "genres_str": str(row["genres_text"]),
            "overview": str(row["overview"]) if str(row["overview"]) else "No description available.",
            "director": str(row["director"]) if str(row["director"]) else "Not specified",
            "cast": str(row["cast_text"]) if str(row["cast_text"]) else "Not specified",
            "similarity_score": round(float(score), 4),
            "similarity_pct": round(float(score) * 100, 1),
            "vote_average": round(float(row["vote_average"]), 1),
            "popularity": round(float(row["popularity"]), 1)
        })
        if len(recommendations) >= top_n:
            break

    return recommendations


@app.route("/", methods=["GET"])
def index():
    """Main route displaying the workflow, statistics, EDA, and recommendation interface."""
    # Ensure artifacts are loaded
    global ARTIFACTS
    if ARTIFACTS is None:
        load_artifacts()

    context = {
        "artifacts_loaded": ARTIFACTS is not None,
        "all_titles": ALL_TITLES,
        "all_genres": ALL_GENRES,
        "total_movies": len(MOVIES_DF) if MOVIES_DF is not None else 0,
        "preprocess_summary": PREPROCESS_SUMMARY,
        "eda_summary": EDA_SUMMARY,
        "eval_metrics": EVAL_METRICS,
        "selected_movie": None,
        "recommendations": None,
        "error_message": None,
        "suggestions": [],
        "genre_filter": "all",
        "num_recommendations": 5
    }
    return render_template("index.html", **context)


@app.route("/recommend", methods=["POST"])
def recommend():
    """Processes movie recommendation requests and renders results."""
    # Ensure artifacts are loaded
    global ARTIFACTS
    if ARTIFACTS is None:
        if not load_artifacts():
            return render_template(
                "index.html",
                artifacts_loaded=False,
                error_message="Model artifacts not found. Please run `python train_model.py` first."
            )

    movie_title = request.form.get("movie_title", "").strip()
    genre_filter = request.form.get("genre_filter", "all").strip()
    try:
        num_recommendations = int(request.form.get("num_recommendations", 5))
        if num_recommendations not in [5, 10]:
            num_recommendations = 5
    except ValueError:
        num_recommendations = 5

    # Base context
    context = {
        "artifacts_loaded": True,
        "all_titles": ALL_TITLES,
        "all_genres": ALL_GENRES,
        "total_movies": len(MOVIES_DF),
        "preprocess_summary": PREPROCESS_SUMMARY,
        "eda_summary": EDA_SUMMARY,
        "eval_metrics": EVAL_METRICS,
        "genre_filter": genre_filter,
        "num_recommendations": num_recommendations,
        "searched_title": movie_title,
        "selected_movie": None,
        "recommendations": None,
        "error_message": None,
        "suggestions": []
    }

    if not movie_title:
        context["error_message"] = "Please enter or select a movie title."
        return render_template("index.html", **context)

    # Search for movie in processed dataset
    movie_idx, suggestions = find_movie_by_title(movie_title)

    if movie_idx is None:
        context["error_message"] = f"Movie '{movie_title}' was not found in the dataset."
        context["suggestions"] = suggestions
        return render_template("index.html", **context)

    # Query movie details
    query_row = MOVIES_DF.iloc[movie_idx]
    selected_movie = {
        "id": int(query_row["id"]),
        "title": str(query_row["title"]),
        "release_year": int(query_row["release_year"]) if query_row["release_year"] > 0 else "N/A",
        "genres": [g for g in str(query_row["genres_text"]).split() if g],
        "overview": str(query_row["overview"]) if str(query_row["overview"]) else "No description available.",
        "director": str(query_row["director"]) if str(query_row["director"]) else "Not specified",
        "cast": str(query_row["cast_text"]) if str(query_row["cast_text"]) else "Not specified",
        "vote_average": round(float(query_row["vote_average"]), 1),
        "popularity": round(float(query_row["popularity"]), 1)
    }

    # Generate recommendations
    recommendations = get_recommendations(movie_idx, genre_filter=genre_filter, top_n=num_recommendations)

    context["selected_movie"] = selected_movie
    context["recommendations"] = recommendations

    # If requested via JSON API or AJAX
    if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.args.get("format") == "json":
        return jsonify({
            "status": "success",
            "selected_movie": selected_movie,
            "recommendations": recommendations
        })

    return render_template("index.html", **context)


@app.route("/api/movies", methods=["GET"])
def api_movies():
    """Returns the list of available movie titles for client-side search autocomplete."""
    if MOVIES_DF is None:
        return jsonify({"titles": [], "genres": []})
    return jsonify({
        "titles": ALL_TITLES,
        "genres": ALL_GENRES
    })


if __name__ == "__main__":
    # College lab local server runner
    print("\n" + "=" * 65)
    print("Movie Recommendation System (Content-Based Filtering)")
    print("Local Flask Web Application Running on http://127.0.0.1:5000")
    print("=" * 65 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
