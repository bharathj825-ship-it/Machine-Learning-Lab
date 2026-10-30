"""
generate_pdf_report.py
======================
Generates an academic College Project Report PDF for the
Movie Recommendation System Using Content-Based Filtering and Cosine Similarity,
matching the exact design, formatting, code boxes, and step-by-step screenshot layout
of the sample lab report.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_IMG_DIR = os.path.join(BASE_DIR, "report_images")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")
PDF_OUTPUT_PATH = os.path.join(BASE_DIR, "Movie_Recommendation_System_Project_Report.pdf")


def build_pdf():
    # 0.75 inch (54 pt) margins on letter size
    doc = SimpleDocTemplate(
        PDF_OUTPUT_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13.5,
        leading=18,
        alignment=1, # Center
        spaceAfter=3
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=11,
        leading=15,
        alignment=1, # Center
        spaceAfter=14
    )

    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10,
        leading=14.5,
        alignment=4, # Justify
        spaceAfter=7
    )

    bullet_style = ParagraphStyle(
        'BulletItem',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.8,
        leading=14,
        leftIndent=16,
        spaceAfter=3
    )

    code_title_style = ParagraphStyle(
        'CodeTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        spaceBefore=8,
        spaceAfter=4
    )

    code_style = ParagraphStyle(
        'CodeContent',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#1f2937')
    )

    caption_style = ParagraphStyle(
        'FigCaption',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=9.5,
        leading=13,
        alignment=1, # Center
        spaceBefore=5,
        spaceAfter=8
    )

    def create_code_box(code_text):
        """Creates a styled gray code box table."""
        p = Paragraph(code_text.replace("\n", "<br/>").replace(" ", "&nbsp;"), code_style)
        t = Table([[p]], colWidths=[500])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8f9fa')),
            ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor('#d1d5db')),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        return t

    story = []

    # ==========================================
    # PAGE 1: Problem Statement, Objectives, Specs
    # ==========================================
    story.append(Paragraph("Movie Recommendation System Using Content-Based Filtering and Cosine Similarity", title_style))
    story.append(Paragraph("(Movie Recommendation System Using Machine Learning)", subtitle_style))

    story.append(Paragraph("Problem Statement:", heading_style))
    story.append(Paragraph(
        "With the explosive expansion of digital entertainment and streaming platforms, users face severe cognitive "
        "information overload when attempting to select movies aligned with their narrative preferences. Traditional "
        "collaborative filtering methods rely on user rating matrices, which severely limits their utility when encountering "
        "new or niche movies (the classic &ldquo;cold-start&rdquo; challenge). Furthermore, user rating patterns are often "
        "sparse, noisy, and subject to external demographic biases.",
        body_style
    ))
    story.append(Paragraph(
        "There is, therefore, a need for an automated, fast, and accessible system that can analyze a movie's inherent metadata "
        "(plot description, genres, thematic keywords, cast, and director) and recommend highly similar films with a measurable "
        "mathematical confidence score. This project designs and implements a web-based Movie Recommendation System that performs "
        "data preprocessing, TF-IDF feature extraction, and similarity computation using Cosine Similarity on the benchmark "
        "TMDB 5000 Movie Dataset, and evaluates the recommendation quality using standard content-based metrics (Genre Overlap "
        "Ratio, Jaccard Similarity, and Baseline Comparison).",
        body_style
    ))

    story.append(Paragraph("Objectives:", heading_style))
    story.append(Paragraph("1. To collect and preprocess a comprehensive dataset of movie metadata (TMDB 5000 dataset).", bullet_style))
    story.append(Paragraph("2. To parse complex JSON columns into clean textual features (overview, genres, keywords, cast, director).", bullet_style))
    story.append(Paragraph("3. To construct a domain-weighted <code>combined_features</code> textual representation for each movie.", bullet_style))
    story.append(Paragraph("4. To perform Exploratory Data Analysis (EDA) on genre distributions, release years, vote ratings, and popularity.", bullet_style))
    story.append(Paragraph("5. To extract a 10,000-dimensional unigram and bigram numerical feature vector using Scikit-learn's TF-IDF Vectorizer.", bullet_style))
    story.append(Paragraph("6. To compute on-demand Cosine Similarity scores to rank and retrieve candidate recommendations.", bullet_style))
    story.append(Paragraph("7. To evaluate the recommendation model on a sample query set and report Genre Overlap Ratio and Baseline Gain.", bullet_style))
    story.append(Paragraph("8. To build a Flask-based backend and web front-end that lets a user select a movie and view real-time recommendations.", bullet_style))

    story.append(Paragraph("Software / Hardware Required:", heading_style))
    story.append(Paragraph("&bull; Operating System: Windows / Linux / macOS", bullet_style))
    story.append(Paragraph("&bull; IDE: Visual Studio Code", bullet_style))
    story.append(Paragraph("&bull; Language: Python 3.12.x", bullet_style))
    story.append(Paragraph("&bull; Backend Framework: Flask (REST API & Local Web Server)", bullet_style))

    story.append(PageBreak())

    # ==========================================
    # PAGE 2: Specs Continued & Directory Structure
    # ==========================================
    story.append(Paragraph("&bull; Machine Learning: scikit-learn (TfidfVectorizer, cosine_similarity)", bullet_style))
    story.append(Paragraph("&bull; Data Processing: Pandas, NumPy", bullet_style))
    story.append(Paragraph("&bull; Data Visualization: Matplotlib, Seaborn", bullet_style))
    story.append(Paragraph("&bull; Model Serialization: Joblib", bullet_style))
    story.append(Paragraph("&bull; Frontend: HTML5, CSS3, JavaScript (Cinematic Dark UI)", bullet_style))
    story.append(Paragraph("&bull; Dataset: TMDB 5000 Movie Dataset (4,803 raw titles; 4,800 usable cleaned records)", bullet_style))
    story.append(Paragraph("&bull; Version Control: Git", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Project Directory Structure:", heading_style))
    story.append(Paragraph(
        "The project (&ldquo;MovieRecommendationSystem&rdquo;) is organized as a Flask backend with a separate dataset, "
        "trained-model, and frontend static/templates folder, as shown below.",
        body_style
    ))

    fig1_path = os.path.join(REPORT_IMG_DIR, "fig1_vscode_explorer.png")
    if os.path.exists(fig1_path):
        story.append(Spacer(1, 4))
        story.append(Image(fig1_path, width=280, height=365))
        story.append(Paragraph("Fig 1: Project directory structure of MovieRecommendationSystem in Visual Studio Code", caption_style))

    story.append(PageBreak())

    # ==========================================
    # PAGE 3: Procedure & Code Snippets 1
    # ==========================================
    story.append(Paragraph("Procedure:", heading_style))
    story.append(Paragraph("1. Collect the TMDB 5000 movie dataset (movies and credits CSVs) and place them in the <code>data/</code> folder.", bullet_style))
    story.append(Paragraph("2. Run <code>train_model.py</code> to clean, parse, and organize the raw dataset into the project's processed structure.", bullet_style))
    story.append(Paragraph("3. Preprocess metadata &mdash; extract director, cast, keywords, and genres from JSON; handle missing plot overviews.", bullet_style))
    story.append(Paragraph("4. Construct a weighted <code>combined_features</code> textual column with duplicated keywords, genres, and director tokens.", bullet_style))
    story.append(Paragraph("5. Generate exploratory visualizations (genre frequencies, release year histogram, popularity, ratings) into <code>static/plots/</code>.", bullet_style))
    story.append(Paragraph("6. Extract a 10,000-dimensional TF-IDF feature space from the combined text using unigrams and bigrams with English stop-words.", bullet_style))
    story.append(Paragraph("7. Calculate Cosine Similarity angles between document vectors to quantify narrative proximity bounded in [0, 1].", bullet_style))
    story.append(Paragraph("8. Evaluate the similarity engine across 50 query movies (500 recommendations) and report Genre Overlap Ratio and Jaccard alignment.", bullet_style))
    story.append(Paragraph("9. Save the fitted vectorizer, sparse matrix, and lookup tables to <code>models/movie_recommender.pkl</code> using Joblib.", bullet_style))
    story.append(Paragraph("10. Launch the Flask web server (<code>app.py</code>) and query recommendations dynamically via the interactive browser interface.", bullet_style))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Implementation Code:", heading_style))
    story.append(Paragraph(
        "Representative implementation of the key modules used in the project, based on the pipeline shown in the application (adapt variable/file names to match your exact source files).",
        body_style
    ))

    story.append(Paragraph("1. Data Preprocessing & JSON Parsing (train_model.py)", code_title_style))
    code1 = """import ast, pandas as pd

def safe_parse_json_list(text, key='name', limit=None):
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        data = ast.literal_eval(text)
        if isinstance(data, list):
            items = [item[key] for item in data if isinstance(item, dict) and key in item]
            return items[:limit] if limit else items
    except Exception:
        pass
    return []

def extract_director(crew_text):
    try:
        data = ast.literal_eval(crew_text)
        for member in data:
            if member.get('job') == 'Director':
                return member.get('name', '').strip()
    except Exception:
        pass
    return ''"""
    story.append(create_code_box(code1))

    story.append(PageBreak())

    # ==========================================
    # PAGE 4: Implementation Code 2 & 3
    # ==========================================
    story.append(Paragraph("2. Feature Extraction & Combined Features (train_model.py)", code_title_style))
    code2 = """# Retain both natural names (for bigrams) and unspaced tokens
merged_df['clean_overview'] = merged_df['overview'].apply(clean_text)
merged_df['clean_genres'] = merged_df['genres_text'].apply(clean_text)
merged_df['clean_keywords'] = merged_df['keywords_text'].apply(clean_text)
merged_df['clean_cast'] = merged_df['cast_list'].apply(
    lambda x: ' '.join([f"{clean_text(n)} {clean_text(n).replace(' ', '')}" for n in x])
)
merged_df['clean_director'] = merged_df['director'].apply(
    lambda x: f"{clean_text(x)} {clean_text(x).replace(' ', '')} {clean_text(x).replace(' ', '')}" if x else ''
)

# Domain-weighted combined features
merged_df['combined_features'] = (
    merged_df['clean_overview'] + ' ' +
    merged_df['clean_genres'] + ' ' + merged_df['clean_genres'] + ' ' +
    merged_df['clean_keywords'] + ' ' + merged_df['clean_keywords'] + ' ' +
    merged_df['clean_cast'] + ' ' +
    merged_df['clean_director']
).str.strip()"""
    story.append(create_code_box(code2))

    story.append(Spacer(1, 8))
    story.append(Paragraph("3. Model Training & Serialization (train_model.py)", code_title_style))
    code3 = """from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# 10,000-dimensional unigram/bigram TF-IDF vectorizer
tfidf = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True
)

# Fit and transform combined text
tfidf_matrix = tfidf.fit_transform(processed_df['combined_features'])
print('TF-IDF Matrix Shape:', tfidf_matrix.shape)

# Persist recommendation model artifacts
artifacts = {
    'tfidf_vectorizer': tfidf,
    'tfidf_matrix': tfidf_matrix,
    'movies_df': processed_df[['id', 'title', 'genres_text', 'overview', 'release_year', 'popularity']],
    'evaluation_metrics': eval_metrics
}
joblib.dump(artifacts, 'models/movie_recommender.pkl')"""
    story.append(create_code_box(code3))

    story.append(PageBreak())

    # ==========================================
    # PAGE 5: Implementation Code 4
    # ==========================================
    story.append(Paragraph("4. Prediction & Recommendation Route (app.py)", code_title_style))
    code4 = """from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity
import joblib, numpy as np

app = Flask(__name__)
artifacts = joblib.load('models/movie_recommender.pkl')
MOVIES_DF = artifacts['movies_df']
TFIDF_MATRIX = artifacts['tfidf_matrix']

def get_recommendations(movie_idx, genre_filter='all', top_n=5):
    query_vector = TFIDF_MATRIX[movie_idx]
    sim_scores = cosine_similarity(query_vector, TFIDF_MATRIX).flatten()
    sim_scores[movie_idx] = -1.0 # Exclude query movie

    if genre_filter and genre_filter.lower() != 'all':
        mask = MOVIES_DF['genres_text'].str.contains(genre_filter, case=False, na=False).values
        sim_scores = np.where(mask, sim_scores, -1.0)

    ranked_indices = np.argsort(sim_scores)[::-1][:top_n]
    recommendations = []
    for idx in ranked_indices:
        row = MOVIES_DF.iloc[idx]
        recommendations.append({
            'title': row['title'],
            'release_year': row['release_year'],
            'genres': row['genres_text'],
            'similarity_score': round(float(sim_scores[idx]), 4),
            'similarity_pct': round(float(sim_scores[idx]) * 100, 1)
        })
    return recommendations

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form.get('movie_title', '').strip()
    genre_filter = request.form.get('genre_filter', 'all').strip()
    top_n = int(request.form.get('num_recommendations', 5))

    movie_idx, suggestions = find_movie_by_title(movie_title)
    if movie_idx is None:
        return render_template('index.html', error_message='Movie not found', suggestions=suggestions)

    recs = get_recommendations(movie_idx, genre_filter=genre_filter, top_n=top_n)
    return render_template('index.html', selected_movie=MOVIES_DF.iloc[movie_idx], recommendations=recs)"""
    story.append(create_code_box(code4))

    story.append(PageBreak())

    # ==========================================
    # PAGE 6: Implementation Code 5 & Output Header + Fig 2
    # ==========================================
    story.append(Paragraph("5. Client Autocomplete API (app.py)", code_title_style))
    code5 = """@app.route('/api/movies', methods=['GET'])
def api_movies():
    all_titles = sorted(MOVIES_DF['title'].dropna().unique().tolist())
    all_genres = sorted(list(set(g for text in MOVIES_DF['genres_text'] for g in text.split())))
    return jsonify({'titles': all_titles, 'genres': all_genres})"""
    story.append(create_code_box(code5))

    story.append(Spacer(1, 14))
    story.append(Paragraph("Output (Step-by-Step Screenshots):", heading_style))

    fig2_path = os.path.join(REPORT_IMG_DIR, "fig2_load_input.png")
    if os.path.exists(fig2_path):
        story.append(Spacer(1, 6))
        story.append(Image(fig2_path, width=460, height=265))
        story.append(Paragraph("Fig 2: Step 1 &mdash; Load Input Movie: interface to select a movie or load a benchmark sample", caption_style))

    story.append(PageBreak())

    # ==========================================
    # PAGE 7: Fig 3 & Fig 4
    # ==========================================
    fig3_path = os.path.join(REPORT_IMG_DIR, "fig3_autocomplete.png")
    if os.path.exists(fig3_path):
        story.append(Image(fig3_path, width=460, height=255))
        story.append(Paragraph("Fig 3: Step 1 &mdash; Movie selection preview with searchable autocomplete and instant sample loader", caption_style))

    fig4_path = os.path.join(REPORT_IMG_DIR, "fig4_preprocessing.png")
    if os.path.exists(fig4_path):
        story.append(Spacer(1, 8))
        story.append(Image(fig4_path, width=460, height=255))
        story.append(Paragraph("Fig 4: Step 2 &mdash; Data preprocessing: raw TMDB dataset vs. cleaned & normalized (4,800 records)", caption_style))

    story.append(PageBreak())

    # ==========================================
    # PAGE 8: Fig 5 & Fig 6 (EDA Part 1)
    # ==========================================
    fig5_path = os.path.join(PLOTS_DIR, "genre_distribution.png")
    if os.path.exists(fig5_path):
        story.append(Image(fig5_path, width=460, height=250))
        story.append(Paragraph("Fig 5: Step 3 &mdash; Exploratory Data Analysis: frequency distribution of top 15 movie genres in TMDB 5000", caption_style))

    fig6_path = os.path.join(PLOTS_DIR, "release_year_distribution.png")
    if os.path.exists(fig6_path):
        story.append(Spacer(1, 8))
        story.append(Image(fig6_path, width=460, height=225))
        story.append(Paragraph("Fig 6: Step 3 &mdash; Exploratory Data Analysis: distribution of movies by release year (1920 &ndash; 2016)", caption_style))

    story.append(PageBreak())

    # ==========================================
    # PAGE 9: Fig 7 & Fig 8 (EDA Part 2 & TF-IDF)
    # ==========================================
    fig7_path = os.path.join(PLOTS_DIR, "popularity_distribution.png")
    if os.path.exists(fig7_path):
        story.append(Image(fig7_path, width=460, height=240))
        story.append(Paragraph("Fig 7: Step 4 &mdash; Exploratory Data Analysis: top 10 most popular movies in TMDB dataset", caption_style))

    fig8_path = os.path.join(REPORT_IMG_DIR, "fig8_tfidf_space.png")
    if os.path.exists(fig8_path):
        story.append(Spacer(1, 8))
        story.append(Image(fig8_path, width=460, height=245))
        story.append(Paragraph("Fig 8: Step 5 &mdash; Feature extraction: 10,000-dimensional TF-IDF feature space and term weight visualization", caption_style))

    story.append(PageBreak())

    # ==========================================
    # PAGE 10: Fig 9 & Fig 10 (Recommendations)
    # ==========================================
    fig9_path = os.path.join(REPORT_IMG_DIR, "fig8_rec_interstellar.png")
    if os.path.exists(fig9_path):
        story.append(Image(fig9_path, width=460, height=255))
        story.append(Paragraph("Fig 9: Step 6 &mdash; Recommendation result: &ldquo;Interstellar&rdquo; (Silent Running 23.1%, 2001: A Space Odyssey 21.7%)", caption_style))

    fig10_path = os.path.join(REPORT_IMG_DIR, "fig9_rec_darkknight.png")
    if os.path.exists(fig10_path):
        story.append(Spacer(1, 8))
        story.append(Image(fig10_path, width=460, height=255))
        story.append(Paragraph("Fig 10: Step 6 &mdash; Recommendation result: &ldquo;The Dark Knight&rdquo; (Batman Begins 49.3%, The Dark Knight Rises 48.2%)", caption_style))

    story.append(PageBreak())

    # ==========================================
    # PAGE 11: Fig 11 & Fig 12 + Result
    # ==========================================
    fig11_path = os.path.join(REPORT_IMG_DIR, "fig10_avengers_filtered.png")
    if os.path.exists(fig11_path):
        story.append(Image(fig11_path, width=460, height=215))
        story.append(Paragraph("Fig 11: Step 7 &mdash; Filtered recommendation result: &ldquo;The Avengers&rdquo; with Action & Sci-Fi genre constraint", caption_style))

    fig12_path = os.path.join(REPORT_IMG_DIR, "fig11_evaluation_metrics.png")
    if os.path.exists(fig12_path):
        story.append(Spacer(1, 4))
        story.append(Image(fig12_path, width=460, height=210))
        story.append(Paragraph("Fig 12: Step 8 &mdash; Performance evaluation metrics (Genre Overlap 92.80%, Baseline 40.80%, Gain +52.00%)", caption_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("Result:", heading_style))
    story.append(Paragraph(
        "A web-based Movie Recommendation System was successfully designed and implemented using a Flask backend, "
        "Scikit-learn TF-IDF Vectorizer, and Cosine Similarity. Movie metadata is preprocessed, a 10,000-dimensional "
        "feature vector is extracted from domain-weighted text (overview, genres, keywords, cast, and director), and "
        "the trained model calculates pairwise angular proximity to recommend top matching films with measurable confidence "
        "percentages. On an empirical evaluation benchmark of 50 query movies (500 recommendations), the model achieved a "
        "Genre Overlap Ratio of 92.80% (compared to 40.80% for an uninformed random baseline, achieving a net gain of +52.00%) "
        "and a mean Jaccard genre alignment of 0.6025, demonstrating excellent thematic relevance for college ML laboratory demonstration.",
        body_style
    ))

    # Build the document
    print("Building updated PDF...")
    doc.build(story)
    print("PDF build complete:", PDF_OUTPUT_PATH)
    print("File size (MB):", round(os.path.getsize(PDF_OUTPUT_PATH) / (1024 * 1024), 2))


if __name__ == "__main__":
    build_pdf()
