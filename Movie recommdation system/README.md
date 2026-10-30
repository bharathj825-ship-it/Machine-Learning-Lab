# Movie Recommendation System Using Content-Based Filtering and Cosine Similarity

An end-to-end Machine Learning Mini-Project implementing a content-based movie recommendation system using **TF-IDF Vectorization** and **Cosine Similarity** on the **TMDB 5000 Movie Dataset**.

This application is designed for college machine learning laboratory demonstrations and follows the standard machine learning development lifecycle from data preprocessing and exploratory data analysis to feature engineering, evaluation, and local web deployment using Flask.

---

## 1. Project Objective

The primary objective of this project is to build an unsupervised content-based recommendation engine that suggests movies sharing high thematic, narrative, and contextual similarity with a user-selected seed movie.

Unlike collaborative filtering systems (which require large volumes of user rating histories and suffer from the "cold start" problem for new items), content-based filtering leverages rich item metadata (overview, genres, keywords, cast, and director) to calculate geometric proximity in a vector space.

### Input Features
1. **Movie Title:** Selected by the user via an autocomplete search bar or quick-pick selector.
2. **Movie Genre:** Optional filter to isolate recommendations within a target category (e.g. Action, Sci-Fi, Drama).
3. **Number of Recommendations:** Configurable between Top 5 and Top 10 results.

### Output
A ranked list of candidate movies ordered by their **Cosine Similarity Score** (0.0 to 1.0) along with:
* Movie title & release year
* Genre badges
* Short overview / synopsis
* Director and top billed cast
* TMDB audience rating and relative popularity

---

## 2. Technology Stack

* **Programming Language:** Python 3.12
* **Web Framework:** Flask
* **Machine Learning & NLP:** Scikit-learn (`TfidfVectorizer`, `cosine_similarity`)
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Model Serialization:** Joblib
* **Frontend:** HTML5, CSS3, JavaScript (Cinematic Dark UI, Responsive Design)
* **Execution Environment:** 100% Local (zero external API keys or cloud services required)

---

## 3. Dataset Information & Setup

The project uses the **TMDB 5000 Movie Dataset**, a premier benchmark dataset containing extensive metadata for approximately 5,000 films.

### Dataset Files
* `data/tmdb_5000_movies.csv` (5.7 MB): Contains columns for `id`, `title`, `overview`, `genres`, `keywords`, `release_date`, `popularity`, `vote_average`, `vote_count`, etc.
* `data/tmdb_5000_credits.csv` (40.0 MB): Contains columns for `movie_id`, `title`, `cast`, and `crew`.
* `data/movies_processed.csv`: The cleaned and parsed dataset generated automatically during model training.

### Obtaining the Dataset
If cloning or setting up the repository from scratch:
1. Download `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` from [Kaggle TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) or public GitHub mirrors.
2. Place both CSV files directly into the `data/` directory:
   ```text
   MovieRecommendationSystem/
   └── data/
       ├── tmdb_5000_movies.csv
       └── tmdb_5000_credits.csv
   ```
3. Run `python train_model.py` to process the data and generate all artifacts.

---

## 4. Complete Machine Learning Workflow

The project implements and showcases all 12 stages of the machine learning pipeline:

```
[Problem Definition] 
         │
[Data Collection (TMDB 5000)]
         │
[Data Preprocessing (JSON Parsing, Nulls, Deduplication)]
         │
[Exploratory Data Analysis (EDA Visualizations)]
         │
[Feature Selection (Overview, Genres, Keywords, Cast, Director)]
         │
[Feature Engineering (Weighted combined_features)]
         │
[Validation Strategy (Query-Based Genre Overlap Testing)]
         │
[TF-IDF Vectorization (10,000 Unigram/Bigram Features)]
         │
[Cosine Similarity Modeling]
         │
[Artifact Serialization (models/movie_recommender.pkl)]
         │
[Quantitative Evaluation (92.8% Genre Overlap vs 40.8% Baseline)]
         │
[Flask Web Application & Interactive UI Delivery]
```

---

## 5. Data Preprocessing & Feature Engineering

### Preprocessing Steps
1. **JSON String Parsing:** Columns such as `genres`, `keywords`, `cast`, and `crew` are stored as JSON/literal string arrays in the raw TMDB CSVs. The parser safely deserializes these structures:
   * Extracts clean lists of genre and keyword names.
   * Extracts the top 3 billed actors from `cast`.
   * Identifies the member whose job is `"Director"` from `crew`.
2. **Missing Value Imputation:** Missing overviews, taglines, and release dates are handled safely without dropping valid records.
3. **Deduplication:** Duplicate movie records and titles are audited; duplicate titles are disambiguated by retaining the record with the highest popularity score.
4. **Text Normalization:** Text is lowercased, punctuation cleaned, and extra whitespace collapsed.
5. **Combined Features Construction:**
   ```python
   combined_features = (
       clean_overview + " " +
       clean_genres * 2 + " " +
       clean_keywords * 2 + " " +
       clean_cast + " " +
       clean_director * 2
   )
   ```
   * *Rationale for Weighting:* Genres and keywords are repeated twice to ensure thematic and tonal alignment carries equal importance relative to long plot descriptions.
   * Both natural names (`christopher nolan`) and joined tokens (`christophernolan`) are retained to maximize unigram and bigram capture.

---

## 6. Mathematical Foundations

### 1. TF-IDF (Term Frequency - Inverse Document Frequency)
TF-IDF measures how informative a word is to a particular movie within the entire movie corpus:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

$$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

* High frequency in a single movie plot combined with low frequency across other movies yields a high TF-IDF score (e.g., *wormhole*, *superhero*, *gotham*).
* Common generic words receive low IDF weights, preventing them from skewing similarity scores.

### 2. Cosine Similarity
Cosine similarity evaluates the angular distance between two non-zero vectors in the 10,000-dimensional TF-IDF space:

$$\text{Cosine Similarity}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

* Bounded in $[0.0, 1.0]$ for non-negative TF-IDF representations.
* Invariant to document length, ensuring movies with longer synopses are not artificially favored.

---

## 7. Validation Strategy & Empirical Evaluation

Because content-based recommendation is an unsupervised similarity matching task lacking ground-truth user preference labels, traditional classification metrics (like accuracy or F1-score) are inapplicable. 

Instead, a reproducible evaluation procedure is implemented:
1. **Query Sample:** 50 diverse popular movies across different genres are selected as test queries.
2. **Top-10 Generation:** The similarity engine generates 500 total recommendations (10 per query).
3. **Evaluation Metrics:**
   * **Average Top-10 Cosine Similarity:** `0.2518` (reflects tight thematic clustering).
   * **Genre Overlap Ratio:** `92.80%` of recommendations share at least one primary genre with the query movie.
   * **Mean Jaccard Genre Similarity:** `0.6025` average Jaccard intersection over union of genre sets.
   * **Baseline Comparison:** Compared against an uninformed random selection baseline (`40.80%` genre overlap), demonstrating a **+52.00% empirical improvement**.

### Evaluation Limitations
Content-based filtering verifies thematic and semantic closeness. It cannot determine subjective user enjoyment or quality preferences without personal rating histories.

---

## 8. Project Directory Structure

```text
MovieRecommendationSystem/
├── app.py                     # Flask web server and recommendation routing
├── train_model.py             # Complete training pipeline, EDA generation, evaluation & serialization
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive project documentation
├── data/
│   ├── tmdb_5000_movies.csv   # Raw TMDB movies dataset (5.7 MB)
│   ├── tmdb_5000_credits.csv  # Raw TMDB credits dataset (40.0 MB)
│   └── movies_processed.csv   # Cleaned 4,800-record dataset
├── models/
│   └── movie_recommender.pkl  # Serialized vectorizer, sparse matrix, and evaluation metrics (5.45 MB)
├── static/
│   ├── style.css              # Custom modern cinematic CSS stylesheet
│   └── plots/
│       ├── genre_distribution.png
│       ├── release_year_distribution.png
│       ├── popularity_distribution.png
│       ├── vote_average_distribution.png
│       ├── missing_values.png
│       └── recommendation_similarity.png
└── templates/
    └── index.html             # Main dashboard, interactive recommender, and workflow sections
```

---

## 9. Installation & Execution Guide

### Prerequisites
* Python 3.10+ (tested on Python 3.12)
* pip package manager

### Step 1: Install Required Packages
Navigate to the project root directory and install dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model and Generate Artifacts
Run the training script to preprocess data, generate EDA graphs, fit the TF-IDF vectorizer, and serialize the model:
```bash
python train_model.py
```

*Expected output:*
```text
======================================================================
STEP 1 & 2: DATA LOADING AND VALIDATION
======================================================================
Loading movies metadata... (4803, 20)
Loading credits metadata... (4803, 4)
...
STEP 3: DATA PREPROCESSING & FEATURE EXTRACTION
Preprocessing completed. Usable movies: 4800 / 4803
...
STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
Saved: static/plots/genre_distribution.png
Saved: static/plots/release_year_distribution.png
Saved: static/plots/popularity_distribution.png
Saved: static/plots/vote_average_distribution.png
Saved: static/plots/missing_values.png
...
STEP 5 & 6: TF-IDF VECTORIZATION & SIMILARITY MODELING
TF-IDF Matrix Shape: (4800, 10000)
...
STEP 7 & 8: MODEL VALIDATION AND EVALUATION
Genre Overlap Ratio: 92.80% (vs Baseline: 40.80%)
Saved: static/plots/recommendation_similarity.png
...
Saved model artifact: models/movie_recommender.pkl
Training and evaluation pipeline completed successfully!
```

### Step 3: Start the Flask Application
Run the local web server:
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 10. How to Use the Application

1. **Enter Movie Title:** Type any movie title into the search input. An autocomplete datalist helps select exact titles from the 4,800 available movies (or click any of the popular demo pills such as *Interstellar*, *Inception*, *The Dark Knight*, *Avatar*, *The Avengers*, or *Toy Story*).
2. **Select Optional Filters:** Optionally filter results by preferred genre or choose between Top 5 or Top 10 recommendations.
3. **Click "Recommend Movies":** The application retrieves the seed movie's TF-IDF vector, calculates cosine similarities against all other movies, excludes the query movie, ranks the candidates, and presents clean cards with similarity scores.
4. **Examine Evidence & Workflow:** Scroll down through the page to review the 12-step ML workflow, dataset statistics, Matplotlib/Seaborn EDA visualizations, mathematical formulas, and the college project record screenshot index.

---

## 11. Laboratory Record Screenshot Index

| Figure | Milestone | Description |
|---|---|---|
| **Fig. 1** | Dataset Collection | TMDB 5000 dataset preview, record count (4,803), columns, and data types. |
| **Fig. 2** | Data Preprocessing | Missing value resolution audit, duplicate checking, and `combined_features` string. |
| **Fig. 3** | Exploratory Data Analysis | Genre frequency bar chart, release year distribution, popularity, and ratings. |
| **Fig. 4** | Feature Selection & TF-IDF | TF-IDF formulation, 10,000-dimensional matrix, and n-gram parameters. |
| **Fig. 5** | Similarity Calculation & Evaluation | Cosine similarity formula, evaluation results (92.8% overlap), and similarity plot. |
| **Fig. 6** | Movie Recommendation Results | Query movie details, recommended movie cards, genres, and similarity scores. |
| **Fig. 7** | Final Application Interface | Full working web application interface with responsive layout. |

---

## 12. Important Academic Note

This system is an **unsupervised content-based recommendation model**. It identifies similarity based strictly on film descriptions, genres, keywords, cast, and directors. It does **not** learn individual user taste profiles unless integrated with user ratings or implicit feedback loops (collaborative filtering).
