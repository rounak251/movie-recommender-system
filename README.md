# 🎬 Movie Recommender System

A content-based movie recommender built on the [TMDB 5000 dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata), with a Streamlit web app that shows the 5 most similar movies and their posters.

## How it works
1. **Data:** merge the TMDB movies and credits tables (about 4,800 movies).
2. **Features:** for each movie, combine the plot overview, genres, keywords, top 3 cast members and the director into one `tags` string. Spaces inside names are removed (`Sam Worthington` → `SamWorthington`) so each name stays one token. Words are lower-cased and Porter-stemmed with NLTK.
3. **Vectors:** `CountVectorizer` keeps the 5,000 most frequent words, ignoring English stop words.
4. **Similarity:** cosine similarity between every pair of movie vectors. A recommendation is the 5 nearest movies.

The full pipeline is in [`model.ipynb`](model.ipynb). It exports `movies.pkl`, which the app loads.

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```
Posters come from the TMDB API. Get a free key at <https://www.themoviedb.org/settings/api> and either
- set it as an environment variable: `export TMDB_API_KEY=your_key`, or
- copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill it in.

Without a key, the app still works and shows placeholder posters.

To re-run the notebook, download `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` from Kaggle into this folder. They aren't committed because they're about 45 MB.

## Tech stack
Python · pandas · scikit-learn · NLTK · Streamlit · TMDB API
