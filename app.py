import os
import pickle

import requests
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PLACEHOLDER_POSTER = "https://placehold.co/500x750?text=No+Poster"


@st.cache_resource
def load_model():
    movies = pickle.load(open("movies.pkl", "rb"))
    # Same vectoriser as model.ipynb; computing it here avoids shipping a ~180 MB similarity.pkl.
    vectors = CountVectorizer(max_features=5000, stop_words="english").fit_transform(movies["tags"])
    return movies.reset_index(drop=True), cosine_similarity(vectors)


def tmdb_api_key():
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        return os.environ.get("TMDB_API_KEY")


@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    key = tmdb_api_key()
    if not key:
        return PLACEHOLDER_POSTER
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={key}&language=en-US"
        poster_path = requests.get(url, timeout=10).json().get("poster_path")
        return f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else PLACEHOLDER_POSTER
    except requests.RequestException:
        return PLACEHOLDER_POSTER


def recommend(movie, movies, similarity, k=5):
    index = movies[movies["title"] == movie].index[0]
    distances = sorted(enumerate(similarity[index]), reverse=True, key=lambda x: x[1])[1 : k + 1]
    return [(movies.iloc[i].title, fetch_poster(movies.iloc[i].movie_id)) for i, _ in distances]


st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.header("🎬 Movie Recommender System")
st.caption("Content-based recommendations from plot, genres, keywords, cast and director (TMDB 5000 dataset).")

movies, similarity = load_model()
selected_movie = st.selectbox("Type or select a movie from the dropdown", movies["title"].values)

if st.button("Show Recommendation"):
    for col, (name, poster) in zip(st.columns(5), recommend(selected_movie, movies, similarity)):
        with col:
            st.image(poster)
            st.markdown(f"**{name}**")
