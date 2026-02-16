import streamlit as st
import pickle
import pandas as pd
import requests
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def fetch_poster(movie_id):
    response = requests.get(
        "https://api.themoviedb.org/3/movie/{}?api_key=fef9863a13e486ed5f253bad426b92e9&language=en-US".format(
            movie_id
        )
    )
    data = response.json()
    return "https://image.tmdb.org/t/p/w500" + data["poster_path"]


def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_movies_poster = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_poster.append(fetch_poster(movie_id))
    return recommended_movies, recommended_movies_poster


base_dir = os.path.dirname(os.path.abspath(__file__))

movies_dict_path = os.path.join(base_dir, "movies_dict.pkl")
movies_pkl_path = os.path.join(base_dir, "movies.pkl")
similarity_path = os.path.join(base_dir, "similarity.pkl")

if os.path.exists(movies_dict_path):
    movies_dict = pickle.load(open(movies_dict_path, "rb"))
elif os.path.exists(movies_pkl_path):
    movies_dict = pickle.load(open(movies_pkl_path, "rb"))
else:
    raise FileNotFoundError("movies_dict.pkl or movies.pkl not found in Movies directory")

movies = pd.DataFrame(movies_dict)

if os.path.exists(similarity_path):
    similarity = pickle.load(open(similarity_path, "rb"))
else:
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(movies["tags"]).toarray()
    similarity = cosine_similarity(vectors)
    with open(similarity_path, "wb") as f:
        pickle.dump(similarity, f)

st.title("Movie Recommender System")

selected_movie_name = st.selectbox(
    "What movies you want to watch today?",
    movies["title"].values,
)

if st.button("Recommend"):
    names, posters = recommend(selected_movie_name)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(names[0])
        st.image(posters[0])

    with col2:
        st.text(names[1])
        st.image(posters[1])

    with col3:
        st.text(names[2])
        st.image(posters[2])

    with col4:
        st.text(names[3])
        st.image(posters[3])

    with col5:
        st.text(names[4])
        st.image(posters[4])
