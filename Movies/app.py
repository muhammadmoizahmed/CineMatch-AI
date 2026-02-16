import streamlit as st
import pickle
import pandas as pd
import requests
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
from collections import Counter


def fetch_poster(movie_id):
    response = requests.get(
        "https://api.themoviedb.org/3/movie/{}?api_key=fef9863a13e486ed5f253bad426b92e9&language=en-US".format(
            movie_id
        )
    )
    data = response.json()
    return "https://image.tmdb.org/t/p/w500" + data["poster_path"]


def fetch_movie_details(movie_id):
    response = requests.get(
        "https://api.themoviedb.org/3/movie/{}?api_key=fef9863a13e486ed5f253bad426b92e9&language=en-US".format(
            movie_id
        )
    )
    data = response.json()
    title = data.get("title")
    overview = data.get("overview") or ""
    release_date = data.get("release_date") or ""
    year = release_date.split("-")[0] if release_date else ""
    rating = data.get("vote_average")
    poster_path = data.get("poster_path")
    poster = (
        "https://image.tmdb.org/t/p/w500" + poster_path
        if poster_path
        else fetch_poster(movie_id)
    )
    return {
        "title": title,
        "overview": overview,
        "year": year,
        "rating": rating,
        "poster": poster,
    }


def recommend(movie_title, top_n=6):
    movie_index = movies[movies["title"] == movie_title].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1 : top_n + 1]

    recommendations = []
    for idx, score in movies_list:
        row = movies.iloc[idx]
        movie_id = int(row.movie_id)
        details = fetch_movie_details(movie_id)
        overview = details["overview"]
        if len(overview) > 180:
            overview = overview[:180] + "..."
        recommendations.append(
            {
                "movie_id": movie_id,
                "title": details["title"] or row.title,
                "year": details["year"],
                "poster": details["poster"],
                "rating": details["rating"],
                "similarity": float(score),
                "overview": overview,
            }
        )
    return recommendations


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

st.set_page_config(page_title="Movie Recommender", layout="wide")

page = st.sidebar.radio("Navigation", ["Home", "Recommendations", "Analytics", "Profile"])

if page == "Home":
    st.title("Movie Recommender")
    st.write(
        "Discover similar movies using a content-based recommendation engine powered by TMDB."
    )
    search_query = st.text_input("Quick search", "")
    top_n_home = st.slider("Number of results", 3, 10, 6, key="home_top_n")
    titles = movies["title"].values
    selected_title = None
    if st.button("Search", key="home_search") and search_query:
        titles_lower = [t.lower() for t in titles]
        matches = difflib.get_close_matches(
            search_query.lower(), titles_lower, n=1, cutoff=0.3
        )
        if matches:
            match = matches[0]
            idx = titles_lower.index(match)
            selected_title = titles[idx]
    if not selected_title:
        selected_title = titles[0]
    recs = recommend(selected_title, top_n=top_n_home)
    st.subheader("Trending and similar titles")
    cols = st.columns(4)
    for i, rec in enumerate(recs):
        col = cols[i % 4]
        with col:
            st.image(rec["poster"], use_container_width=True)
            st.markdown(
                f"**{rec['title']}**"
                + (f" ({rec['year']})" if rec["year"] else "")
            )
            meta = []
            if rec["rating"] is not None:
                meta.append(f"★ {rec['rating']:.1f}")
            meta.append(f"Sim {rec['similarity']*100:.0f}%")
            st.caption(" · ".join(meta))
            if rec["overview"]:
                st.write(rec["overview"])

elif page == "Recommendations":
    st.title("Recommendations")
    titles = movies["title"].values
    search_query = st.text_input("Movie name", "")
    selected_movie = st.selectbox("Or pick from the list", titles)
    top_n = st.slider("Number of results", 3, 10, 6)
    col_a, col_b = st.columns(2)
    recommend_clicked = col_a.button("Recommend", key="rec_btn")
    random_clicked = col_b.button("Random", key="rec_random")

    if recommend_clicked or random_clicked:
        movie_title = selected_movie
        if random_clicked:
            movie_title = movies.sample(1)["title"].values[0]
        elif search_query:
            titles_lower = [t.lower() for t in titles]
            matches = difflib.get_close_matches(
                search_query.lower(), titles_lower, n=1, cutoff=0.3
            )
            if matches:
                match = matches[0]
                idx = titles_lower.index(match)
                movie_title = titles[idx]
        st.subheader(f"Selected: {movie_title}")
        try:
            recs = recommend(movie_title, top_n=top_n)
            cols = st.columns(4)
            for i, rec in enumerate(recs):
                col = cols[i % 4]
                with col:
                    st.image(rec["poster"], use_container_width=True)
                    st.markdown(
                        f"**{rec['title']}**"
                        + (f" ({rec['year']})" if rec["year"] else "")
                    )
                    meta = []
                    if rec["rating"] is not None:
                        meta.append(f"★ {rec['rating']:.1f}")
                    meta.append(f"Sim {rec['similarity']*100:.0f}%")
                    st.caption(" · ".join(meta))
                    if rec["overview"]:
                        st.write(rec["overview"])
        except Exception as exc:
            st.error(str(exc))

elif page == "Analytics":
    st.title("Analytics")
    total_movies = int(len(movies))
    tags = []
    if "tags" in movies.columns:
        for t in movies["tags"].dropna():
            tags.extend(str(t).split())
    unique_tags = len(set(tags)) if tags else 0
    col1, col2 = st.columns(2)
    col1.metric("Total movies", total_movies)
    col2.metric("Unique tags", unique_tags)

    if tags:
        top_tags = Counter(tags).most_common(10)
        tag_df = pd.DataFrame(top_tags, columns=["tag", "count"])
        st.subheader("Top tags")
        st.bar_chart(tag_df.set_index("tag"))

    year_col = None
    if "release_date" in movies.columns:
        year_col = movies["release_date"].astype(str).str[:4]
    elif "year" in movies.columns:
        year_col = movies["year"].astype(str)
    if year_col is not None:
        year_counts = year_col.value_counts().sort_index()
        year_df = pd.DataFrame(
            {"year": year_counts.index, "count": year_counts.values}
        )
        st.subheader("Movies per year")
        st.line_chart(year_df.set_index("year"))

elif page == "Profile":
    st.title("User profile")
    st.write(
        "This page shows simple suggestions based on a base title. "
        "A full account system with login, favorites and watchlist can be added on top of this flow."
    )
    base_title = movies["title"].values[0]
    recs = recommend(base_title, top_n=8)
    cols = st.columns(4)
    for i, rec in enumerate(recs):
        col = cols[i % 4]
        with col:
            st.image(rec["poster"], use_container_width=True)
            st.markdown(
                f"**{rec['title']}**"
                + (f" ({rec['year']})" if rec["year"] else "")
            )
            if rec["rating"] is not None:
                st.caption(f"★ {rec['rating']:.1f}")
