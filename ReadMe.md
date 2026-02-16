# 🎥 Movie Recommender System
This is a Movie Recommender System built using Streamlit, Flask and The Movie Database (TMDB) API.\
It provides movie recommendations based on similarity scores between movies, and it fetches and displays movie posters and details from the TMDB API.

[Dataset Link](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

## 🔍 Project Overview
The Movie Recommender System recommends similar movies based on user input. The system:

- Takes a movie name as input from the user.
- Computes similarity between movies using a content-based model (bag-of-words + cosine similarity).
- Fetches movie posters and metadata (title, year, rating, overview) from the TMDB API.
- Displays recommendations in an interactive web UI (Streamlit or Flask).

## 🎯 Features
- **Multi‑page Flask web app**
  - Home page with hero section, quick search and trending movies.
  - Recommendation page with search, fuzzy matching, controls and result grid.
  - Movie details page with poster, year, rating, overview and similar titles.
  - Analytics dashboard page with dataset statistics and charts.
  - User profile page with recently viewed movies and simple suggestions.
- **Content-based similarity :** Recommends movies based on textual tags and cosine similarity.
- **Interactive UIs :** 
  - Streamlit app for quick experimentation.
  - Flask app with a modern dark UI and card-based layout.
- **Search & fuzzy match :** Type a movie name (case-insensitive), the app finds the closest title.
- **Configurable results :** Choose how many similar movies to see (3–10).
- **Random discovery :** Get recommendations for a random movie with one click.
- **TMDB integration :** Shows poster, year, rating and a short overview for each recommendation.

## 🛠️ Technologies Used
- **Python :** Core language for developing the recommender system.
- **Streamlit :** For the original interactive UI.
- **Flask :** For the simple Python web application with a realistic movie UI.
- **Pandas :** For data manipulation and loading the movie dataset.
- **scikit-learn :** For vectorization and cosine similarity.
- **Pickle :** For loading/saving similarity scores and movie metadata.
- **Requests :** For making HTTP requests to the TMDb API.
- **TMDB API :** To fetch movie posters and details.

## 📦 Installation
To set up the project locally, follow these steps:
1. **Clone the repository :**
```bash
git clone https://github.com/ArpanSurin/Movie-Recommender-System.git
cd Movie-Recommender-System
```

2. **Create a virtual environment (optional but recommended) :**
```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

3. **Install the required dependencies :**
```bash
pip install -r requirements.txt
```

4. **Get your TMDB API Key :**

- Sign up on The Movie Database (TMDB) to get an API key.
- Replace the placeholder `api_key` in the code with your API key.

Run the Streamlit app:

```bash
streamlit run Movies/app.py
```

Or run the Flask web app:

```bash
python flask_app.py
```

## 🖥️ Usage
- **Select / search movie :**
  - In the Flask app, pick a movie from the list or type its name in the search box (search is case‑insensitive and uses fuzzy matching).
  - In the Streamlit app, choose from the dropdown.
- **Adjust results (Flask) :** Set how many similar movies you want to see (3–10).
- **Random mode (Flask) :** Click the `Random` button to discover movies based on a random title.
- **View recommendations :** See similar movies with poster, year, rating, similarity percentage and short overview.
- **Open movie details :** Click on a movie card to see a dedicated details page with a larger poster, overview and similar movies.
- **Analytics dashboard :** Open `/dashboard` in the Flask app to see total movies, tag statistics and simple charts (top tags, movies per year).
- **User profile :** Open `/profile` in the Flask app to see recently viewed movies for the current session and simple suggestions.

## ⚙️ How It Works
1. **Movie data :** The movies dataset is preprocessed into a compact dataframe (`movies_dict.pkl`) containing titles, IDs and tags.
2. **Text vectorization :** Tags are converted into vectors using `CountVectorizer` with English stop words removed.
3. **Cosine similarity :** A similarity matrix is computed once (and cached in `similarity.pkl`) using cosine similarity between movie vectors.
4. **Recommendation engine :** For a selected movie, the system finds the top-N most similar movies from this matrix.
5. **TMDB calls :** For each recommended movie, the TMDb API returns poster path, title, overview, release date and rating.
6. **Display :** The web app shows a clean grid of cards with poster, title, year, rating and a short overview.

## 👏 Acknowledgments
Special thanks to **TMDB** for providing the API and movie data.

---

## 🚀 Roadmap

The current implementation focuses on a content‑based recommender with a multi‑page Flask interface. Potential future enhancements include:

1. **Recommendation engine**
   - Add collaborative filtering (for example, matrix factorization).
   - Combine content-based and collaborative signals into a hybrid recommender.
   - Provide more detailed explanations for each recommendation.

2. **Filtering and ranking**
   - Genre, year range and minimum rating filters.
   - Toggles for top‑rated and trending titles.

3. **Data storage**
   - Replace pickle files with a relational database (SQLite or PostgreSQL).
   - Store user ratings, favorites and watchlist entries.

4. **User accounts**
   - Registration and login.
   - Persistent favorites and watchlists.
   - Personalised recommendations based on user activity.

5. **Analytics**
   - Additional charts (movies per year, genre distribution, rating distribution).
   - Popularity and engagement metrics.

6. **Infrastructure**
   - Refined project structure (separate routes, models, services and templates).
   - Environment‑based configuration for API keys and secrets.
   - Deployment configuration for platforms such as Render or similar.

7. **Performance**
   - Caching of similarity data and TMDB responses.
   - Graceful handling of missing or partial data.
