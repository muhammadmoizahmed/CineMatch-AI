from flask import Flask, render_template_string, request, session
import pickle
import os
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import json
import difflib


base_dir = os.path.dirname(os.path.abspath(__file__))
movies_dir = os.path.join(base_dir, "Movies")

movies_dict_path = os.path.join(movies_dir, "movies_dict.pkl")
movies_pkl_path = os.path.join(movies_dir, "movies.pkl")
similarity_path = os.path.join(movies_dir, "similarity.pkl")


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


app = Flask(__name__)
app.secret_key = "dev-secret-key"


HOME_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Movie Recommender - Home</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
  <style>
    body {
      background: radial-gradient(circle at top, #1f2933 0, #050816 55%);
      color: #f9fafb;
    }
    .card img {
      object-fit: cover;
      height: 260px;
    }
    .card {
      border-radius: 0.75rem;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card:hover {
      transform: translateY(-4px);
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.5);
    }
  </style>
</head>
<body>
  <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
    <div class="container">
      <a class="navbar-brand" href="/">MovieRS</a>
      <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="navbarNav">
        <ul class="navbar-nav ms-auto">
          <li class="nav-item"><a class="nav-link active" href="/">Home</a></li>
          <li class="nav-item"><a class="nav-link" href="/recommend">Recommend</a></li>
          <li class="nav-item"><a class="nav-link" href="/dashboard">Analytics</a></li>
          <li class="nav-item"><a class="nav-link" href="/profile">Profile</a></li>
        </ul>
      </div>
    </div>
  </nav>

  <div class="container py-4">
    <div class="row align-items-center mb-5">
      <div class="col-md-7">
        <h1 class="fw-bold mb-3">Discover movies you’ll actually enjoy</h1>
        <p class="text-secondary mb-4">
          Content-based recommendations powered by TMDB. Search, filter and explore similar titles
          with posters, ratings and overviews, all in a modern dark UI.
        </p>
        <a href="/recommend" class="btn btn-primary btn-lg me-2">Start recommending</a>
        <a href="/dashboard" class="btn btn-outline-light btn-lg">View analytics</a>
      </div>
      <div class="col-md-5">
        <div class="bg-secondary rounded-4 p-3 shadow">
          <h5 class="mb-3">Quick search</h5>
          <form method="get" action="/recommend" class="row g-2">
            <div class="col-8">
              <input type="text" class="form-control" name="query" placeholder="Type a movie name">
            </div>
            <div class="col-4">
              <button class="btn btn-info w-100" type="submit">Go</button>
            </div>
          </form>
        </div>
      </div>
    </div>

    {% if trending %}
      <h3 class="mb-3">Trending picks</h3>
      <div class="row">
        {% for rec in trending %}
          <div class="col-lg-3 col-md-4 col-sm-6 col-12 mb-4">
            <div class="card bg-secondary text-light h-100 position-relative">
              <img src="{{ rec.poster }}" class="card-img-top" alt="{{ rec.title }}">
              <div class="card-body">
                <h6 class="card-title mb-1">
                  <a href="/movie/{{ rec.movie_id }}" class="stretched-link text-decoration-none text-light">
                    {{ rec.title }}{% if rec.year %} <span class="text-muted">({{ rec.year }})</span>{% endif %}
                  </a>
                </h6>
                <div class="d-flex justify-content-between align-items-center small mb-1">
                  {% if rec.rating is not none %}
                    <span class="text-warning">★ {{ '{:.1f}'.format(rec.rating) }}</span>
                  {% endif %}
                  <span class="text-info">Sim: {{ '{:.0f}'.format(rec.similarity * 100) }}%</span>
                </div>
                {% if rec.overview %}
                  <p class="small mb-0">{{ rec.overview }}</p>
                {% endif %}
              </div>
            </div>
          </div>
        {% endfor %}
      </div>
    {% endif %}
  </div>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""


DETAIL_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{{ movie.title }} - Details</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
  <style>
    body {
      background: radial-gradient(circle at top, #1f2933 0, #050816 55%);
      color: #f9fafb;
    }
  </style>
</head>
<body>
  <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
    <div class="container">
      <a class="navbar-brand" href="/">MovieRS</a>
      <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="navbarNav">
        <ul class="navbar-nav ms-auto">
          <li class="nav-item"><a class="nav-link" href="/">Home</a></li>
          <li class="nav-item"><a class="nav-link" href="/recommend">Recommend</a></li>
          <li class="nav-item"><a class="nav-link" href="/dashboard">Analytics</a></li>
          <li class="nav-item"><a class="nav-link" href="/profile">Profile</a></li>
        </ul>
      </div>
    </div>
  </nav>

  <div class="container py-4">
    <a href="/recommend" class="btn btn-outline-light btn-sm mb-3">← Back to recommendations</a>
    <div class="row mb-4">
      <div class="col-md-4">
        <img src="{{ movie.poster }}" class="img-fluid rounded-4 shadow" alt="{{ movie.title }}">
      </div>
      <div class="col-md-8">
        <h1 class="fw-bold mb-2">
          {{ movie.title }}{% if movie.year %} <span class="text-muted">({{ movie.year }})</span>{% endif %}
        </h1>
        {% if movie.rating is not none %}
          <p class="mb-2 text-warning">★ {{ '{:.1f}'.format(movie.rating) }}</p>
        {% endif %}
        {% if movie.overview %}
          <p class="mb-3 text-secondary">{{ movie.overview }}</p>
        {% endif %}
      </div>
    </div>

    {% if similar %}
      <h4 class="mb-3">Similar movies</h4>
      <div class="row">
        {% for rec in similar %}
          <div class="col-lg-3 col-md-4 col-sm-6 col-12 mb-4">
            <div class="card bg-secondary text-light h-100 position-relative">
              <img src="{{ rec.poster }}" class="card-img-top" alt="{{ rec.title }}">
              <div class="card-body">
                <h6 class="card-title mb-1">
                  <a href="/movie/{{ rec.movie_id }}" class="stretched-link text-decoration-none text-light">
                    {{ rec.title }}{% if rec.year %} <span class="text-muted">({{ rec.year }})</span>{% endif %}
                  </a>
                </h6>
                <div class="d-flex justify-content-between align-items-center small mb-1">
                  {% if rec.rating is not none %}
                    <span class="text-warning">★ {{ '{:.1f}'.format(rec.rating) }}</span>
                  {% endif %}
                  <span class="text-info">Sim: {{ '{:.0f}'.format(rec.similarity * 100) }}%</span>
                </div>
                {% if rec.overview %}
                  <p class="small mb-0">{{ rec.overview }}</p>
                {% endif %}
              </div>
            </div>
          </div>
        {% endfor %}
      </div>
    {% endif %}
  </div>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""


DASHBOARD_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Analytics Dashboard</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
  <style>
    body {
      background: radial-gradient(circle at top, #1f2933 0, #050816 55%);
      color: #f9fafb;
    }
    .card {
      border-radius: 0.75rem;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
  </style>
</head>
<body>
  <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
    <div class="container">
      <a class="navbar-brand" href="/">MovieRS</a>
      <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="navbarNav">
        <ul class="navbar-nav ms-auto">
          <li class="nav-item"><a class="nav-link" href="/">Home</a></li>
          <li class="nav-item"><a class="nav-link" href="/recommend">Recommend</a></li>
          <li class="nav-item"><a class="nav-link active" href="/dashboard">Analytics</a></li>
          <li class="nav-item"><a class="nav-link" href="/profile">Profile</a></li>
        </ul>
      </div>
    </div>
  </nav>

  <div class="container py-4">
    <h1 class="fw-bold mb-4">Analytics dashboard</h1>
    <div class="row mb-4">
      <div class="col-md-4">
        <div class="card bg-secondary text-light">
          <div class="card-body">
            <h5>Total movies</h5>
            <p class="display-6">{{ stats.total_movies }}</p>
          </div>
        </div>
      </div>
      <div class="col-md-4">
        <div class="card bg-secondary text-light">
          <div class="card-body">
            <h5>Unique tags</h5>
            <p class="display-6">{{ stats.unique_tags }}</p>
          </div>
        </div>
      </div>
      <div class="col-md-4">
        <div class="card bg-secondary text-light">
          <div class="card-body">
            <h5>Sample titles</h5>
            <p class="mb-0 small">{{ stats.sample_titles }}</p>
          </div>
        </div>
      </div>
    </div>

    <div class="row">
      <div class="col-lg-6 mb-4">
        <div class="card bg-secondary text-light">
          <div class="card-body">
            <h5 class="card-title">Top tag frequency</h5>
            <canvas id="tagsChart" height="220"></canvas>
          </div>
        </div>
      </div>
      <div class="col-lg-6 mb-4">
        <div class="card bg-secondary text-light">
          <div class="card-body">
            <h5 class="card-title">Movies per year</h5>
            <canvas id="yearChart" height="220"></canvas>
          </div>
        </div>
      </div>
    </div>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script>
    const tagLabels = {{ stats.tag_labels|safe }};
    const tagCounts = {{ stats.tag_counts|safe }};
    const yearLabels = {{ stats.year_labels|safe }};
    const yearCounts = {{ stats.year_counts|safe }};

    if (tagLabels.length && document.getElementById('tagsChart')) {
      new Chart(document.getElementById('tagsChart'), {
        type: 'bar',
        data: {
          labels: tagLabels,
          datasets: [{
            label: 'Count',
            backgroundColor: '#38bdf8',
            data: tagCounts
          }]
        },
        options: {
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: '#e5e7eb' } },
            y: { ticks: { color: '#e5e7eb' } }
          }
        }
      });
    }

    if (yearLabels.length && document.getElementById('yearChart')) {
      new Chart(document.getElementById('yearChart'), {
        type: 'line',
        data: {
          labels: yearLabels,
          datasets: [{
            label: 'Movies',
            borderColor: '#a855f7',
            backgroundColor: 'rgba(168, 85, 247, 0.2)',
            fill: true,
            tension: 0.3,
            data: yearCounts
          }]
        },
        options: {
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: '#e5e7eb' } },
            y: { ticks: { color: '#e5e7eb' } }
          }
        }
      });
    }
  </script>
</body>
</html>
"""


PROFILE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>User Profile</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
  <style>
    body {
      background: radial-gradient(circle at top, #1f2933 0, #050816 55%);
      color: #f9fafb;
    }
  </style>
</head>
<body>
  <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
    <div class="container">
      <a class="navbar-brand" href="/">MovieRS</a>
      <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="navbarNav">
        <ul class="navbar-nav ms-auto">
          <li class="nav-item"><a class="nav-link" href="/">Home</a></li>
          <li class="nav-item"><a class="nav-link" href="/recommend">Recommend</a></li>
          <li class="nav-item"><a class="nav-link" href="/dashboard">Analytics</a></li>
          <li class="nav-item"><a class="nav-link active" href="/profile">Profile</a></li>
        </ul>
      </div>
    </div>
  </nav>

  <div class="container py-4">
    <h1 class="fw-bold mb-3">User profile</h1>
    <p class="text-secondary">
      This page shows your recently viewed movies and simple personalized suggestions.
      A full login, favorites and watchlist system can be added on top of this flow.
    </p>

    {% if recent_movies %}
      <h4 class="mt-4 mb-3">Recently viewed</h4>
      <div class="row">
        {% for rec in recent_movies %}
          <div class="col-lg-3 col-md-4 col-sm-6 col-12 mb-4">
            <div class="card bg-secondary text-light h-100 position-relative">
              <img src="{{ rec.poster }}" class="card-img-top" alt="{{ rec.title }}">
              <div class="card-body">
                <h6 class="card-title mb-1">
                  <a href="/movie/{{ rec.movie_id }}" class="stretched-link text-decoration-none text-light">
                    {{ rec.title }}{% if rec.year %} <span class="text-muted">({{ rec.year }})</span>{% endif %}
                  </a>
                </h6>
                {% if rec.rating is not none %}
                  <p class="small text-warning mb-0">★ {{ '{:.1f}'.format(rec.rating) }}</p>
                {% endif %}
              </div>
            </div>
          </div>
        {% endfor %}
      </div>
    {% endif %}

    {% if suggested_movies %}
      <h4 class="mt-4 mb-3">Suggested for you</h4>
      <div class="row">
        {% for rec in suggested_movies %}
          <div class="col-lg-3 col-md-4 col-sm-6 col-12 mb-4">
            <div class="card bg-secondary text-light h-100 position-relative">
              <img src="{{ rec.poster }}" class="card-img-top" alt="{{ rec.title }}">
              <div class="card-body">
                <h6 class="card-title mb-1">
                  <a href="/movie/{{ rec.movie_id }}" class="stretched-link text-decoration-none text-light">
                    {{ rec.title }}{% if rec.year %} <span class="text-muted">({{ rec.year }})</span>{% endif %}
                  </a>
                </h6>
                {% if rec.rating is not none %}
                  <p class="small text-warning mb-0">★ {{ '{:.1f}'.format(rec.rating) }}</p>
                {% endif %}
              </div>
            </div>
          </div>
        {% endfor %}
      </div>
    {% endif %}
  </div>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""


PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Movie Recommender</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
  <style>
    body {
      background: radial-gradient(circle at top, #1f2933 0, #050816 55%);
      color: #f9fafb;
    }
    .card img {
      object-fit: cover;
      height: 260px;
    }
    .card {
      border-radius: 0.75rem;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card:hover {
      transform: translateY(-4px);
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.5);
    }
  </style>
</head>
<body>
  <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
    <div class="container">
      <a class="navbar-brand" href="/">MovieRS</a>
      <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="navbarNav">
        <ul class="navbar-nav ms-auto">
          <li class="nav-item"><a class="nav-link" href="/">Home</a></li>
          <li class="nav-item"><a class="nav-link active" href="/recommend">Recommend</a></li>
          <li class="nav-item"><a class="nav-link" href="/dashboard">Analytics</a></li>
          <li class="nav-item"><a class="nav-link" href="/profile">Profile</a></li>
        </ul>
      </div>
    </div>
  </nav>

  <div class="container py-4">
    <div class="d-flex flex-column flex-md-row justify-content-between align-items-md-center mb-4">
      <div>
        <h1 class="fw-bold mb-1">Movie Recommender System</h1>
        <p class="text-secondary mb-0">Choose a movie, tweak options, and discover similar titles.</p>
      </div>
      <span class="badge bg-info text-dark mt-3 mt-md-0">Powered by TMDB + Content Similarity</span>
    </div>

    <form method="get" class="row gy-3 gx-3 align-items-end mb-4">
      <div class="col-md-5">
        <label class="form-label">Search movie name</label>
        <input
          type="text"
          name="query"
          class="form-control"
          placeholder="Type movie name (e.g. Avatar)"
          value="{{ search_query }}"
        >
      </div>
      <div class="col-md-4">
        <label class="form-label">Or pick from the list</label>
        <select name="movie" class="form-select">
          {% for title in movies %}
            <option value="{{ title }}" {% if title == selected_movie %}selected{% endif %}>{{ title }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="col-md-2">
        <label class="form-label">How many results?</label>
        <input
          type="number"
          name="n"
          class="form-control"
          min="3"
          max="10"
          value="{{ top_n }}"
        >
      </div>
      <div class="col-md-1 d-grid gap-2">
        <button type="submit" class="btn btn-primary">Go</button>
        <button type="submit" name="random" value="1" class="btn btn-outline-warning">Random</button>
      </div>
    </form>

    {% if selected_movie %}
      <div class="mb-4">
        <h5 class="mb-1">Selected movie: <span class="text-info">{{ selected_movie }}</span></h5>
      </div>
    {% endif %}

    {% if error %}
      <div class="alert alert-danger" role="alert">
        {{ error }}
      </div>
    {% endif %}

    {% if recommendations %}
      <div class="row">
        {% for rec in recommendations %}
          <div class="col-lg-3 col-md-4 col-sm-6 col-12 mb-4">
            <div class="card bg-secondary text-light h-100 position-relative">
              <img src="{{ rec.poster }}" class="card-img-top" alt="{{ rec.title }}">
              <div class="card-body">
                <h6 class="card-title mb-1">
                  <a href="/movie/{{ rec.movie_id }}" class="stretched-link text-decoration-none text-light">
                    {{ rec.title }}{% if rec.year %} <span class="text-muted">({{ rec.year }})</span>{% endif %}
                  </a>
                </h6>
                <div class="d-flex justify-content-between align-items-center small mb-1">
                  {% if rec.rating is not none %}
                    <span class="text-warning">★ {{ '{:.1f}'.format(rec.rating) }}</span>
                  {% endif %}
                  <span class="text-info">Sim: {{ '{:.0f}'.format(rec.similarity * 100) }}%</span>
                </div>
                {% if rec.overview %}
                  <p class="small mb-0">{{ rec.overview }}</p>
                {% endif %}
              </div>
            </div>
          </div>
        {% endfor %}
      </div>
    {% endif %}
  </div>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    base_title = movies["title"].values[0]
    try:
        trending = recommend(base_title, top_n=8)
    except Exception:
        trending = []
    return render_template_string(HOME_TEMPLATE, trending=trending)


@app.route("/recommend", methods=["GET"])
def index():
    search_query = (request.args.get("query") or "").strip()
    top_n_default = 6
    top_n = top_n_default
    top_n_param = request.args.get("n")
    if top_n_param:
        try:
            top_n_val = int(top_n_param)
            if 3 <= top_n_val <= 10:
                top_n = top_n_val
        except ValueError:
            top_n = top_n_default

    titles = movies["title"].values
    selected_movie = request.args.get("movie") or titles[0]

    if request.args.get("random"):
        selected_movie = movies.sample(1)["title"].values[0]
    elif search_query:
        titles_lower = [t.lower() for t in titles]
        matches = difflib.get_close_matches(
            search_query.lower(), titles_lower, n=1, cutoff=0.3
        )
        if matches:
            match = matches[0]
            match_index = titles_lower.index(match)
            selected_movie = titles[match_index]

    recommendations = None
    error = None

    try:
        recommendations = recommend(selected_movie, top_n=top_n)
    except Exception as exc:
        error = str(exc)

    return render_template_string(
        PAGE_TEMPLATE,
        movies=movies["title"].values,
        selected_movie=selected_movie,
        recommendations=recommendations,
        error=error,
        top_n=top_n,
        search_query=search_query,
    )


@app.route("/movie/<int:movie_id>", methods=["GET"])
def movie_detail(movie_id):
    details = fetch_movie_details(movie_id)
    overview = details["overview"]
    if len(overview) > 220:
        overview = overview[:220] + "..."
    movie = {
        "title": details["title"],
        "year": details["year"],
        "rating": details["rating"],
        "poster": details["poster"],
        "overview": overview,
    }
    recent = session.get("recent_ids", [])
    if movie_id in recent:
        recent.remove(movie_id)
    recent.insert(0, movie_id)
    session["recent_ids"] = recent[:12]
    try:
        title_match = movies[movies["movie_id"] == movie_id]["title"].values[0]
        similar = recommend(title_match, top_n=6)
    except Exception:
        similar = []
    return render_template_string(DETAIL_TEMPLATE, movie=movie, similar=similar)


@app.route("/dashboard", methods=["GET"])
def dashboard():
    total_movies = int(len(movies))
    sample_titles = movies["title"].head(5).tolist()
    tags = []
    if "tags" in movies.columns:
        for t in movies["tags"].dropna():
            tags.extend(str(t).split())
    unique_tags = len(set(tags)) if tags else 0

    from collections import Counter

    top_tags = Counter(tags).most_common(8) if tags else []
    tag_labels = [t[0] for t in top_tags]
    tag_counts = [t[1] for t in top_tags]

    year_col = None
    if "release_date" in movies.columns:
        year_col = movies["release_date"].astype(str).str[:4]
    elif "year" in movies.columns:
        year_col = movies["year"].astype(str)

    year_labels = []
    year_counts = []
    if year_col is not None:
        year_counts_series = year_col.value_counts().sort_index()
        year_labels = list(year_counts_series.index)
        year_counts = list(year_counts_series.values)

    stats = type(
        "Stats",
        (),
        {
            "total_movies": total_movies,
            "unique_tags": unique_tags,
            "sample_titles": ", ".join(sample_titles),
            "tag_labels": json.dumps(tag_labels),
            "tag_counts": json.dumps(tag_counts),
            "year_labels": json.dumps(year_labels),
            "year_counts": json.dumps(year_counts),
        },
    )()
    return render_template_string(DASHBOARD_TEMPLATE, stats=stats)


@app.route("/profile", methods=["GET"])
def profile():
    recent_ids = session.get("recent_ids", [])
    recent_movies = []
    if recent_ids and "movie_id" in movies.columns:
        subset = movies[movies["movie_id"].isin(recent_ids)]
        for mid in recent_ids:
            row = subset[subset["movie_id"] == mid]
            if row.empty:
                continue
            movie_id = int(mid)
            details = fetch_movie_details(movie_id)
            recent_movies.append(
                {
                    "movie_id": movie_id,
                    "title": details["title"] or row.iloc[0]["title"],
                    "year": details["year"],
                    "poster": details["poster"],
                    "rating": details["rating"],
                }
            )
    suggested_movies = []
    if not recent_movies:
        try:
            base_title = movies["title"].values[0]
            suggested_movies = recommend(base_title, top_n=8)
        except Exception:
            suggested_movies = []
    return render_template_string(
        PROFILE_TEMPLATE,
        recent_movies=recent_movies,
        suggested_movies=suggested_movies,
    )


if __name__ == "__main__":
    app.run(debug=True)
