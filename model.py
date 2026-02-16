import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- LOAD DATA ----------------

ratings = pd.read_csv(
    "u.data",
    sep="\t",
    names=["userId", "movieId", "rating", "timestamp"]
)

genre_cols = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

movies = pd.read_csv(
    "u.item",
    sep="|",
    encoding="latin-1",
    names=[
        "movieId", "title", "release_date", "video_release_date",
        "IMDb_URL"
    ] + genre_cols
)

movies_df = movies[["movieId", "title"] + genre_cols].copy()

# ---------------- USER ITEM MATRIX ----------------

user_item_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

# ---------------- USER SIMILARITY ----------------

user_similarity = cosine_similarity(user_item_matrix)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

# ---------------- COLLABORATIVE SCORE ----------------

def get_collaborative_scores(user_id):

    if user_id not in user_item_matrix.index:
        return {}

    similar_users = (
        user_similarity_df[user_id]
        .sort_values(ascending=False)[1:6]
    )

    similar_ratings = user_item_matrix.loc[similar_users.index]

    weighted_scores = np.dot(
        similar_users.values,
        similar_ratings
    )

    weighted_scores = pd.Series(
        weighted_scores,
        index=user_item_matrix.columns
    )

    user_rated = user_item_matrix.loc[user_id]
    weighted_scores[user_rated > 0] = 0

    max_score = weighted_scores.max()

    scores = {}

    for movie_id, score in weighted_scores.items():

        if max_score > 0:
            normalized = score / max_score
        else:
            normalized = 0

        scores[movie_id] = normalized

    return scores


# ---------------- GENRE SCORE ----------------

def get_genre_scores(genre_weights):

    scores = {}

    for _, row in movies_df.iterrows():

        score = 0

        for genre, weight in genre_weights.items():
            if genre in row and row[genre] == 1:
                score += weight

        scores[row["movieId"]] = score / 100

    return scores


# ---------------- HYBRID RECOMMENDATION ----------------

def hybrid_recommend(user_id, genre_weights, n=10):

    collab_scores = get_collaborative_scores(user_id)
    genre_scores = get_genre_scores(genre_weights)

    final_scores = {}

    for movie_id in movies_df["movieId"]:

        c_score = collab_scores.get(movie_id, 0)
        g_score = genre_scores.get(movie_id, 0)

        final_scores[movie_id] = 0.6 * c_score + 0.4 * g_score

    sorted_movies = sorted(
        final_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:n]

    results = []

    for movie_id, score in sorted_movies:

        movie_row = movies_df[movies_df["movieId"] == movie_id]
        if movie_row.empty:
            continue

        title = movie_row.iloc[0]["title"]

        genres = [
            g for g in genre_cols
            if movie_row.iloc[0][g] == 1
        ]

        results.append((
            title,
           round(min(score * 10, 10), 1),
            genres
        ))

    return results


# ---------------- MANUAL RATING (COLD START) ----------------

def recommend_from_manual_ratings(user_ratings_dict, n=10):

    if not user_ratings_dict:
        return []

    new_user_vector = np.zeros(len(user_item_matrix.columns))
    movie_ids = list(user_item_matrix.columns)

    # Fill ratings into vector
    for title, rating in user_ratings_dict.items():

        movie_row = movies_df[movies_df["title"] == title]

        if not movie_row.empty:
            movie_id = movie_row.iloc[0]["movieId"]

            if movie_id in movie_ids:
                index = movie_ids.index(movie_id)
                new_user_vector[index] = rating

    # Similarity with existing users
    similarities = cosine_similarity(
        [new_user_vector],
        user_item_matrix
    )[0]

    similarity_series = pd.Series(
        similarities,
        index=user_item_matrix.index
    )

    top_similar_users = similarity_series.sort_values(
        ascending=False
    )[1:6]

    similar_ratings = user_item_matrix.loc[top_similar_users.index]

    weighted_scores = np.dot(
        top_similar_users.values,
        similar_ratings
    )

    weighted_scores = pd.Series(
        weighted_scores,
        index=user_item_matrix.columns
    )

    # Remove already rated movies
    for title in user_ratings_dict:
        movie_row = movies_df[movies_df["title"] == title]
        if not movie_row.empty:
            movie_id = movie_row.iloc[0]["movieId"]
            weighted_scores[movie_id] = 0

    top_ids = weighted_scores.sort_values(
        ascending=False
    ).head(n).index

    results = []

    for movie_id in top_ids:

        movie_row = movies_df[movies_df["movieId"] == movie_id]
        if movie_row.empty:
            continue

        title = movie_row.iloc[0]["title"]
        results.append(title)

    return results
