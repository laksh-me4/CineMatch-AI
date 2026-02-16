import streamlit as st
from model import hybrid_recommend, recommend_from_manual_ratings
from model import movies_df, user_item_matrix, user_similarity_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------

st.set_page_config(page_title="CineMatch AI", layout="centered")

st.title("🎬 CineMatch AI")
st.subheader("Hybrid Recommendation System (Collaborative + Content-Based)")

# ---------------- SIDEBAR ----------------

st.sidebar.title("🎬 CineMatch Dashboard")

st.sidebar.markdown("## 📊 Dataset Overview")
st.sidebar.write(f"👥 Total Users: {len(user_item_matrix.index)}")
st.sidebar.write(f"🎞 Total Movies: {len(movies_df)}")
st.sidebar.write(f"🧠 Total Ratings: {user_item_matrix.astype(bool).sum().sum()}")

st.sidebar.markdown("---")

st.sidebar.markdown("## 📈 Analytics")
show_ratings_chart = st.sidebar.checkbox("Show Ratings Distribution")
show_genre_chart = st.sidebar.checkbox("Show Genre Popularity")
show_trending_chart = st.sidebar.checkbox("Show Trending Movies")
show_heatmap = st.sidebar.checkbox("Show User Similarity Heatmap")

# ---------------- USER INPUT ----------------

user_id = st.number_input("Enter Existing User ID", min_value=1, step=1)

st.header("🎭 Genre Preferences")

genre_list = ["Action", "Adventure", "Comedy",
              "Drama", "Romance", "Sci-Fi", "Thriller"]

genre_weights = {}
for genre in genre_list:
    genre_weights[genre] = st.slider(genre, 0, 100, 50)

st.header("⭐ Rate Movies (New User Mode)")

movie_titles = sorted(movies_df["title"].values)
selected_movie = st.selectbox("Select Movie", movie_titles)
rating = st.slider("Your Rating", 1, 10, 5)

if "manual_ratings" not in st.session_state:
    st.session_state.manual_ratings = {}

if st.button("Add Rating"):
    st.session_state.manual_ratings[selected_movie] = rating
    st.success(f"Added rating for {selected_movie}")

if st.session_state.manual_ratings:
    st.write("### Your Ratings")
    for movie, rate in st.session_state.manual_ratings.items():
        st.write(f"{movie} — {rate}/10")

# ---------------- GENERATE ----------------

if st.button("🚀 Generate Recommendations"):

    st.balloons()  # 🎉 wow effect

    # -------- MANUAL MODE --------
    if st.session_state.manual_ratings:

        results = recommend_from_manual_ratings(
            st.session_state.manual_ratings
        )

        st.subheader("🎯 Personalized Recommendations")

        for title in results:
            st.success(f"🎬 {title} — ⭐ Recommended")

    # -------- HYBRID MODE --------
    else:

        results = hybrid_recommend(user_id, genre_weights)

        if not results:
            st.warning("No recommendations found.")
        else:
            st.subheader("🎯 Hybrid Recommendations")

            for title, score, genres in results:

                numeric_score = float(score)

                # Star calculation
                star_value = numeric_score / 2
                full_stars = int(star_value)
                half_star = (star_value - full_stars) >= 0.5
                empty_stars = 5 - full_stars - (1 if half_star else 0)

                stars = "⭐" * full_stars
                if half_star:
                    stars += "⭐"
                stars += "☆" * empty_stars

                rating_percent = int((numeric_score / 10) * 100)

                # -------- CARD BLOCK --------
                st.markdown(f"""
                <div style="
                    background-color:#1f2937;
                    padding:18px;
                    border-radius:12px;
                    margin-bottom:18px;
                    border:1px solid #374151;
                ">

                <h4 style="margin-bottom:6px;">🎬 {title}</h4>

                <p style="font-size:18px; margin:0;">
                {stars} ({numeric_score}/10)
                </p>

                <div style="
                    background-color:#374151;
                    border-radius:10px;
                    height:12px;
                    margin-top:8px;
                ">
                    <div style="
                        width:{rating_percent}%;
                        background:linear-gradient(90deg,#fbbf24,#f59e0b);
                        height:12px;
                        border-radius:10px;
                    ">
                    </div>
                </div>

                <p style="margin-top:10px; font-size:13px; color:#9ca3af;">
                Genres: {genres}
                </p>

                </div>
                """, unsafe_allow_html=True)

# ---------------- ANALYTICS ----------------

if show_ratings_chart:
    st.header("📊 Ratings Distribution")
    all_ratings = user_item_matrix.values.flatten()
    all_ratings = all_ratings[all_ratings > 0]
    fig, ax = plt.subplots()
    ax.hist(all_ratings, bins=10)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

if show_genre_chart:
    st.header("🎭 Genre Popularity")
    genre_counts = {}
    for genre in movies_df.columns[2:]:
        genre_counts[genre] = movies_df[genre].sum()
    fig, ax = plt.subplots()
    ax.bar(genre_counts.keys(), genre_counts.values())
    ax.set_xticklabels(genre_counts.keys(), rotation=90)
    st.pyplot(fig)

if show_trending_chart:
    st.header("🔥 Top 10 Most Rated Movies")
    popularity = (
        user_item_matrix.astype(bool)
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    titles = []
    counts = []
    for movie_id, count in popularity.items():
        movie = movies_df[movies_df["movieId"] == movie_id]
        if not movie.empty:
            titles.append(movie.iloc[0]["title"])
            counts.append(count)
    fig, ax = plt.subplots()
    ax.barh(titles, counts)
    ax.invert_yaxis()
    st.pyplot(fig)

if show_heatmap:
    st.header("🧠 User Similarity Heatmap (First 20 Users)")
    sample_similarity = user_similarity_df.iloc[:20, :20]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(sample_similarity, cmap="coolwarm")
    st.pyplot(fig)
