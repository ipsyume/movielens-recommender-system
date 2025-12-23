import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# -----------------------------
# 1. Load MovieLens data
# -----------------------------

# Ratings file
ratings = pd.read_csv(
    "u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# Movie titles file
movies = pd.read_csv(
    "u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "title"]
)

# -----------------------------
# 2. Prepare Surprise dataset
# -----------------------------
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(
    ratings[["user_id", "movie_id", "rating"]],
    reader
)

# -----------------------------
# 3. Model selection & evaluation
# -----------------------------
algo = SVD(
    n_factors=100,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)

print("\nEvaluating model using 5-fold cross-validation...\n")
cross_validate(
    algo,
    data,
    measures=["RMSE", "MAE"],
    cv=5,
    verbose=True
)

# -----------------------------
# 4. Train on full dataset
# -----------------------------
trainset = data.build_full_trainset()
algo.fit(trainset)

# -----------------------------
# 5. Generate recommendations
# -----------------------------
USER_ID = 1

# Movies already rated by the user
rated_movies = set(
    ratings[ratings["user_id"] == USER_ID]["movie_id"]
)

# All movies in dataset
all_movies = set(movies["movie_id"])

# Recommend only unseen movies
unseen_movies = all_movies - rated_movies

# Predict ratings for unseen movies
predictions = []
for movie_id in unseen_movies:
    pred = algo.predict(USER_ID, movie_id)
    predictions.append((movie_id, pred.est))

# Top-N recommendations
TOP_N = 5
top_recommendations = sorted(
    predictions, key=lambda x: x[1], reverse=True
)[:TOP_N]

# -----------------------------
# 6. Display results
# -----------------------------
print(f"\nTop {TOP_N} movie recommendations for User {USER_ID}:\n")

for movie_id, score in top_recommendations:
    title = movies[movies["movie_id"] == movie_id]["title"].values[0]
    print(f"{title}  |  Predicted Rating: {score:.2f}")
