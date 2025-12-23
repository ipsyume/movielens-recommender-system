# MovieLens Recommender System using Collaborative Filtering

This project implements a personalized movie recommendation system using collaborative filtering and matrix factorization (SVD). The system is trained and evaluated on the MovieLens 100K dataset and generates top-N movie recommendations for users based on their historical ratings.

---

## Problem Statement
Recommender systems are widely used in platforms such as Netflix and Amazon to personalize content for users. The goal of this project is to predict user preferences and recommend unseen movies using historical rating data without relying on explicit content features.

---

## Dataset
- **MovieLens 100K Dataset**
- 100,000 ratings from 943 users on 1,682 movies
- Rating scale: 1 to 5

Dataset source:  
https://grouplens.org/datasets/movielens/100k/

---

## Methodology
- Collaborative Filtering
- Matrix Factorization using Singular Value Decomposition (SVD)
- Implemented using the `scikit-surprise` library

---

## Model Evaluation
The model was evaluated using **5-fold cross-validation** to ensure robustness and generalization.

Evaluation metrics:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

### Cross-validation Results
![Cross Validation Results](cross_validation_results.png)

---

## Recommendation Output
After training on the full dataset, the model generates personalized movie recommendations for a given user by predicting ratings for unseen movies.

### Sample Recommendations
![Final Recommendations](final_recommendations_output.png)

---

## Project Structure
movielens-recommender-system/
│
├── movielens_recommender.py
├── README.md
├── requirements.txt
└── screenshots/

---

## How to Run
```bash
pip install -r requirements.txt
python movielens_recommender.py
