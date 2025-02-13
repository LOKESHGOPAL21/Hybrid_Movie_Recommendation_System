import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Load datasets
ratings_data = pd.read_csv("ratings.csv")  # Format: userId, movieId, rating
movies_data = pd.read_csv("movies_metadata.csv")  # Format: id, title, genres, overview

# Collaborative Filtering Model
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_data[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)
collab_model = SVD()
collab_model.fit(trainset)

# Content-Based Filtering (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
movies_data['overview'] = movies_data['overview'].fillna("")
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_data['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Functions for Recommendations
def content_based_recommend(movie_name):
    idx = movies_data[movies_data['title'].str.lower() == movie_name.lower()].index
    if len(idx) == 0:
        return []
    
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    
    movie_indices = [i[0] for i in sim_scores]
    return movies_data.iloc[movie_indices][['title', 'id']].to_dict(orient='records')

def collaborative_filtering_recommend(movie_id):
    similar_movies = ratings_data[ratings_data['movieId'] == movie_id]
    if similar_movies.empty:
        return []
    
    similar_movies = similar_movies.groupby('movieId').agg({'rating': 'mean'}).reset_index()
    similar_movies = similar_movies.sort_values(by='rating', ascending=False).head(5)
    
    recommended_movies = movies_data[movies_data['id'].isin(similar_movies['movieId'])][['title', 'id']]
    return recommended_movies.to_dict(orient='records')
