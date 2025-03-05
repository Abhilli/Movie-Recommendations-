# Movie-Recommendations-

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset
data = {
    "movie_id": [1, 2, 3, 4, 5],
    "title": ["Inception", "Interstellar", "Avatar", "Titanic", "The Dark Knight"],
    "genre": ["Sci-Fi, Thriller", "Sci-Fi, Adventure", "Sci-Fi, Action", "Romance, Drama", "Action, Crime"]
}

df = pd.DataFrame(data)

# Convert genre text into numeric features using TF-IDF
vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(df['genre'])

# Compute similarity scores
similarity = cosine_similarity(genre_matrix)

# Function to recommend movies based on similarity
def recommend_movies(movie_title, num_recommendations=3):
    if movie_title not in df['title'].values:
        return "Movie not found in database."
    
    movie_idx = df[df['title'] == movie_title].index[0]
    scores = list(enumerate(similarity[movie_idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    recommended_movies = [df.iloc[i[0]]['title'] for i in scores[1:num_recommendations+1]]
    return recommended_movies

# Example usage
print(recommend_movies("Inception", 2))
