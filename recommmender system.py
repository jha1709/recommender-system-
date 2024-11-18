import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load the datasets (replace with your file paths)
movies = pd.read_csv(r"C:/Users/Admin/Downloads/ml-25m/movies.csv")  # Adjust path as needed
ratings = pd.read_csv(r"C:/Users/Admin/Downloads/ml-25m/ratings.csv")  # Adjust path as needed

# --- 1. Visualize Data: Box Plot, Histogram, Scatter Plot ---

# Box Plot: Distribution of Ratings
plt.figure(figsize=(8, 6))
sns.boxplot(x='rating', data=ratings)
plt.title('Box Plot of Movie Ratings')
plt.show()

# Histogram: Distribution of Ratings
plt.figure(figsize=(8, 6))
sns.histplot(ratings['rating'], kde=True, bins=20)
plt.title('Histogram of Movie Ratings')
plt.show()

# Scatter Plot: Movie Ratings vs. User IDs
plt.figure(figsize=(8, 6))
sns.scatterplot(x='userId', y='rating', data=ratings)
plt.title('Scatter Plot of User Ratings')
plt.show()

# --- 2. Pearson Correlation ---

# Compute Pearson Correlation Matrix
correlation_matrix = ratings[['userId', 'movieId', 'rating']].corr()

# Plot Pearson Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Pearson Correlation Matrix')
plt.show()

# --- 3. Identify Dependent and Independent Features ---
# Dependent feature: Rating (we want to predict ratings)
# Independent features: UserId, MovieId, Genres, etc.

# --- 4. Collaborative Filtering for Movie Recommendation ---

# Create user-item matrix for collaborative filtering
user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Compute user similarity using Cosine Similarity
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Recommendation function based on collaborative filtering
def recommend_movies(user_id, num_recommendations=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]  # Get similar users
    recommendations = {}
    
    # Gather recommendations from similar users
    for similar_user in similar_users:
        user_ratings = user_movie_matrix.loc[similar_user]
        for movie, rating in user_ratings[user_ratings > 3].items():  # Consider ratings above 3
            if movie not in user_movie_matrix.loc[user_id]:  # Exclude already rated movies
                recommendations[movie] = recommendations.get(movie, 0) + rating
    
    # Get top N recommendations
    recommended_movie_ids = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
    recommended_movie_titles = movies[movies['movieId'].isin([id[0] for id in recommended_movie_ids])]
    return recommended_movie_titles['title'].tolist()

# Example usage: Recommend movies for User 1
recommended_movies = recommend_movies(user_id=1, num_recommendations=5)
print("Recommended Movies for User 1:", recommended_movies)

# --- 5. Content-Based Filtering (Optional) ---

# Prepare genres for content-based filtering (optional)
movies['genres'] = movies['genres'].str.replace('|', ' ')  # Replace "|" with spaces for genre names
count_vectorizer = CountVectorizer(stop_words='english')
genre_matrix = count_vectorizer.fit_transform(movies['genres'])

# Compute cosine similarity between movies based on genres
movie_similarity = cosine_similarity(genre_matrix)
movie_similarity_df = pd.DataFrame(movie_similarity, index=movies['title'], columns=movies['title'])

# Content-Based Recommendation Function
def recommend_similar_movies(movie_title, num_recommendations=5):
    similar_movies = movie_similarity_df[movie_title].sort_values(ascending=False)[1:num_recommendations + 1]
    return similar_movies.index.tolist()

# Example usage: Recommend movies similar to "Toy Story (1995)"
similar_movies = recommend_similar_movies(movie_title="Toy Story (1995)", num_recommendations=5)
print("Content-Based Recommendations for 'Toy Story (1995)':", similar_movies)
