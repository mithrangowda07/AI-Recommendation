import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class HybridMovieRecommender:
    def __init__(self, movies_path: str = 'data/movies.json', 
                 ratings_path: str = 'data/ratings.csv',
                 models_dir: str = 'models'):
        self.movies_path = movies_path
        self.ratings_path = ratings_path
        self.models_dir = models_dir
        self.movies_df = None
        self.ratings_df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.svd_model = None
        self.movie_id_to_index = {}
        self.index_to_movie_id = {}
        
    def load_data(self):
        """Load movies and ratings data"""
        print("Loading movies data...")
        with open(self.movies_path, 'r', encoding='utf-8') as f:
            movies_data = json.load(f)
        self.movies_df = pd.DataFrame(movies_data)
        
        print("Loading ratings data...")
        self.ratings_df = pd.read_csv(self.ratings_path)
        
        # Create mapping between movie_id and index
        self.movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(self.movies_df['movie_id'])}
        self.index_to_movie_id = {idx: movie_id for movie_id, idx in self.movie_id_to_index.items()}
        
        print(f"Loaded {len(self.movies_df)} movies and {len(self.ratings_df)} ratings")
        
    def make_combined_text(self, row):
        """Create combined text field from movie metadata"""
        # Get top 5 cast members
        cast_list = row['cast'].split(', ')[:5] if pd.notna(row['cast']) else []
        cast_text = ' '.join(cast_list)
        
        # Combine all text fields
        combined = f"{row['overview']} {row['genre']} {cast_text} {row['director']} {row['director']} {row['tagline']} {row['original_language']}"
        
        # Clean up the text
        combined = ' '.join(combined.split())  # Remove extra whitespace
        return combined.lower()
    
    def prepare_content_data(self):
        """Prepare data for content-based filtering"""
        print("Preparing content data...")
        self.movies_df['combined_text'] = self.movies_df.apply(self.make_combined_text, axis=1)
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            stop_words='english',
            lowercase=True
        )
        
        # Fit and transform the combined text
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df['combined_text'])
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
    def train_collaborative_model(self):
        """Train collaborative filtering model using matrix factorization"""
        print("Training collaborative filtering model...")
        
        # Create user-item matrix
        user_item_matrix = self.ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating', 
            fill_value=0
        )
        
        # Fill missing values with user mean ratings
        user_means = user_item_matrix.mean(axis=1)
        user_item_matrix = user_item_matrix.apply(lambda row: row.replace(0, user_means[row.name]), axis=1)
        
        # Apply SVD for matrix factorization
        self.svd_model = TruncatedSVD(n_components=50, random_state=42)
        self.user_factors = self.svd_model.fit_transform(user_item_matrix)
        self.item_factors = self.svd_model.components_.T
        
        # Store user and item mappings
        self.user_ids = user_item_matrix.index.tolist()
        self.item_ids = user_item_matrix.columns.tolist()
        
        # Create user and item ID to index mappings
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(self.item_ids)}
        
        print(f"Collaborative filtering model trained with {len(self.user_ids)} users and {len(self.item_ids)} items")
        
    def save_models(self):
        """Save trained models"""
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Save TF-IDF model and matrix
        joblib.dump(self.tfidf_vectorizer, os.path.join(self.models_dir, 'tfidf.pkl'))
        joblib.dump(self.tfidf_matrix, os.path.join(self.models_dir, 'tfidf_matrix.pkl'))
        
        # Save SVD model and factors
        joblib.dump(self.svd_model, os.path.join(self.models_dir, 'svd.pkl'))
        joblib.dump(self.user_factors, os.path.join(self.models_dir, 'user_factors.pkl'))
        joblib.dump(self.item_factors, os.path.join(self.models_dir, 'item_factors.pkl'))
        joblib.dump(self.user_ids, os.path.join(self.models_dir, 'user_ids.pkl'))
        joblib.dump(self.item_ids, os.path.join(self.models_dir, 'item_ids.pkl'))
        joblib.dump(self.user_id_to_idx, os.path.join(self.models_dir, 'user_id_to_idx.pkl'))
        joblib.dump(self.item_id_to_idx, os.path.join(self.models_dir, 'item_id_to_idx.pkl'))
        
        # Save mappings
        joblib.dump(self.movie_id_to_index, os.path.join(self.models_dir, 'movie_id_to_index.pkl'))
        joblib.dump(self.index_to_movie_id, os.path.join(self.models_dir, 'index_to_movie_id.pkl'))
        
        # Save movies dataframe
        self.movies_df.to_pickle(os.path.join(self.models_dir, 'movies_df.pkl'))
        
        print("Models saved successfully!")
        
    def load_models(self):
        """Load pre-trained models"""
        print("Loading pre-trained models...")
        
        self.tfidf_vectorizer = joblib.load(os.path.join(self.models_dir, 'tfidf.pkl'))
        self.tfidf_matrix = joblib.load(os.path.join(self.models_dir, 'tfidf_matrix.pkl'))
        self.svd_model = joblib.load(os.path.join(self.models_dir, 'svd.pkl'))
        self.user_factors = joblib.load(os.path.join(self.models_dir, 'user_factors.pkl'))
        self.item_factors = joblib.load(os.path.join(self.models_dir, 'item_factors.pkl'))
        self.user_ids = joblib.load(os.path.join(self.models_dir, 'user_ids.pkl'))
        self.item_ids = joblib.load(os.path.join(self.models_dir, 'item_ids.pkl'))
        self.user_id_to_idx = joblib.load(os.path.join(self.models_dir, 'user_id_to_idx.pkl'))
        self.item_id_to_idx = joblib.load(os.path.join(self.models_dir, 'item_id_to_idx.pkl'))
        self.movie_id_to_index = joblib.load(os.path.join(self.models_dir, 'movie_id_to_index.pkl'))
        self.index_to_movie_id = joblib.load(os.path.join(self.models_dir, 'index_to_movie_id.pkl'))
        self.movies_df = pd.read_pickle(os.path.join(self.models_dir, 'movies_df.pkl'))
        
        print("Models loaded successfully!")
        
    def get_content_similarity(self, movie_title: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """Get content-based similarity scores for a movie"""
        # Find movie by title (case-insensitive)
        movie_idx = None
        for idx, title in enumerate(self.movies_df['title']):
            if title.lower() == movie_title.lower():
                movie_idx = idx
                break
                
        if movie_idx is None:
            # Try partial matching
            for idx, title in enumerate(self.movies_df['title']):
                if movie_title.lower() in title.lower():
                    movie_idx = idx
                    break
                    
        if movie_idx is None:
            return []
            
        # Get similarity scores
        movie_vector = self.tfidf_matrix[movie_idx]
        similarities = cosine_similarity(movie_vector, self.tfidf_matrix).flatten()
        
        # Get top-k similar movies (excluding the input movie itself)
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        similar_scores = similarities[similar_indices]
        
        return list(zip(similar_indices, similar_scores))
        
    def get_collaborative_scores(self, user_id: int, movie_indices: List[int]) -> List[float]:
        """Get collaborative filtering scores for a user and list of movies"""
        scores = []
        
        # Get user index
        if user_id not in self.user_id_to_idx:
            # Return default scores if user not found
            return [2.5] * len(movie_indices)
        
        user_idx = self.user_id_to_idx[user_id]
        user_vector = self.user_factors[user_idx]
        
        for movie_idx in movie_indices:
            movie_id = self.index_to_movie_id[movie_idx]
            
            # Check if movie exists in collaborative filtering data
            if movie_id in self.item_id_to_idx:
                item_idx = self.item_id_to_idx[movie_id]
                item_vector = self.item_factors[item_idx]
                
                # Calculate predicted rating using dot product
                predicted_rating = np.dot(user_vector, item_vector)
                
                # Ensure rating is within valid range (1-5)
                predicted_rating = max(1.0, min(5.0, predicted_rating))
                scores.append(predicted_rating)
            else:
                # Default rating for movies not in collaborative data
                scores.append(2.5)
        
        return scores
        
    def hybrid_recommend(self, user_id: int, movie_title: str, alpha: float = 0.6, 
                        top_n: int = 10) -> pd.DataFrame:
        """
        Generate hybrid recommendations combining content-based and collaborative filtering
        
        Args:
            user_id: User ID for collaborative filtering
            movie_title: Title of the movie to find similar movies for
            alpha: Weight for content-based filtering (0-1, higher = more content-based)
            top_n: Number of recommendations to return
            
        Returns:
            DataFrame with recommendations
        """
        # Get content-based similarities
        content_similarities = self.get_content_similarity(movie_title, top_k=100)
        
        if not content_similarities:
            return pd.DataFrame(columns=['movie_id', 'title', 'genre', 'overview', 'rating', 'score'])
            
        # Extract movie indices and content scores
        movie_indices = [idx for idx, _ in content_similarities]
        content_scores = [score for _, score in content_similarities]
        
        # Get collaborative filtering scores
        collab_scores = self.get_collaborative_scores(user_id, movie_indices)
        
        # Normalize scores to 0-1 range
        content_scores = np.array(content_scores)
        collab_scores = np.array(collab_scores)
        
        # Normalize content scores
        if content_scores.max() > content_scores.min():
            content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min())
        
        # Normalize collaborative scores (assuming ratings are 1-5)
        collab_scores = (collab_scores - 1) / 4  # Scale from 1-5 to 0-1
        
        # Calculate hybrid scores
        hybrid_scores = alpha * content_scores + (1 - alpha) * collab_scores
        
        # Get top recommendations
        top_indices = np.argsort(hybrid_scores)[::-1][:top_n]
        
        # Create results dataframe
        results = []
        for idx in top_indices:
            movie_idx = movie_indices[idx]
            movie_id = self.index_to_movie_id[movie_idx]
            movie_data = self.movies_df.iloc[movie_idx]
            
            results.append({
                'movie_id': movie_id,
                'title': movie_data['title'],
                'genre': movie_data['genre'],
                'overview': movie_data['overview'],
                'rating': movie_data['rating'],
                'score': hybrid_scores[idx],
                'content_score': content_scores[idx],
                'collab_score': collab_scores[idx]
            })
            
        return pd.DataFrame(results)


# Global recommender instance
recommender = None

def get_recommender():
    """Get or create the global recommender instance"""
    global recommender
    if recommender is None:
        recommender = HybridMovieRecommender()
        try:
            recommender.load_models()
        except FileNotFoundError:
            print("Models not found. Please run train.py first.")
            return None
    return recommender

def hybrid_recommend(user_id: int, movie_title: str, alpha: float = 0.6, top_n: int = 10):
    """Convenience function for hybrid recommendations"""
    rec = get_recommender()
    if rec is None:
        return pd.DataFrame()
    return rec.hybrid_recommend(user_id, movie_title, alpha, top_n)
