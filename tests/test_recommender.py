"""
Tests for the hybrid movie recommendation system.

This module contains comprehensive tests for the recommender functionality,
including content-based filtering, collaborative filtering, and hybrid recommendations.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import app modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.recommender import HybridMovieRecommender, hybrid_recommend, get_recommender
from app.utils import save_model, load_model, get_model_info


class TestHybridMovieRecommender:
    """Test cases for the HybridMovieRecommender class"""
    
    @pytest.fixture
    def sample_movies_data(self):
        """Sample movies data for testing"""
        return [
            {
                "movie_id": 1,
                "title": "Toy Story",
                "overview": "A story about toys that come to life",
                "genre": "Animation, Comedy, Family",
                "cast": "Tom Hanks, Tim Allen",
                "director": "John Lasseter",
                "tagline": "A toy story",
                "original_language": "en",
                "rating": 3.8
            },
            {
                "movie_id": 2,
                "title": "The Dark Knight",
                "overview": "Batman faces the Joker in Gotham City",
                "genre": "Action, Crime, Drama",
                "cast": "Christian Bale, Heath Ledger",
                "director": "Christopher Nolan",
                "tagline": "Why so serious?",
                "original_language": "en",
                "rating": 4.5
            },
            {
                "movie_id": 3,
                "title": "Inception",
                "overview": "A thief who enters dreams",
                "genre": "Action, Sci-Fi, Thriller",
                "cast": "Leonardo DiCaprio, Marion Cotillard",
                "director": "Christopher Nolan",
                "tagline": "Your mind is the scene of the crime",
                "original_language": "en",
                "rating": 4.2
            }
        ]
    
    @pytest.fixture
    def sample_ratings_data(self):
        """Sample ratings data for testing"""
        return pd.DataFrame({
            'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'movieId': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'rating': [4.0, 5.0, 4.5, 3.5, 4.5, 4.0, 4.2, 4.8, 4.3]
        })
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def recommender(self, sample_movies_data, sample_ratings_data, temp_dir):
        """Create a recommender instance for testing"""
        # Create temporary data files
        movies_file = os.path.join(temp_dir, 'movies.json')
        ratings_file = os.path.join(temp_dir, 'ratings.csv')
        
        import json
        with open(movies_file, 'w') as f:
            json.dump(sample_movies_data, f)
        
        sample_ratings_data.to_csv(ratings_file, index=False)
        
        # Create recommender instance
        rec = HybridMovieRecommender(movies_file, ratings_file, temp_dir)
        rec.load_data()
        return rec
    
    def test_load_data(self, recommender):
        """Test data loading functionality"""
        assert recommender.movies_df is not None
        assert recommender.ratings_df is not None
        assert len(recommender.movies_df) == 3
        assert len(recommender.ratings_df) == 9
        assert 'movie_id' in recommender.movies_df.columns
        assert 'userId' in recommender.ratings_df.columns
    
    def test_make_combined_text(self, recommender):
        """Test the make_combined_text function"""
        movie_row = recommender.movies_df.iloc[0]
        combined_text = recommender.make_combined_text(movie_row)
        
        assert isinstance(combined_text, str)
        assert 'toy story' in combined_text.lower()
        assert 'animation' in combined_text.lower()
        assert 'tom hanks' in combined_text.lower()
        assert 'john lasseter' in combined_text.lower()
    
    def test_prepare_content_data(self, recommender):
        """Test content data preparation"""
        recommender.prepare_content_data()
        
        assert recommender.tfidf_vectorizer is not None
        assert recommender.tfidf_matrix is not None
        assert recommender.tfidf_matrix.shape[0] == 3  # 3 movies
        assert 'combined_text' in recommender.movies_df.columns
    
    def test_train_collaborative_model(self, recommender):
        """Test collaborative filtering model training"""
        recommender.train_collaborative_model()
        
        assert recommender.svd_model is not None
        # Test that the model can make predictions
        pred = recommender.svd_model.predict(1, 1)
        assert isinstance(pred.est, float)
        assert 1 <= pred.est <= 5
    
    def test_get_content_similarity(self, recommender):
        """Test content-based similarity calculation"""
        recommender.prepare_content_data()
        
        similarities = recommender.get_content_similarity("Toy Story", top_k=2)
        
        assert len(similarities) <= 2
        if similarities:
            assert all(isinstance(idx, int) for idx, _ in similarities)
            assert all(isinstance(score, float) for _, score in similarities)
    
    def test_get_content_similarity_nonexistent(self, recommender):
        """Test content similarity with non-existent movie"""
        recommender.prepare_content_data()
        
        similarities = recommender.get_content_similarity("Non-existent Movie", top_k=5)
        assert similarities == []
    
    def test_get_collaborative_scores(self, recommender):
        """Test collaborative filtering score calculation"""
        recommender.train_collaborative_model()
        
        movie_indices = [0, 1, 2]
        scores = recommender.get_collaborative_scores(1, movie_indices)
        
        assert len(scores) == 3
        assert all(isinstance(score, float) for score in scores)
        assert all(1 <= score <= 5 for score in scores)
    
    def test_hybrid_recommend(self, recommender):
        """Test hybrid recommendation generation"""
        recommender.prepare_content_data()
        recommender.train_collaborative_model()
        
        recommendations = recommender.hybrid_recommend(
            user_id=1,
            movie_title="Toy Story",
            alpha=0.6,
            top_n=2
        )
        
        assert isinstance(recommendations, pd.DataFrame)
        assert len(recommendations) <= 2
        assert 'movie_id' in recommendations.columns
        assert 'title' in recommendations.columns
        assert 'score' in recommendations.columns
        assert 'content_score' in recommendations.columns
        assert 'collab_score' in recommendations.columns
    
    def test_hybrid_recommend_nonexistent_movie(self, recommender):
        """Test hybrid recommendation with non-existent movie"""
        recommender.prepare_content_data()
        recommender.train_collaborative_model()
        
        recommendations = recommender.hybrid_recommend(
            user_id=1,
            movie_title="Non-existent Movie",
            alpha=0.6,
            top_n=5
        )
        
        assert isinstance(recommendations, pd.DataFrame)
        assert len(recommendations) == 0
    
    def test_save_and_load_models(self, recommender, temp_dir):
        """Test model saving and loading"""
        recommender.prepare_content_data()
        recommender.train_collaborative_model()
        recommender.save_models()
        
        # Create new recommender instance and load models
        new_rec = HybridMovieRecommender(models_dir=temp_dir)
        new_rec.load_models()
        
        assert new_rec.tfidf_vectorizer is not None
        assert new_rec.tfidf_matrix is not None
        assert new_rec.svd_model is not None
        assert new_rec.movies_df is not None
        assert len(new_rec.movies_df) == 3


class TestUtilityFunctions:
    """Test cases for utility functions"""
    
    def test_save_and_load_model(self, tempfile):
        """Test model saving and loading utilities"""
        test_model = {"test": "data", "number": 42}
        
        # Save model
        filepath = save_model(test_model, "test_model.pkl", tempfile.name)
        assert os.path.exists(filepath)
        
        # Load model
        loaded_model = load_model("test_model.pkl", tempfile.name)
        assert loaded_model == test_model
    
    def test_get_model_info(self, tempfile):
        """Test model info retrieval"""
        info = get_model_info(tempfile.name)
        
        assert 'models_dir' in info
        assert 'available_models' in info
        assert 'total_models' in info
        assert 'missing_models' in info
        assert 'is_ready' in info
        assert info['models_dir'] == tempfile.name


class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_hybrid_recommend_function(self, sample_movies_data, sample_ratings_data, temp_dir):
        """Test the hybrid_recommend convenience function"""
        # This test would require setting up the global recommender
        # For now, we'll test the function exists and can be imported
        assert callable(hybrid_recommend)
    
    def test_get_recommender_function(self):
        """Test the get_recommender function"""
        # This test would require models to be available
        # For now, we'll test the function exists and can be imported
        assert callable(get_recommender)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_data(self, temp_dir):
        """Test handling of empty data"""
        # Create empty data files
        movies_file = os.path.join(temp_dir, 'empty_movies.json')
        ratings_file = os.path.join(temp_dir, 'empty_ratings.csv')
        
        with open(movies_file, 'w') as f:
            f.write('[]')
        
        pd.DataFrame(columns=['userId', 'movieId', 'rating']).to_csv(ratings_file, index=False)
        
        rec = HybridMovieRecommender(movies_file, ratings_file, temp_dir)
        
        with pytest.raises(Exception):
            rec.load_data()
    
    def test_invalid_alpha_values(self, sample_movies_data, sample_ratings_data, temp_dir):
        """Test handling of invalid alpha values"""
        # This would be tested in the Flask routes, but we can test the recommender
        movies_file = os.path.join(temp_dir, 'movies.json')
        ratings_file = os.path.join(temp_dir, 'ratings.csv')
        
        import json
        with open(movies_file, 'w') as f:
            json.dump(sample_movies_data, f)
        
        sample_ratings_data.to_csv(ratings_file, index=False)
        
        rec = HybridMovieRecommender(movies_file, ratings_file, temp_dir)
        rec.load_data()
        rec.prepare_content_data()
        rec.train_collaborative_model()
        
        # Test with invalid alpha values
        with pytest.raises(ValueError):
            rec.hybrid_recommend(1, "Toy Story", alpha=-0.1, top_n=5)
        
        with pytest.raises(ValueError):
            rec.hybrid_recommend(1, "Toy Story", alpha=1.1, top_n=5)


if __name__ == "__main__":
    pytest.main([__file__])
