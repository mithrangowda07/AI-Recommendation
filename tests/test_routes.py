"""
Tests for Flask routes and API endpoints.

This module contains tests for the web interface and REST API functionality.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
import pandas as pd

# Add the parent directory to the path to import app modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app import create_app


@pytest.fixture
def app():
    """Create a test Flask application"""
    app = create_app()
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Create a test client for the Flask application"""
    return app.test_client()


@pytest.fixture
def sample_recommendations():
    """Sample recommendations data for testing"""
    return pd.DataFrame([
        {
            'movie_id': 1,
            'title': 'Toy Story 2',
            'genre': 'Animation, Comedy, Family',
            'overview': 'Woody is stolen by a toy collector',
            'rating': 3.9,
            'score': 0.85,
            'content_score': 0.8,
            'collab_score': 0.9
        },
        {
            'movie_id': 2,
            'title': 'Monsters Inc.',
            'genre': 'Animation, Comedy, Family',
            'overview': 'Monsters generate power by scaring children',
            'rating': 4.1,
            'score': 0.78,
            'content_score': 0.75,
            'collab_score': 0.81
        }
    ])


class TestWebRoutes:
    """Test cases for web interface routes"""
    
    def test_index_route(self, client):
        """Test the homepage route"""
        response = client.get('/')
        assert response.status_code == 200
        assert b'Movie Recommendation System' in response.data
        assert b'Get Recommendations' in response.data
    
    def test_recommend_route_get(self, client):
        """Test GET request to recommend route (should redirect)"""
        response = client.get('/recommend')
        assert response.status_code == 405  # Method not allowed
    
    @patch('app.routes.get_recommender')
    def test_recommend_route_post_success(self, mock_get_recommender, client, sample_recommendations):
        """Test successful POST request to recommend route"""
        # Mock the recommender
        mock_recommender = MagicMock()
        mock_recommender.hybrid_recommend.return_value = sample_recommendations
        mock_get_recommender.return_value = mock_recommender
        
        response = client.post('/recommend', data={
            'movie_title': 'Toy Story',
            'user_id': '1',
            'alpha': '0.6',
            'top_n': '10'
        })
        
        assert response.status_code == 200
        assert b'Toy Story 2' in response.data
        assert b'Monsters Inc.' in response.data
    
    def test_recommend_route_missing_title(self, client):
        """Test POST request with missing movie title"""
        response = client.post('/recommend', data={
            'user_id': '1',
            'alpha': '0.6',
            'top_n': '10'
        })
        
        assert response.status_code == 302  # Redirect
        # Check that flash message was set (would need to follow redirect)
    
    def test_recommend_route_missing_user_id(self, client):
        """Test POST request with missing user ID"""
        response = client.post('/recommend', data={
            'movie_title': 'Toy Story',
            'alpha': '0.6',
            'top_n': '10'
        })
        
        assert response.status_code == 302  # Redirect
    
    def test_recommend_route_invalid_alpha(self, client):
        """Test POST request with invalid alpha value"""
        response = client.post('/recommend', data={
            'movie_title': 'Toy Story',
            'user_id': '1',
            'alpha': '1.5',  # Invalid alpha
            'top_n': '10'
        })
        
        assert response.status_code == 302  # Redirect
    
    def test_recommend_route_invalid_top_n(self, client):
        """Test POST request with invalid top_n value"""
        response = client.post('/recommend', data={
            'movie_title': 'Toy Story',
            'user_id': '1',
            'alpha': '0.6',
            'top_n': '100'  # Invalid top_n
        })
        
        assert response.status_code == 302  # Redirect
    
    @patch('app.routes.get_recommender')
    def test_recommend_route_no_recommendations(self, mock_get_recommender, client):
        """Test POST request when no recommendations are found"""
        # Mock the recommender to return empty DataFrame
        mock_recommender = MagicMock()
        mock_recommender.hybrid_recommend.return_value = pd.DataFrame()
        mock_get_recommender.return_value = mock_recommender
        
        response = client.post('/recommend', data={
            'movie_title': 'Unknown Movie',
            'user_id': '1',
            'alpha': '0.6',
            'top_n': '10'
        })
        
        assert response.status_code == 302  # Redirect
    
    @patch('app.routes.get_recommender')
    def test_recommend_route_recommender_unavailable(self, mock_get_recommender, client):
        """Test POST request when recommender is unavailable"""
        mock_get_recommender.return_value = None
        
        response = client.post('/recommend', data={
            'movie_title': 'Toy Story',
            'user_id': '1',
            'alpha': '0.6',
            'top_n': '10'
        })
        
        assert response.status_code == 302  # Redirect


class TestAPIRoutes:
    """Test cases for API endpoints"""
    
    def test_api_recommend_success(self, client, sample_recommendations):
        """Test successful API recommendation request"""
        with patch('app.routes.get_recommender') as mock_get_recommender:
            mock_recommender = MagicMock()
            mock_recommender.hybrid_recommend.return_value = sample_recommendations
            mock_get_recommender.return_value = mock_recommender
            
            response = client.get('/api/recommend?movie_title=Toy%20Story&user_id=1&alpha=0.6&top_n=10')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'recommendations' in data
            assert len(data['recommendations']) == 2
            assert data['movie_title'] == 'Toy Story'
            assert data['user_id'] == 1
    
    def test_api_recommend_missing_title(self, client):
        """Test API request with missing movie title"""
        response = client.get('/api/recommend?user_id=1&alpha=0.6&top_n=10')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'movie_title parameter is required' in data['error']
    
    def test_api_recommend_missing_user_id(self, client):
        """Test API request with missing user ID"""
        response = client.get('/api/recommend?movie_title=Toy%20Story&alpha=0.6&top_n=10')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'user_id parameter is required' in data['error']
    
    def test_api_recommend_invalid_alpha(self, client):
        """Test API request with invalid alpha value"""
        response = client.get('/api/recommend?movie_title=Toy%20Story&user_id=1&alpha=1.5&top_n=10')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'alpha must be between 0 and 1' in data['error']
    
    def test_api_recommend_invalid_top_n(self, client):
        """Test API request with invalid top_n value"""
        response = client.get('/api/recommend?movie_title=Toy%20Story&user_id=1&alpha=0.6&top_n=100')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'top_n must be between 1 and 50' in data['error']
    
    @patch('app.routes.get_recommender')
    def test_api_recommend_no_recommendations(self, mock_get_recommender, client):
        """Test API request when no recommendations are found"""
        mock_recommender = MagicMock()
        mock_recommender.hybrid_recommend.return_value = pd.DataFrame()
        mock_get_recommender.return_value = mock_recommender
        
        response = client.get('/api/recommend?movie_title=Unknown%20Movie&user_id=1&alpha=0.6&top_n=10')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No recommendations found' in data['error']
    
    @patch('app.routes.get_recommender')
    def test_api_recommend_recommender_unavailable(self, mock_get_recommender, client):
        """Test API request when recommender is unavailable"""
        mock_get_recommender.return_value = None
        
        response = client.get('/api/recommend?movie_title=Toy%20Story&user_id=1&alpha=0.6&top_n=10')
        
        assert response.status_code == 503
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Recommendation system not available' in data['error']
    
    def test_api_status(self, client):
        """Test API status endpoint"""
        response = client.get('/api/status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data
        assert 'models' in data
        assert 'message' in data
    
    @patch('app.routes.get_recommender')
    def test_api_movies(self, mock_get_recommender, client):
        """Test API movies search endpoint"""
        # Mock movies DataFrame
        movies_df = pd.DataFrame([
            {'movie_id': 1, 'title': 'Toy Story', 'genre': 'Animation', 'rating': 3.8},
            {'movie_id': 2, 'title': 'Batman', 'genre': 'Action', 'rating': 4.2}
        ])
        
        mock_recommender = MagicMock()
        mock_recommender.movies_df = movies_df
        mock_get_recommender.return_value = mock_recommender
        
        response = client.get('/api/movies?q=toy')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'movies' in data
        assert len(data['movies']) == 1
        assert data['movies'][0]['title'] == 'Toy Story'
    
    @patch('app.routes.get_recommender')
    def test_api_movies_no_query(self, mock_get_recommender, client):
        """Test API movies endpoint without query"""
        movies_df = pd.DataFrame([
            {'movie_id': 1, 'title': 'Toy Story', 'genre': 'Animation', 'rating': 3.8}
        ])
        
        mock_recommender = MagicMock()
        mock_recommender.movies_df = movies_df
        mock_get_recommender.return_value = mock_recommender
        
        response = client.get('/api/movies')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'movies' in data
        assert len(data['movies']) == 1
    
    @patch('app.routes.get_recommender')
    def test_api_movies_recommender_unavailable(self, mock_get_recommender, client):
        """Test API movies endpoint when recommender is unavailable"""
        mock_get_recommender.return_value = None
        
        response = client.get('/api/movies?q=toy')
        
        assert response.status_code == 503
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Recommendation system not available' in data['error']


class TestErrorHandlers:
    """Test error handling routes"""
    
    def test_404_error(self, client):
        """Test 404 error handler"""
        response = client.get('/nonexistent-route')
        assert response.status_code == 404
        assert b'404' in response.data
    
    def test_500_error(self, client):
        """Test 500 error handler"""
        # This would require triggering an internal server error
        # For now, we'll just test that the route exists
        with patch('app.routes.render_template') as mock_render:
            mock_render.side_effect = Exception("Test error")
            response = client.get('/')
            # The error handler should catch this and return 500


if __name__ == "__main__":
    pytest.main([__file__])
