from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
from app.recommender import get_recommender, hybrid_recommend
from app.utils import get_model_info, validate_models
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

main = Blueprint('main', __name__)


@main.route('/')
def index():
    """Homepage with recommendation form"""
    # Check if models are ready
    model_info = get_model_info()
    
    if not model_info['is_ready']:
        flash('Models not found. Please run train.py first to train the recommendation models.', 'warning')
    
    return render_template('index.html', model_info=model_info)


@main.route('/recommend', methods=['POST'])
def recommend():
    """Handle recommendation form submission"""
    try:
        # Get form data
        movie_title = request.form.get('movie_title', '').strip()
        user_id = request.form.get('user_id', '').strip()
        alpha = float(request.form.get('alpha', 0.6))
        top_n = int(request.form.get('top_n', 10))
        
        # Validate inputs
        if not movie_title:
            flash('Please enter a movie title.', 'error')
            return redirect(url_for('main.index'))
        
        if not user_id:
            flash('Please enter a user ID.', 'error')
            return redirect(url_for('main.index'))
        
        try:
            user_id = int(user_id)
        except ValueError:
            flash('User ID must be a number.', 'error')
            return redirect(url_for('main.index'))
        
        if not (0 <= alpha <= 1):
            flash('Alpha must be between 0 and 1.', 'error')
            return redirect(url_for('main.index'))
        
        if not (1 <= top_n <= 50):
            flash('Number of recommendations must be between 1 and 50.', 'error')
            return redirect(url_for('main.index'))
        
        # Get recommendations
        recommender = get_recommender()
        if recommender is None:
            flash('Recommendation system not available. Please run train.py first.', 'error')
            return redirect(url_for('main.index'))
        
        recommendations = hybrid_recommend(user_id, movie_title, alpha, top_n)
        
        if recommendations.empty:
            flash(f'No recommendations found for "{movie_title}". Please try a different movie title.', 'warning')
            return redirect(url_for('main.index'))
        
        # Convert to list of dicts for template
        recommendations_list = recommendations.to_dict('records')
        
        return render_template('results.html', 
                             recommendations=recommendations_list,
                             movie_title=movie_title,
                             user_id=user_id,
                             alpha=alpha,
                             top_n=top_n)
        
    except Exception as e:
        logger.error(f"Error in recommend route: {e}")
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('main.index'))


@main.route('/api/recommend')
def api_recommend():
    """JSON API endpoint for recommendations"""
    try:
        # Get query parameters
        movie_title = request.args.get('movie_title', '').strip()
        user_id = request.args.get('user_id', type=int)
        alpha = request.args.get('alpha', 0.6, type=float)
        top_n = request.args.get('top_n', 10, type=int)
        
        # Validate inputs
        if not movie_title:
            return jsonify({'error': 'movie_title parameter is required'}), 400
        
        if user_id is None:
            return jsonify({'error': 'user_id parameter is required'}), 400
        
        if not (0 <= alpha <= 1):
            return jsonify({'error': 'alpha must be between 0 and 1'}), 400
        
        if not (1 <= top_n <= 50):
            return jsonify({'error': 'top_n must be between 1 and 50'}), 400
        
        # Get recommendations
        recommender = get_recommender()
        if recommender is None:
            return jsonify({'error': 'Recommendation system not available'}), 503
        
        recommendations = hybrid_recommend(user_id, movie_title, alpha, top_n)
        
        if recommendations.empty:
            return jsonify({'error': f'No recommendations found for "{movie_title}"'}), 404
        
        # Convert to JSON-serializable format
        result = {
            'movie_title': movie_title,
            'user_id': user_id,
            'alpha': alpha,
            'top_n': top_n,
            'recommendations': recommendations.to_dict('records')
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in API recommend: {e}")
        return jsonify({'error': str(e)}), 500


@main.route('/api/status')
def api_status():
    """API endpoint to check system status"""
    try:
        model_info = get_model_info()
        is_ready = validate_models() if model_info['is_ready'] else False
        
        status = {
            'status': 'ready' if is_ready else 'not_ready',
            'models': model_info,
            'message': 'System ready' if is_ready else 'Models not found or invalid'
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error in status check: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@main.route('/api/movies')
def api_movies():
    """API endpoint to search movies"""
    try:
        query = request.args.get('q', '').strip()
        limit = request.args.get('limit', 20, type=int)
        
        recommender = get_recommender()
        if recommender is None:
            return jsonify({'error': 'Recommendation system not available'}), 503
        
        movies_df = recommender.movies_df
        
        if query:
            # Filter movies by title (case-insensitive)
            mask = movies_df['title'].str.contains(query, case=False, na=False)
            movies = movies_df[mask]
        else:
            movies = movies_df
        
        # Limit results
        movies = movies.head(limit)
        
        # Convert to list of dicts
        result = {
            'query': query,
            'total': len(movies),
            'movies': movies[['movie_id', 'title', 'genre', 'rating']].to_dict('records')
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in movies search: {e}")
        return jsonify({'error': str(e)}), 500


@main.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


@main.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500
