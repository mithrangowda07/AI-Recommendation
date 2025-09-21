#!/usr/bin/env python3
"""
Training script for the hybrid movie recommendation system.

This script:
1. Loads the movies and ratings data
2. Prepares the data for content-based filtering (TF-IDF)
3. Trains the collaborative filtering model (SVD)
4. Saves all models and data for the Flask application

Usage:
    python train.py
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.recommender import HybridMovieRecommender
from app.utils import get_model_info, cleanup_models

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main training function"""
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("Starting Hybrid Movie Recommendation System Training")
    logger.info("=" * 60)
    
    # Check if data files exist
    movies_path = 'data/movies.json'
    ratings_path = 'data/ratings.csv'
    
    if not os.path.exists(movies_path):
        logger.error(f"Movies data file not found: {movies_path}")
        sys.exit(1)
    
    if not os.path.exists(ratings_path):
        logger.error(f"Ratings data file not found: {ratings_path}")
        sys.exit(1)
    
    # Clean up existing models (optional)
    cleanup_choice = input("Clean up existing models? (y/N): ").strip().lower()
    if cleanup_choice in ['y', 'yes']:
        logger.info("Cleaning up existing models...")
        cleanup_models()
    
    try:
        # Initialize recommender
        logger.info("Initializing recommender...")
        recommender = HybridMovieRecommender(movies_path, ratings_path)
        
        # Load data
        logger.info("Loading data...")
        recommender.load_data()
        
        # Prepare content data
        logger.info("Preparing content-based filtering data...")
        recommender.prepare_content_data()
        
        # Train collaborative filtering model
        logger.info("Training collaborative filtering model...")
        recommender.train_collaborative_model()
        
        # Save models
        logger.info("Saving models...")
        recommender.save_models()
        
        # Validate saved models
        logger.info("Validating saved models...")
        model_info = get_model_info()
        
        if model_info['is_ready']:
            logger.info("✅ All models saved and validated successfully!")
            logger.info(f"Models saved in: {model_info['models_dir']}")
            logger.info(f"Available models: {', '.join(model_info['available_models'])}")
        else:
            logger.error("❌ Model validation failed!")
            logger.error(f"Missing models: {', '.join(model_info['missing_models'])}")
            sys.exit(1)
        
        # Training summary
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("Training Summary")
        logger.info("=" * 60)
        logger.info(f"Total training time: {training_time:.2f} seconds")
        logger.info(f"Movies processed: {len(recommender.movies_df)}")
        logger.info(f"Ratings processed: {len(recommender.ratings_df)}")
        logger.info(f"TF-IDF matrix shape: {recommender.tfidf_matrix.shape}")
        logger.info("Models ready for Flask application!")
        logger.info("=" * 60)
        
        # Test recommendation
        logger.info("Testing recommendation system...")
        try:
            test_recommendations = recommender.hybrid_recommend(
                user_id=1, 
                movie_title="Toy Story", 
                alpha=0.6, 
                top_n=5
            )
            
            if not test_recommendations.empty:
                logger.info("✅ Test recommendation successful!")
                logger.info("Sample recommendations:")
                for _, rec in test_recommendations.head(3).iterrows():
                    logger.info(f"  - {rec['title']} (Score: {rec['score']:.3f})")
            else:
                logger.warning("⚠️ Test recommendation returned no results")
                
        except Exception as e:
            logger.warning(f"⚠️ Test recommendation failed: {e}")
        
        logger.info("\n🎉 Training completed successfully!")
        logger.info("You can now run the Flask application with: python run.py")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.exception("Full error traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
