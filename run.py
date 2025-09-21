#!/usr/bin/env python3
"""
Flask application entrypoint for the hybrid movie recommendation system.

This script starts the Flask web application that provides:
- Web interface for movie recommendations
- REST API for programmatic access
- Real-time recommendation generation

Usage:
    python run.py

Environment Variables:
    FLASK_ENV: Set to 'development' for debug mode
    PORT: Port number (default: 5000)
    HOST: Host address (default: 127.0.0.1)
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from app import create_app
from app.utils import get_model_info, validate_models

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_models():
    """Check if models are available and valid"""
    model_info = get_model_info()
    
    if not model_info['is_ready']:
        logger.error("❌ Models not found or incomplete!")
        logger.error(f"Missing models: {', '.join(model_info['missing_models'])}")
        logger.error("Please run 'python train.py' first to train the models.")
        return False
    
    if not validate_models():
        logger.error("❌ Model validation failed!")
        logger.error("Please retrain the models with 'python train.py'")
        return False
    
    logger.info("✅ Models loaded and validated successfully!")
    return True


def main():
    """Main application function"""
    logger.info("=" * 60)
    logger.info("Starting Hybrid Movie Recommendation System")
    logger.info("=" * 60)
    
    # Check if models are available
    if not check_models():
        sys.exit(1)
    
    # Create Flask app
    app = create_app()
    
    # Get configuration from environment
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Flask application on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    logger.info("=" * 60)
    
    # Start the application
    try:
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("\n👋 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Application failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
