import os
import joblib
import pandas as pd
from typing import Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_models_dir(models_dir: str = 'models') -> str:
    """Ensure models directory exists"""
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def save_model(model: Any, filename: str, models_dir: str = 'models') -> str:
    """Save a model to the models directory"""
    ensure_models_dir(models_dir)
    filepath = os.path.join(models_dir, filename)
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")
    return filepath


def load_model(filename: str, models_dir: str = 'models') -> Any:
    """Load a model from the models directory"""
    filepath = os.path.join(models_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model


def save_dataframe(df: pd.DataFrame, filename: str, models_dir: str = 'models') -> str:
    """Save a DataFrame to the models directory"""
    ensure_models_dir(models_dir)
    filepath = os.path.join(models_dir, filename)
    df.to_pickle(filepath)
    logger.info(f"DataFrame saved to {filepath}")
    return filepath


def load_dataframe(filename: str, models_dir: str = 'models') -> pd.DataFrame:
    """Load a DataFrame from the models directory"""
    filepath = os.path.join(models_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"DataFrame file not found: {filepath}")
    
    df = pd.read_pickle(filepath)
    logger.info(f"DataFrame loaded from {filepath}")
    return df


def model_exists(filename: str, models_dir: str = 'models') -> bool:
    """Check if a model file exists"""
    filepath = os.path.join(models_dir, filename)
    return os.path.exists(filepath)


def list_models(models_dir: str = 'models') -> list:
    """List all model files in the models directory"""
    if not os.path.exists(models_dir):
        return []
    
    files = [f for f in os.listdir(models_dir) if f.endswith(('.pkl', '.joblib'))]
    return sorted(files)


def get_model_info(models_dir: str = 'models') -> Dict[str, Any]:
    """Get information about available models"""
    models = list_models(models_dir)
    info = {
        'models_dir': models_dir,
        'available_models': models,
        'total_models': len(models)
    }
    
    # Check for required models
    required_models = ['tfidf.pkl', 'tfidf_matrix.pkl', 'svd.pkl', 'movies_df.pkl']
    missing_models = [model for model in required_models if model not in models]
    
    info['missing_models'] = missing_models
    info['is_ready'] = len(missing_models) == 0
    
    return info


def cleanup_models(models_dir: str = 'models') -> None:
    """Remove all model files (use with caution)"""
    if not os.path.exists(models_dir):
        return
    
    models = list_models(models_dir)
    for model_file in models:
        filepath = os.path.join(models_dir, model_file)
        os.remove(filepath)
        logger.info(f"Removed {filepath}")
    
    logger.info(f"Cleaned up {len(models)} model files from {models_dir}")


def validate_models(models_dir: str = 'models') -> bool:
    """Validate that all required models can be loaded"""
    try:
        required_models = ['tfidf.pkl', 'tfidf_matrix.pkl', 'svd.pkl', 'movies_df.pkl']
        
        for model_file in required_models:
            load_model(model_file, models_dir)
        
        logger.info("All required models validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return False
