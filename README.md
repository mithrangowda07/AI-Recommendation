# 🎬 Hybrid Movie Recommendation System

A sophisticated Flask web application that provides personalized movie recommendations using a hybrid approach combining **content-based filtering** (TF-IDF) and **collaborative filtering** (SVD). The system analyzes movie metadata, user ratings, and preferences to deliver accurate and diverse recommendations.

## ✨ Features

- **🤖 Hybrid AI System**: Combines content-based and collaborative filtering for superior recommendations
- **🎯 Customizable Balance**: Adjust the weight between content similarity and user preferences
- **🌐 Web Interface**: Beautiful, responsive Bootstrap UI with dark/light theme toggle
- **🔌 REST API**: Programmatic access for integration with other applications
- **📊 Detailed Scoring**: Transparent scoring system showing content and collaborative scores
- **⚡ Real-time Results**: Instant recommendations with comprehensive movie information
- **🧪 Comprehensive Testing**: Full test suite with pytest for reliability

## 🏗️ Architecture

### Content-Based Filtering
- **TF-IDF Vectorization**: Analyzes movie overviews, genres, cast, directors, and taglines
- **Cosine Similarity**: Finds movies with similar content characteristics
- **Text Processing**: Advanced preprocessing with n-grams and stop word removal

### Collaborative Filtering
- **SVD (Singular Value Decomposition)**: Matrix factorization for user-item interactions
- **Rating Prediction**: Predicts user ratings for unseen movies
- **User Similarity**: Leverages patterns from similar users

### Hybrid Approach
- **Weighted Combination**: `final_score = α × content_score + (1-α) × collab_score`
- **Normalized Scores**: Both components normalized to 0-1 range
- **Configurable Balance**: Users can adjust α parameter (0-1)

## 📁 Project Structure

```
AI-Recommendation/
├── app/                    # Flask application package
│   ├── __init__.py        # Flask app factory
│   ├── routes.py          # Web routes and API endpoints
│   ├── recommender.py     # Core recommendation logic
│   └── utils.py           # Utility functions for model management
├── data/                  # Data files
│   ├── movies.json        # Movie metadata (movie_id, title, overview, genre, etc.)
│   └── ratings.csv        # User ratings (userId, movieId, rating)
├── models/                # Trained models (created after training)
│   ├── tfidf.pkl         # TF-IDF vectorizer
│   ├── tfidf_matrix.pkl  # TF-IDF similarity matrix
│   ├── svd.pkl           # SVD collaborative filtering model
│   └── movies_df.pkl     # Processed movies DataFrame
├── templates/             # HTML templates
│   ├── base.html         # Base template with Bootstrap
│   ├── index.html        # Homepage with recommendation form
│   ├── results.html      # Results display page
│   ├── 404.html          # 404 error page
│   └── 500.html          # 500 error page
├── static/               # Static assets (CSS, JS, images)
├── tests/                # Test suite
│   ├── __init__.py
│   ├── test_recommender.py  # Recommender logic tests
│   └── test_routes.py       # Flask routes tests
├── train.py              # Model training script
├── run.py                # Flask application entrypoint
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or download the project**
   ```bash
   # If using git
   git clone <repository-url>
   cd AI-Recommendation
   ```

2. **Create a virtual environment**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the models**
   ```bash
   python setup.py
   python train.py
   ```
   This will:
   - Load and preprocess the movie and rating data
   - Train the TF-IDF content-based model
   - Train the SVD collaborative filtering model
   - Save all models to the `models/` directory

5. **Start the Flask application**
   ```bash
   python run.py
   ```

6. **Access the application**
   - Open your browser and go to `http://127.0.0.1:5000`
   - Enter a movie title and user ID to get recommendations

## 🎯 Usage

### Web Interface

1. **Enter Movie Title**: Type the name of a movie you like
2. **Enter User ID**: Provide a user ID (1-1000) for personalized recommendations
3. **Adjust Balance**: Use the slider to balance content vs. collaborative filtering (0-1)
4. **Select Count**: Choose how many recommendations to receive (5-20)
5. **Get Recommendations**: Click the button to see personalized results
6. **Theme**: Use the Theme toggle in the navbar to switch light/dark

### Homepage Sections
- "Popular in <country>" and "In <language>" cards show curated tiles with improved spacing and compact headings for readability.

### API Endpoints

#### Get Recommendations
```http
GET /api/recommend?movie_title=Toy%20Story&user_id=1&alpha=0.6&top_n=10
```

**Parameters:**
- `movie_title` (required): Title of the reference movie
- `user_id` (required): User ID for personalization (1-1000)
- `alpha` (optional): Balance between content and collaborative filtering (0-1, default: 0.6)
- `top_n` (optional): Number of recommendations (1-50, default: 10)

**Response:**
```json
{
  "movie_title": "Toy Story",
  "user_id": 1,
  "alpha": 0.6,
  "top_n": 10,
  "recommendations": [
    {
      "movie_id": 2,
      "title": "Toy Story 2",
      "genre": "Animation, Comedy, Family",
      "overview": "Woody is stolen by a toy collector...",
      "rating": 3.9,
      "score": 0.85,
      "content_score": 0.8,
      "collab_score": 0.9
    }
  ]
}
```

#### Check System Status
```http
GET /api/status
```

#### Search Movies
```http
GET /api/movies?q=batman&limit=20
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_recommender.py

# Run with verbose output
pytest -v
```

## 📊 Data Format

### Movies Data (`data/movies.json`)
```json
[
  {
    "movie_id": 862,
    "title": "Toy Story",
    "overview": "A story about toys that come to life...",
    "genre": "Animation, Comedy, Family",
    "cast": "Tom Hanks, Tim Allen, Don Rickles...",
    "director": "John Lasseter",
    "release_date": "30-10-1995",
    "rating": 3.87,
    "original_language": "en",
    "production_company": "Pixar Animation Studios",
    "production_country": "US",
    "tagline": "A toy story",
    "runtime": 81
  }
]
```

### Ratings Data (`data/ratings.csv`)
```csv
userId,movieId,rating
1,31,2.5
1,1029,3
1,1061,3
```

## ⚙️ Configuration

### Environment Variables

- `FLASK_ENV`: Set to `development` for debug mode
- `PORT`: Port number (default: 5000)
- `HOST`: Host address (default: 127.0.0.1)
- `SECRET_KEY`: Flask secret key for sessions

### Model Parameters

You can modify the recommendation behavior by editing `app/recommender.py`:

- **TF-IDF Parameters**: `ngram_range`, `max_features`, `stop_words`
- **SVD Parameters**: `n_factors`, `random_state`
- **Hybrid Weights**: Default `alpha` value and normalization

## 🔧 Advanced Usage

### Custom Model Training

```python
from app.recommender import HybridMovieRecommender

# Create recommender with custom parameters
recommender = HybridMovieRecommender(
    movies_path='data/movies.json',
    ratings_path='data/ratings.csv',
    models_dir='models'
)

# Load and prepare data
recommender.load_data()
recommender.prepare_content_data()
recommender.train_collaborative_model()

# Save models
recommender.save_models()
```

### Programmatic Recommendations

```python
from app.recommender import hybrid_recommend

# Get recommendations
recommendations = hybrid_recommend(
    user_id=1,
    movie_title="The Dark Knight",
    alpha=0.7,  # More content-based
    top_n=5
)

print(recommendations[['title', 'score', 'genre']])
```

## 🐛 Troubleshooting

### Common Issues

1. **"Models not found" error**
   - Run `python train.py` to train the models first

2. **Memory issues during training**
   - Reduce `max_features` in TF-IDF parameters
   - Use a smaller subset of the data for testing

3. **No recommendations returned**
   - Check if the movie title exists in the dataset
   - Try different user IDs
   - Verify the movie title spelling

4. **Slow performance**
   - Ensure models are properly cached
   - Consider reducing the number of features
   - Use a more powerful machine for training

### Debug Mode

Enable debug mode for detailed logging:

```bash
export FLASK_ENV=development
python run.py
```

## 📈 Performance

### Model Training Time
- **Small dataset** (< 1000 movies): ~2-5 minutes
- **Medium dataset** (1000-10000 movies): ~10-30 minutes
- **Large dataset** (> 10000 movies): ~30+ minutes

### Recommendation Speed
- **First request**: ~1-3 seconds (model loading)
- **Subsequent requests**: ~0.1-0.5 seconds

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **scikit-learn**: For TF-IDF vectorization and cosine similarity
- **Surprise**: For collaborative filtering with SVD
- **Flask**: For the web framework
- **Bootstrap**: For the responsive UI components
- **Pandas & NumPy**: For data manipulation and analysis

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the test cases for usage examples
3. Open an issue with detailed error information
4. Include your Python version and operating system

---

**Happy Movie Recommending! 🍿**
