# Makefile for Hybrid Movie Recommendation System

.PHONY: help install train run test clean setup

# Default target
help:
	@echo "🎬 Hybrid Movie Recommendation System"
	@echo "====================================="
	@echo ""
	@echo "Available commands:"
	@echo "  setup     - Run initial setup (create venv, install deps)"
	@echo "  install   - Install dependencies"
	@echo "  train     - Train the recommendation models"
	@echo "  run       - Start the Flask application"
	@echo "  test      - Run the test suite"
	@echo "  test-cov  - Run tests with coverage report"
	@echo "  clean     - Clean up generated files"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Quick start:"
	@echo "  make setup && make train && make run"

# Setup virtual environment and install dependencies
setup:
	@echo "🔄 Setting up the project..."
	python setup.py

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

# Train the models
train:
	@echo "🤖 Training recommendation models..."
	python train.py

# Run the Flask application
run:
	@echo "🚀 Starting Flask application..."
	python run.py

# Run tests
test:
	@echo "🧪 Running test suite..."
	pytest

# Run tests with coverage
test-cov:
	@echo "🧪 Running tests with coverage..."
	pytest --cov=app --cov-report=html --cov-report=term

# Clean up generated files
clean:
	@echo "🧹 Cleaning up..."
	rm -rf models/*.pkl
	rm -rf models/*.joblib
	rm -rf __pycache__
	rm -rf app/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf *.log
	@echo "✅ Cleanup completed"

# Development setup
dev-setup: setup train
	@echo "🎉 Development environment ready!"
	@echo "Run 'make run' to start the application"
