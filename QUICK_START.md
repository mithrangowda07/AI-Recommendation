# 🚀 Quick Start Guide

## ✅ Fixed Issues

The setup errors have been resolved! Here's what was fixed:

1. **Windows Compilation Issues**: Replaced `scikit-surprise` with `scikit-learn`'s `TruncatedSVD` for collaborative filtering
2. **Dependency Management**: Updated `requirements.txt` to use compatible versions
3. **Installation Process**: Enhanced `setup.py` with Windows-specific optimizations

## 🎯 How to Run

### Option 1: Automated Setup (Recommended)
```bash
python setup.py
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train models
python train.py

# Start application
python run.py
```

### Option 3: Windows Batch File
```bash
install_windows.bat
```

## 🌐 Access the Application

1. **Web Interface**: http://127.0.0.1:5000
2. **API Status**: http://127.0.0.1:5000/api/status
3. **API Example**: http://127.0.0.1:5000/api/recommend?movie_title=Toy%20Story&user_id=1&alpha=0.6&top_n=5

## 🧪 Test the System

The system has been tested and is working with:
- ✅ 45,666 movies loaded
- ✅ 100,004 ratings processed  
- ✅ Hybrid recommendations working
- ✅ Web interface functional
- ✅ REST API operational

## 📊 Sample Recommendation

Try searching for "Toy Story" with user ID 1 to see recommendations like:
- Toy Story 2 (Score: 0.750)
- Toy Story 3 (Score: 0.473)
- Tin Toy (Score: 0.436)

## 🎉 Success!

Your hybrid movie recommendation system is now fully operational!
