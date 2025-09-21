@echo off
echo 🎬 Hybrid Movie Recommendation System - Windows Installation
echo ============================================================

echo.
echo 🔄 Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo 🔄 Upgrading pip...
python -m pip install --upgrade pip

echo.
echo 🔄 Installing core dependencies...
pip install Flask Werkzeug

echo.
echo 🔄 Installing data science packages...
pip install pandas numpy scikit-learn

echo.
echo 🔄 Installing recommendation system...
pip install scikit-surprise

echo.
echo 🔄 Installing utilities...
pip install joblib pytest pytest-cov

echo.
echo ✅ Installation completed!
echo.
echo Next steps:
echo 1. Run: python train.py
echo 2. Run: python run.py
echo 3. Open: http://127.0.0.1:5000
echo.
pause
