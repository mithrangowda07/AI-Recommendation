#!/usr/bin/env python3
"""
Setup script for the Hybrid Movie Recommendation System.

This script helps with initial setup and dependency installation.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def create_virtual_environment():
    """Create a virtual environment"""
    if os.path.exists('venv'):
        print("📁 Virtual environment already exists")
        return True
    
    return run_command(
        f"{sys.executable} -m venv venv",
        "Creating virtual environment"
    )


def get_activation_command():
    """Get the command to activate the virtual environment"""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"


def install_dependencies():
    """Install required dependencies"""
    if platform.system() == "Windows":
        pip_command = "venv\\Scripts\\pip"
    else:
        pip_command = "venv/bin/pip"
    
    # First, upgrade pip to avoid compatibility issues
    print("🔄 Upgrading pip...")
    run_command(f"{pip_command} install --upgrade pip", "Upgrading pip")
    
    # Install dependencies with specific options for Windows
    if platform.system() == "Windows":
        # Try Windows-optimized installation first
        print("🔄 Attempting Windows-optimized installation...")
        success = run_command(
            f"{pip_command} install --only-binary=all -r requirements.txt",
            "Installing dependencies (Windows optimized)"
        )
        
        if not success:
            print("⚠️  Windows-optimized installation failed, trying alternative method...")
            # Fallback: install packages individually with pre-compiled wheels
            packages = [
                "Flask>=2.3.0",
                "Werkzeug>=2.3.0", 
                "pandas>=2.0.0",
                "numpy>=1.24.0",
                "scikit-learn>=1.3.0",
                "joblib>=1.3.0",
                "pytest>=7.4.0",
                "pytest-cov>=4.1.0"
            ]
            
            for package in packages:
                print(f"🔄 Installing {package}...")
                if not run_command(f"{pip_command} install --only-binary=all {package}", f"Installing {package}"):
                    print(f"⚠️  Failed to install {package}, trying without binary constraint...")
                    run_command(f"{pip_command} install {package}", f"Installing {package} (fallback)")
            
            return True
        return success
    else:
        return run_command(
            f"{pip_command} install -r requirements.txt",
            "Installing dependencies"
        )


def check_data_files():
    """Check if required data files exist"""
    required_files = ['data/movies.json', 'data/ratings.csv']
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required data files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease ensure the data files are in the correct location.")
        return False
    
    print("✅ All required data files found")
    return True


def main():
    """Main setup function"""
    print("🎬 Hybrid Movie Recommendation System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check data files
    if not check_data_files():
        print("\n⚠️  Please add the required data files before continuing.")
        print("   The system needs:")
        print("   - data/movies.json (movie metadata)")
        print("   - data/ratings.csv (user ratings)")
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    print(f"   {get_activation_command()}")
    print("\n2. Train the models:")
    print("   python train.py")
    print("\n3. Start the application:")
    print("   python run.py")
    print("\n4. Open your browser and go to: http://127.0.0.1:5000")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
