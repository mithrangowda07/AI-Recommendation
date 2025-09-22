from flask import Flask
import os
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'auth.login'

def create_app():
    # Get the directory where this file is located
    basedir = os.path.abspath(os.path.dirname(__file__))
    # Go up one level to the project root
    project_root = os.path.dirname(basedir)
    
    app = Flask(__name__, 
                template_folder=os.path.join(project_root, 'templates'),
                static_folder=os.path.join(project_root, 'static'))
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', f"sqlite:///{os.path.join(project_root, 'app.db')}")
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    
    # Import models to register with SQLAlchemy
    from app import models  # noqa: F401
    
    from app.routes import main
    app.register_blueprint(main)
    
    # Auth blueprint will be registered later
    try:
        from app.auth import auth  # type: ignore
        app.register_blueprint(auth)
    except Exception:
        pass
    
    # Create DB tables if not exist
    with app.app_context():
        db.create_all()
    
    return app
