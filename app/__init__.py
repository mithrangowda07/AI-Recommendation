from flask import Flask
import os

def create_app():
    # Get the directory where this file is located
    basedir = os.path.abspath(os.path.dirname(__file__))
    # Go up one level to the project root
    project_root = os.path.dirname(basedir)
    
    app = Flask(__name__, 
                template_folder=os.path.join(project_root, 'templates'),
                static_folder=os.path.join(project_root, 'static'))
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
    
    from app.routes import main
    app.register_blueprint(main)
    
    return app
