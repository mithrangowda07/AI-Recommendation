from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from sqlalchemy.exc import IntegrityError

from app import db
from app.models import User, Preference
from app.recommender import get_recommender


auth = Blueprint('auth', __name__)


@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()
        # Accept multi-select values
        countries = request.form.getlist('country') or [request.form.get('country', '').strip()]
        languages = request.form.getlist('language') or [request.form.get('language', '').strip()]
        genres = request.form.getlist('genre') or [request.form.get('genre', '').strip()]

        if not email or not password:
            flash('Email and password are required.', 'error')
            return redirect(url_for('auth.signup'))

        user = User(email=email)
        user.set_password(password)

        try:
            db.session.add(user)
            db.session.flush()

            pref = Preference(
                user_id=user.id,
                country=','.join(sorted({v for v in countries if v})) or None,
                language=','.join(sorted({v for v in languages if v})) or None,
                genre=','.join(sorted({v for v in genres if v})) or None,
            )
            db.session.add(pref)
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            flash('Email is already registered.', 'error')
            return redirect(url_for('auth.signup'))

        login_user(user)
        flash('Account created successfully!', 'success')
        return redirect(url_for('main.index'))

    # Build choices from movies dataset
    countries, languages, genres = _get_distinct_metadata()
    return render_template('signup.html', countries=countries, languages=languages, genres=genres)


@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()

        if not email or not password:
            flash('Email and password are required.', 'error')
            return redirect(url_for('auth.login'))

        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            flash('Invalid email or password.', 'error')
            return redirect(url_for('auth.login'))

        login_user(user)
        flash('Logged in successfully.', 'success')
        return redirect(url_for('main.index'))

    return render_template('login.html')


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out.', 'success')
    return redirect(url_for('main.index'))


@auth.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    pref = Preference.query.filter_by(user_id=current_user.id).first()
    if request.method == 'POST':
        countries = request.form.getlist('country') or [request.form.get('country', '').strip()]
        languages = request.form.getlist('language') or [request.form.get('language', '').strip()]
        genres = request.form.getlist('genre') or [request.form.get('genre', '').strip()]

        if not pref:
            pref = Preference(user_id=current_user.id)
            db.session.add(pref)

        pref.country = ','.join(sorted({v for v in countries if v})) or None
        pref.language = ','.join(sorted({v for v in languages if v})) or None
        pref.genre = ','.join(sorted({v for v in genres if v})) or None
        db.session.commit()
        flash('Preferences updated.', 'success')
        return redirect(url_for('auth.profile'))

    countries, languages, genres = _get_distinct_metadata()
    # Pre-split pref values for template convenience
    selected_countries = set([v.strip() for v in (pref.country or '').split(',') if v.strip()]) if pref else set()
    selected_languages = set([v.strip() for v in (pref.language or '').split(',') if v.strip()]) if pref else set()
    selected_genres = set([v.strip() for v in (pref.genre or '').split(',') if v.strip()]) if pref else set()
    return render_template('profile.html', pref=pref, countries=countries, languages=languages, genres=genres,
                           selected_countries=selected_countries, selected_languages=selected_languages, selected_genres=selected_genres)


def _get_distinct_metadata():
    recommender = get_recommender()
    countries = []
    languages = []
    genres = []
    if recommender is not None and recommender.movies_df is not None:
        df = recommender.movies_df
        if 'production_country' in df.columns:
            country_set = set()
            for raw in df['production_country'].dropna().astype(str).tolist():
                for c in raw.split(','):
                    c = c.strip()
                    if c:
                        country_set.add(c)
            countries = sorted(country_set)
        if 'original_language' in df.columns:
            languages = sorted({c.strip() for c in df['original_language'].dropna().astype(str).tolist()})
        if 'genre' in df.columns:
            # split comma-separated genres into unique set
            genre_set = set()
            for val in df['genre'].dropna().astype(str).tolist():
                for g in val.split(','):
                    g = g.strip()
                    if g:
                        genre_set.add(g)
            genres = sorted(genre_set)
    return countries, languages, genres


