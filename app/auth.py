from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from flask_login import login_user, logout_user, login_required, current_user
from sqlalchemy.exc import IntegrityError

from app import db
from app.models import User, Preference
from app.recommender import get_recommender

# ISO 639-1 language codes to English names
# Kept local to avoid adding a new dependency
LANGUAGE_CODE_TO_NAME = {
    'aa': 'Afar', 'ab': 'Abkhaz', 'ae': 'Avestan', 'af': 'Afrikaans', 'ak': 'Akan',
    'am': 'Amharic', 'an': 'Aragonese', 'ar': 'Arabic', 'as': 'Assamese', 'av': 'Avaric',
    'ay': 'Aymara', 'az': 'Azerbaijani', 'ba': 'Bashkir', 'be': 'Belarusian', 'bg': 'Bulgarian',
    'bh': 'Bihari languages', 'bi': 'Bislama', 'bm': 'Bambara', 'bn': 'Bengali', 'bo': 'Tibetan',
    'br': 'Breton', 'bs': 'Bosnian', 'ca': 'Catalan', 'ce': 'Chechen', 'ch': 'Chamorro',
    'co': 'Corsican', 'cr': 'Cree', 'cs': 'Czech', 'cu': 'Church Slavic', 'cv': 'Chuvash',
    'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'dv': 'Dhivehi', 'dz': 'Dzongkha',
    'ee': 'Ewe', 'el': 'Greek', 'en': 'English', 'eo': 'Esperanto', 'es': 'Spanish',
    'et': 'Estonian', 'eu': 'Basque', 'fa': 'Persian', 'ff': 'Fulah', 'fi': 'Finnish',
    'fj': 'Fijian', 'fo': 'Faroese', 'fr': 'French', 'fy': 'Western Frisian', 'ga': 'Irish',
    'gd': 'Scottish Gaelic', 'gl': 'Galician', 'gn': 'Guarani', 'gu': 'Gujarati', 'gv': 'Manx',
    'ha': 'Hausa', 'he': 'Hebrew', 'hi': 'Hindi', 'ho': 'Hiri Motu', 'hr': 'Croatian',
    'ht': 'Haitian', 'hu': 'Hungarian', 'hy': 'Armenian', 'hz': 'Herero', 'ia': 'Interlingua',
    'id': 'Indonesian', 'ie': 'Interlingue', 'ig': 'Igbo', 'ii': 'Sichuan Yi', 'ik': 'Inupiaq',
    'io': 'Ido', 'is': 'Icelandic', 'it': 'Italian', 'iu': 'Inuktitut', 'ja': 'Japanese',
    'jv': 'Javanese', 'ka': 'Georgian', 'kg': 'Kongo', 'kk': 'Kazakh', 'kl': 'Kalaallisut',
    'km': 'Central Khmer', 'kn': 'Kannada', 'ko': 'Korean', 'kr': 'Kanuri', 'ks': 'Kashmiri',
    'ku': 'Kurdish', 'kv': 'Komi', 'kw': 'Cornish', 'ky': 'Kirghiz', 'la': 'Latin',
    'lb': 'Luxembourgish', 'lg': 'Ganda', 'li': 'Limburgan', 'ln': 'Lingala', 'lo': 'Lao',
    'lt': 'Lithuanian', 'lu': 'Luba-Katanga', 'lv': 'Latvian', 'mg': 'Malagasy', 'mh': 'Marshallese',
    'mi': 'Maori', 'mk': 'Macedonian', 'ml': 'Malayalam', 'mn': 'Mongolian', 'mr': 'Marathi',
    'ms': 'Malay', 'mt': 'Maltese', 'my': 'Burmese', 'na': 'Nauru', 'nb': 'Norwegian Bokmål',
    'nd': 'North Ndebele', 'ne': 'Nepali', 'ng': 'Ndonga', 'nl': 'Dutch', 'nn': 'Norwegian Nynorsk',
    'no': 'Norwegian', 'nr': 'South Ndebele', 'nv': 'Navajo', 'ny': 'Chichewa', 'oc': 'Occitan',
    'oj': 'Ojibwa', 'om': 'Oromo', 'or': 'Odia', 'os': 'Ossetian', 'pa': 'Punjabi',
    'pi': 'Pali', 'pl': 'Polish', 'ps': 'Pashto', 'pt': 'Portuguese', 'qu': 'Quechua',
    'rm': 'Romansh', 'rn': 'Rundi', 'ro': 'Romanian', 'ru': 'Russian', 'rw': 'Kinyarwanda',
    'sa': 'Sanskrit', 'sc': 'Sardinian', 'sd': 'Sindhi', 'se': 'Northern Sami', 'sg': 'Sango',
    'si': 'Sinhala', 'sk': 'Slovak', 'sl': 'Slovenian', 'sm': 'Samoan', 'sn': 'Shona',
    'so': 'Somali', 'sq': 'Albanian', 'sr': 'Serbian', 'ss': 'Swati', 'st': 'Southern Sotho',
    'su': 'Sundanese', 'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil', 'te': 'Telugu',
    'tg': 'Tajik', 'th': 'Thai', 'ti': 'Tigrinya', 'tk': 'Turkmen', 'tl': 'Tagalog',
    'tn': 'Tswana', 'to': 'Tonga', 'tr': 'Turkish', 'ts': 'Tsonga', 'tt': 'Tatar',
    'tw': 'Twi', 'ty': 'Tahitian', 'ug': 'Uighur', 'uk': 'Ukrainian', 'ur': 'Urdu',
    'uz': 'Uzbek', 've': 'Venda', 'vi': 'Vietnamese', 'vo': 'Volapük', 'wa': 'Walloon',
    'wo': 'Wolof', 'xh': 'Xhosa', 'yi': 'Yiddish', 'yo': 'Yoruba', 'za': 'Zhuang', 'zh': 'Chinese',
    'zu': 'Zulu'
}


auth = Blueprint('auth', __name__)


@auth.route('/signup')
def signup():
    # Redirect old endpoint to the new multi-step start
    return redirect(url_for('auth.signup_start'))


@auth.route('/signup/start', methods=['GET', 'POST'])
def signup_start():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()
        if not email or not password:
            flash('Email and password are required.', 'error')
            return redirect(url_for('auth.signup_start'))
        # Save credentials temporarily in session
        session['signup'] = {'email': email, 'password': password}
        return redirect(url_for('auth.signup_country'))
    return render_template('signup_step1.html')


@auth.route('/signup/country', methods=['GET', 'POST'])
def signup_country():
    countries, languages, genres = _get_distinct_metadata()
    if request.method == 'POST':
        selected = request.form.getlist('country')
        data = session.get('signup', {})
        data['country'] = selected
        session['signup'] = data
        return redirect(url_for('auth.signup_language'))
    return render_template('signup_countries.html', countries=countries)


@auth.route('/signup/language', methods=['GET', 'POST'])
def signup_language():
    countries, languages, genres = _get_distinct_metadata()
    if request.method == 'POST':
        selected = request.form.getlist('language')
        data = session.get('signup', {})
        data['language'] = selected
        session['signup'] = data
        return redirect(url_for('auth.signup_genre'))
    return render_template('signup_languages.html', languages=languages)


@auth.route('/signup/genre', methods=['GET', 'POST'])
def signup_genre():
    countries, languages, genres = _get_distinct_metadata()
    if request.method == 'POST':
        selected = request.form.getlist('genre')
        data = session.get('signup', {})
        data['genre'] = selected
        session['signup'] = data
        email = (data.get('email') or '').strip().lower()
        password = (data.get('password') or '').strip()
        if not email or not password:
            flash('Session expired. Please start again.', 'error')
            return redirect(url_for('auth.signup_start'))

        # Create user and preferences
        user = User(email=email)
        user.set_password(password)
        try:
            db.session.add(user)
            db.session.flush()
            pref = Preference(
                user_id=user.id,
                country=','.join(sorted(set(data.get('country', [])))) or None,
                language=','.join(sorted(set(data.get('language', [])))) or None,
                genre=','.join(sorted(set(data.get('genre', [])))) or None,
            )
            db.session.add(pref)
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            flash('Email is already registered.', 'error')
            return redirect(url_for('auth.signup_start'))
        finally:
            session.pop('signup', None)

        login_user(user)
        flash('Account created successfully!', 'success')
        return redirect(url_for('main.index'))
    return render_template('signup_genres.html', genres=genres)


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
            code_set = {c.strip() for c in df['original_language'].dropna().astype(str).tolist()}
            # Build (code, name) tuples; fallback to code if unknown
            languages = sorted(
                [(code, LANGUAGE_CODE_TO_NAME.get(code.lower(), code.upper())) for code in code_set],
                key=lambda x: x[1]
            )
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


