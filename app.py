"""
app.py
======
Main entry point for the Sentiment Analysis Dashboard.
A professional dark-themed analytics app built with Streamlit.

Run with:
    streamlit run app.py
"""

# ── Standard Library ─────────────────────────────────────────────────────────
import os
import io
import re
import time
import random
import string
import pickle
import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Third-Party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── NLTK ──────────────────────────────────────────────────────────────────────
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('punkt',     quiet=True)

# ── Local Module ──────────────────────────────────────────────────────────────
from train_model import get_model, predict_single, predict_batch, clean_text


# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be the very first Streamlit call)
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title  = "SentiScope | ML Sentiment Analyzer",
    page_icon   = "🧠",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


# ════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS  — dark glassmorphism design
# ════════════════════════════════════════════════════════════════════════════

GLOBAL_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root Palette ── */
:root {
    --bg-primary   : #0a0a0f;
    --bg-secondary : #111118;
    --bg-card      : rgba(20, 20, 30, 0.85);
    --bg-glass     : rgba(255,255,255,0.04);
    --border       : rgba(255,255,255,0.07);
    --accent-green : #00ff87;
    --accent-yellow: #ffe600;
    --accent-red   : #ff4d6d;
    --accent-blue  : #00cfff;
    --text-primary : #e8e8f0;
    --text-secondary: #8888aa;
    --text-muted   : #55556a;
    --positive     : #00ff87;
    --negative     : #ff4d6d;
    --neutral      : #ffe600;
    --radius-sm    : 8px;
    --radius-md    : 14px;
    --radius-lg    : 20px;
    --shadow-card  : 0 4px 32px rgba(0,0,0,0.45);
    --shadow-glow  : 0 0 24px rgba(0,255,135,0.15);
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebarContent"] { padding-top: 1rem !important; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 1400px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: #333345; border-radius: 3px; }

/* ── Glassmorphism Cards ── */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: var(--shadow-card);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 1rem;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-card), var(--shadow-glow);
}

/* ── Metric Cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.3rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.25s ease;
    backdrop-filter: blur(12px);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent-green);
    border-radius: 2px 2px 0 0;
}
.metric-card.negative::before  { background: var(--accent-red); }
.metric-card.neutral::before   { background: var(--accent-yellow); }
.metric-card.blue::before      { background: var(--accent-blue); }
.metric-card.total::before     { background: linear-gradient(90deg, var(--accent-green), var(--accent-blue)); }
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(0,255,135,0.12);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--accent-green);
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
}
.metric-card.negative .metric-value  { color: var(--accent-red); }
.metric-card.neutral .metric-value   { color: var(--accent-yellow); }
.metric-card.blue .metric-value      { color: var(--accent-blue); }
.metric-card.total .metric-value     { color: var(--text-primary); }
.metric-label {
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.3rem;
    font-weight: 500;
}
.metric-icon {
    font-size: 1.6rem;
    margin-bottom: 0.4rem;
    display: block;
}

/* ── Section Headers ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: 0.04em;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header .accent-dot {
    width: 8px; height: 8px;
    background: var(--accent-green);
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 6px var(--accent-green);
}

/* ── Page Title ── */
.page-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}
.page-subtitle {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-top: 0.2rem;
}
.title-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}

/* ── Badge / Chips ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-positive { background: rgba(0,255,135,0.12); color: var(--positive); border: 1px solid rgba(0,255,135,0.25); }
.badge-negative { background: rgba(255,77,109,0.12); color: var(--negative); border: 1px solid rgba(255,77,109,0.25); }
.badge-neutral  { background: rgba(255,230,0,0.12);  color: var(--neutral);  border: 1px solid rgba(255,230,0,0.25); }

/* ── Review Table ── */
.review-table-row {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.5rem;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    transition: background 0.15s ease;
}
.review-table-row:hover { background: rgba(255,255,255,0.06); }

/* ── Progress Bars ── */
.confidence-bar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 4px;
    overflow: hidden;
    height: 6px;
    width: 100%;
    margin-top: 0.3rem;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent-green), var(--accent-blue));
    transition: width 0.5s ease;
}

/* ── Live Analyzer Box ── */
.analyzer-result-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 2rem;
    text-align: center;
    backdrop-filter: blur(16px);
    margin-top: 1rem;
}
.result-sentiment-label {
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    margin-bottom: 0.3rem;
}
.result-sentiment-label.positive { color: var(--positive); }
.result-sentiment-label.negative { color: var(--negative); }
.result-sentiment-label.neutral  { color: var(--neutral); }

/* ── Sidebar items ── */
.sidebar-nav-item {
    padding: 0.6rem 0.9rem;
    border-radius: var(--radius-sm);
    margin: 2px 0;
    cursor: pointer;
    font-size: 0.88rem;
    color: var(--text-secondary);
    transition: background 0.15s, color 0.15s;
    display: flex;
    align-items: center;
    gap: 0.55rem;
}
.sidebar-nav-item:hover { background: var(--bg-glass); color: var(--text-primary); }
.sidebar-nav-item.active { background: rgba(0,255,135,0.1); color: var(--accent-green); font-weight: 600; }

/* ── Ticker ── */
.ticker-strip {
    background: rgba(0,255,135,0.06);
    border: 1px solid rgba(0,255,135,0.12);
    border-radius: var(--radius-sm);
    padding: 0.5rem 1rem;
    font-size: 0.78rem;
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 1rem;
    white-space: nowrap;
    overflow: hidden;
}
.ticker-live {
    display: inline-block;
    width: 7px; height: 7px;
    background: var(--accent-green);
    border-radius: 50%;
    margin-right: 6px;
    animation: blink 1.2s infinite;
    vertical-align: middle;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.2; }
}

/* ── Keyword Pills ── */
.keyword-pill {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 2px;
}
.kw-positive { background: rgba(0,255,135,0.1);  color: var(--positive); }
.kw-negative { background: rgba(255,77,109,0.1); color: var(--negative); }

/* ── Login Card ── */
.login-container {
    max-width: 420px;
    margin: 4rem auto 0 auto;
    padding: 2.5rem 2.2rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    backdrop-filter: blur(20px);
    box-shadow: var(--shadow-card);
}
.login-logo {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    text-align: center;
    margin-bottom: 0.3rem;
    color: var(--text-primary);
}
.login-logo span { color: var(--accent-green); }
.login-sub {
    text-align: center;
    color: var(--text-secondary);
    font-size: 0.82rem;
    margin-bottom: 1.8rem;
}

/* ── Input Overrides ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] select {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent-green) !important;
    box-shadow: 0 0 0 2px rgba(0,255,135,0.15) !important;
}

/* ── Button Overrides ── */
[data-testid="stButton"] > button {
    background: var(--accent-green) !important;
    color: #000 !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 700 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.15s, transform 0.15s !important;
    padding: 0.5rem 1.2rem !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] > button.secondary-btn {
    background: var(--bg-glass) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    overflow: hidden;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

/* ── Metric override ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem !important;
}
[data-testid="stMetric"] label { color: var(--text-secondary) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-glass) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-secondary) !important;
}

/* ── Tab ── */
[data-testid="stTabs"] [role="tab"] {
    color: var(--text-secondary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent-green) !important;
    border-bottom-color: var(--accent-green) !important;
}

/* ── Sidebar logo ── */
.sb-logo {
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    padding: 0.5rem 1rem 1.2rem 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.8rem;
    color: var(--text-primary);
}
.sb-logo span { color: var(--accent-green); }
.sb-version {
    font-size: 0.65rem;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    display: block;
    margin-top: 2px;
}

/* ── Plotly dark override ── */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── Divider ── */
.thin-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1rem 0;
}

/* ── Confidence meter ── */
.meter-container {
    width: 100%;
    height: 10px;
    background: rgba(255,255,255,0.06);
    border-radius: 5px;
    overflow: hidden;
    margin-top: 0.5rem;
}
.meter-fill-positive { height:100%; border-radius:5px; background: linear-gradient(90deg,#00b85f,#00ff87); }
.meter-fill-negative { height:100%; border-radius:5px; background: linear-gradient(90deg,#c0173a,#ff4d6d); }
.meter-fill-neutral  { height:100%; border-radius:5px; background: linear-gradient(90deg,#c7b000,#ffe600); }
</style>
"""


# ════════════════════════════════════════════════════════════════════════════
#  PLOTLY THEME
# ════════════════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor = 'rgba(0,0,0,0)',
    plot_bgcolor  = 'rgba(0,0,0,0)',
    font          = dict(family="Space Grotesk, sans-serif", color="#8888aa", size=12),
    title_font    = dict(family="Space Grotesk, sans-serif", color="#e8e8f0", size=15),
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.07)", borderwidth=1),
    xaxis         = dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.07)"),
    yaxis         = dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.07)"),
    margin        = dict(l=30, r=20, t=50, b=30),
)

SENTIMENT_COLORS = {
    'positive': '#00ff87',
    'negative': '#ff4d6d',
    'neutral' : '#ffe600',
}

SENTIMENT_EMOJI = {
    'positive': '😊',
    'negative': '😠',
    'neutral' : '😐',
}


# ════════════════════════════════════════════════════════════════════════════
#  SESSION STATE DEFAULTS
# ════════════════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        'authenticated': False,
        'username'     : '',
        'page'         : 'Dashboard',
        'dark_mode'    : True,
        'ticker_index' : 0,
        'df'           : None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ════════════════════════════════════════════════════════════════════════════
#  FAKE LIVE TICKER MESSAGES
# ════════════════════════════════════════════════════════════════════════════

TICKER_EVENTS = [
    "🟢  User @alex_r submitted a Positive review  •  Confidence 94.2%",
    "🔴  User @meena_k submitted a Negative review  •  Confidence 88.7%",
    "🟡  User @jake_w submitted a Neutral review   •  Confidence 79.3%",
    "🟢  Batch of 42 reviews analyzed — 71% Positive",
    "🔴  Alert: Spike in Negative reviews for 'Category: Electronics'",
    "🟢  User @priya_s submitted a Positive review  •  Confidence 97.1%",
    "🟡  Model retrained — New F1-Score: 0.913",
    "🟢  CSV upload processed  •  120 reviews analyzed",
    "🔴  User @tom_h submitted a Negative review  •  Confidence 91.4%",
    "🟢  Dashboard viewed by 3 active sessions",
]


# ════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data():
    """Load the base dataset from CSV."""
    path = "dataset.csv"
    if not os.path.exists(path):
        # Generate a minimal placeholder so the app doesn't crash
        rows = []
        sentiments = ['positive', 'negative', 'neutral']
        for i in range(30):
            rows.append({
                'review'   : f"Sample review number {i+1}",
                'sentiment': sentiments[i % 3],
                'rating'   : random.randint(1, 5),
                'date'     : (datetime.date(2024, 1, 1) + datetime.timedelta(days=i)).isoformat(),
            })
        return pd.DataFrame(rows)

    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()

    if 'date' not in df.columns:
        df['date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D').strftime('%Y-%m-%d')

    df['date'] = pd.to_datetime(df['date'])

    if 'rating' not in df.columns:
        rating_map = {'positive': 5, 'negative': 1, 'neutral': 3}
        df['rating'] = df['sentiment'].map(rating_map)

    return df


def load_ml_model():
    """Load (or train) the ML model."""
    if "ml_model" not in st.session_state:
        st.session_state["ml_model"] = get_model()
    return st.session_state["ml_model"]


# ════════════════════════════════════════════════════════════════════════════
#  HELPER UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def compute_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    counts = df['sentiment'].value_counts()
    pos  = counts.get('positive', 0)
    neg  = counts.get('negative', 0)
    neu  = counts.get('neutral',  0)
    avg_rating = df['rating'].mean() if 'rating' in df.columns else 0.0
    return {
        'total'     : total,
        'positive'  : pos,
        'negative'  : neg,
        'neutral'   : neu,
        'pos_pct'   : (pos / total * 100) if total else 0,
        'neg_pct'   : (neg / total * 100) if total else 0,
        'neu_pct'   : (neu / total * 100) if total else 0,
        'avg_rating': avg_rating,
    }


def render_badge(sentiment: str) -> str:
    return f'<span class="badge badge-{sentiment}">{SENTIMENT_EMOJI.get(sentiment,"")} {sentiment.title()}</span>'


def render_confidence_bar(conf: float, sentiment: str) -> str:
    pct = int(conf * 100)
    return (
        f'<div class="meter-container">'
        f'<div class="meter-fill-{sentiment}" style="width:{pct}%"></div>'
        f'</div>'
    )


POSITIVE_KEYWORDS = {
    'amazing','excellent','fantastic','great','wonderful','love','best',
    'outstanding','superb','perfect','brilliant','exceptional','incredible',
    'awesome','good','happy','satisfied','recommend','quality','fast',
    'premium','durable','beautiful','stylish','efficient','reliable',
}
NEGATIVE_KEYWORDS = {
    'terrible','awful','horrible','worst','bad','poor','disappointing',
    'broken','useless','waste','cheap','defective','damaged','failed',
    'regret','avoid','scam','fraud','fake','misleading','overpriced',
    'slow','unreliable','flimsy','garbage','rubbish','appalling',
}

def extract_keywords(text: str):
    tokens = set(re.sub(r'[^a-zA-Z\s]', '', text.lower()).split())
    pos_found = sorted(tokens & POSITIVE_KEYWORDS)
    neg_found = sorted(tokens & NEGATIVE_KEYWORDS)
    return pos_found, neg_found


# ════════════════════════════════════════════════════════════════════════════
#  LOGIN PAGE
# ════════════════════════════════════════════════════════════════════════════

DEMO_CREDENTIALS = {
    'admin' : 'admin123',
    'demo'  : 'demo2024',
    'guest' : 'guest',
}

def page_login():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # Centered logo above form
    st.markdown("""
    <div style="text-align:center; padding-top:3rem; margin-bottom:2rem;">
        <div style="font-size:3rem; margin-bottom:0.5rem;">🧠</div>
        <div style="font-size:2rem; font-weight:800; letter-spacing:-0.04em; color:#e8e8f0; font-family:'Space Grotesk',sans-serif;">
            Senti<span style="color:#00ff87;">Scope</span>
        </div>
        <div style="font-size:0.82rem; color:#8888aa; margin-top:0.3rem; font-family:'Space Grotesk',sans-serif;">
            ML-Powered Sentiment Analytics Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 1.1, 1])
    with col_m:
        st.markdown("""
        <div style="background:rgba(20,20,30,0.85);border:1px solid rgba(255,255,255,0.07);
                    border-radius:20px;padding:2.5rem 2.2rem;backdrop-filter:blur(20px);">
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size:1.1rem;font-weight:700;color:#e8e8f0;margin-bottom:0.2rem;font-family:'Space Grotesk',sans-serif;">
            Sign in to your account
        </div>
        <div style="font-size:0.78rem;color:#8888aa;margin-bottom:1.5rem;font-family:'Space Grotesk',sans-serif;">
            Use  <b style="color:#00ff87;">admin / admin123</b>  or  <b style="color:#00ff87;">guest / guest</b>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Enter username", key="login_user")
        password = st.text_input("Password", type="password", placeholder="Enter password", key="login_pass")

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        login_btn = st.button("Sign In →", use_container_width=True)

        if login_btn:
            if username in DEMO_CREDENTIALS and DEMO_CREDENTIALS[username] == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid credentials. Try admin / admin123")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-top:2rem;font-size:0.73rem;color:#55556a;font-family:'Space Grotesk',sans-serif;">
        BCA Final Year Project  ·  Sentiment Analysis using Machine Learning  ·  2024
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

NAV_ITEMS = [
    ("📊", "Dashboard"),
    ("📋", "All Reviews"),
    ("✍️",  "Submit Review"),
    ("📈", "Analytics"),
    ("⚡", "Live Analyzer"),
    ("📖", "About / Dictionary"),
]

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sb-logo">
            Senti<span>Scope</span>
            <span class="sb-version">v2.4.1  ·  ML Edition</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<div style='font-size:0.78rem;color:#55556a;padding:0 1rem 0.8rem;'>👤 {st.session_state.username}</div>", unsafe_allow_html=True)

        for icon, label in NAV_ITEMS:
            active = "active" if st.session_state.page == label else ""
            if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
                st.session_state.page = label
                st.rerun()

        st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1rem 0;'>", unsafe_allow_html=True)

        # Live status indicator
        st.markdown("""
        <div style="padding:0 0.4rem;">
            <div style="font-size:0.7rem;color:#55556a;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">System Status</div>
            <div style="display:flex;align-items:center;gap:0.5rem;font-size:0.78rem;color:#8888aa;">
                <span style="width:7px;height:7px;background:#00ff87;border-radius:50%;display:inline-block;box-shadow:0 0 6px #00ff87;"></span>
                Model Loaded
            </div>
            <div style="display:flex;align-items:center;gap:0.5rem;font-size:0.78rem;color:#8888aa;margin-top:0.3rem;">
                <span style="width:7px;height:7px;background:#00ff87;border-radius:50%;display:inline-block;box-shadow:0 0 6px #00ff87;"></span>
                API Active
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='flex:1'></div>", unsafe_allow_html=True)
        st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1rem 0;'>", unsafe_allow_html=True)

        if st.button("🚪  Sign Out", use_container_width=True, key="signout"):
            st.session_state.authenticated = False
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
#  LIVE TICKER COMPONENT
# ════════════════════════════════════════════════════════════════════════════

def render_ticker():
    idx = random.randint(0, len(TICKER_EVENTS) - 1)
    msg = TICKER_EVENTS[idx]
    ts  = datetime.datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div class="ticker-strip">
        <span class="ticker-live"></span>
        <b style="color:#e8e8f0;">LIVE</b>
        &nbsp;&nbsp;{ts} &nbsp;·&nbsp; {msg}
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ════════════════════════════════════════════════════════════════════════════

def chart_sentiment_pie(df: pd.DataFrame):
    counts = df['sentiment'].value_counts()
    colors = [SENTIMENT_COLORS.get(s, '#aaa') for s in counts.index]
    fig = go.Figure(go.Pie(
        labels      = [s.title() for s in counts.index],
        values      = counts.values,
        hole        = 0.55,
        marker_colors = colors,
        textinfo    = 'percent',
        hovertemplate = "<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Sentiment Distribution", height=320,
                      annotations=[dict(text="Sentiment", x=0.5, y=0.5, font_size=13,
                                        font_color="#e8e8f0", showarrow=False)])
    return fig


def chart_sentiment_bar(df: pd.DataFrame):
    counts = df['sentiment'].value_counts().reset_index()
    counts.columns = ['Sentiment', 'Count']
    colors = [SENTIMENT_COLORS.get(s, '#aaa') for s in counts['Sentiment']]
    fig = go.Figure(go.Bar(
        x           = [s.title() for s in counts['Sentiment']],
        y           = counts['Count'],
        marker_color = colors,
        marker_line_width = 0,
        text        = counts['Count'],
        textposition = 'outside',
        textfont    = dict(color='#e8e8f0', size=13),
        hovertemplate = "<b>%{x}</b><br>Reviews: %{y}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Review Count by Sentiment", height=320,
                      bargap=0.45)
    return fig


def chart_sentiment_over_time(df: pd.DataFrame):
    if 'date' not in df.columns:
        return None
    df2 = df.copy()
    df2['date'] = pd.to_datetime(df2['date'])
    df2['week'] = df2['date'].dt.to_period('W').apply(lambda x: x.start_time)
    pivot = df2.groupby(['week', 'sentiment']).size().unstack(fill_value=0).reset_index()

    fig = go.Figure()
    for sent, color in SENTIMENT_COLORS.items():
        if sent in pivot.columns:
            fig.add_trace(go.Scatter(
                x    = pivot['week'],
                y    = pivot[sent],
                name = sent.title(),
                mode = 'lines+markers',
                line = dict(color=color, width=2.5),
                marker = dict(size=6, color=color),
                fill = 'tozeroy',
                fillcolor = color.replace(')', ',0.07)').replace('rgb', 'rgba') if 'rgb' in color else color + '12',
                hovertemplate = f"<b>{sent.title()}</b><br>Week: %{{x|%b %d}}<br>Count: %{{y}}<extra></extra>",
            ))

    fig.update_layout(**PLOTLY_LAYOUT, title="Sentiment Trend Over Time", height=320,
                      hovermode='x unified')
    return fig


def chart_rating_distribution(df: pd.DataFrame):
    if 'rating' not in df.columns:
        return None
    counts = df['rating'].value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x           = counts.index.astype(str),
        y           = counts.values,
        marker_color = ['#ff4d6d','#ff8c55','#ffe600','#9bde7e','#00ff87'],
        marker_line_width = 0,
        text        = counts.values,
        textposition = 'outside',
        textfont    = dict(color='#e8e8f0'),
        hovertemplate = "Rating ⭐ %{x}<br>Reviews: %{y}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Rating Distribution (1–5 Stars)", height=320,
                      xaxis_title="Star Rating", yaxis_title="Count", bargap=0.35)
    return fig


def chart_wordcloud(df: pd.DataFrame, sentiment_filter: str = 'all'):
    from wordcloud import WordCloud
    try:
        if sentiment_filter != 'all':
            text_data = ' '.join(df[df['sentiment'] == sentiment_filter]['review'].astype(str))
        else:
            text_data = ' '.join(df['review'].astype(str))

        cleaned = ' '.join(clean_text(text_data).split())
        if not cleaned.strip():
            return None

        color = {
            'positive': '#00ff87',
            'negative': '#ff4d6d',
            'neutral' : '#ffe600',
            'all'     : '#00cfff',
        }.get(sentiment_filter, '#00cfff')

        wc = WordCloud(
            width            = 700,
            height           = 300,
            background_color = '#0a0a0f',
            colormap         = 'Greens' if sentiment_filter == 'positive' else
                               'Reds'   if sentiment_filter == 'negative' else
                               'Wistia' if sentiment_filter == 'neutral'  else 'Blues',
            max_words        = 80,
            prefer_horizontal = 0.75,
            min_font_size    = 10,
            max_font_size    = 72,
            contour_width    = 0,
            font_step        = 1,
            random_state     = 42,
        ).generate(cleaned)

        fig, ax = plt.subplots(figsize=(9, 3.5), facecolor='#0a0a0f')
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        return fig
    except Exception:
        return None


def chart_confusion_matrix(cm, classes):
    fig = go.Figure(go.Heatmap(
        z           = cm,
        x           = [c.title() for c in classes],
        y           = [c.title() for c in classes],
        colorscale  = [[0, '#0a0a0f'], [0.5, '#003a22'], [1, '#00ff87']],
        showscale   = True,
        text        = cm,
        texttemplate = "%{text}",
        textfont    = dict(size=16, color='#e8e8f0'),
        hovertemplate = "Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Confusion Matrix",
                      xaxis_title="Predicted", yaxis_title="Actual", height=340)
    return fig


def chart_probability_bars(prob_dict: dict):
    labels = [k.title() for k in prob_dict]
    values = [v * 100 for v in prob_dict.values()]
    colors = [SENTIMENT_COLORS.get(k, '#aaa') for k in prob_dict]
    fig = go.Figure(go.Bar(
        x           = labels,
        y           = values,
        marker_color = colors,
        marker_line_width = 0,
        text        = [f"{v:.1f}%" for v in values],
        textposition = 'outside',
        textfont    = dict(color='#e8e8f0'),
        hovertemplate = "<b>%{x}</b><br>Probability: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Prediction Probabilities",
                      yaxis_title="Probability (%)", yaxis_range=[0, 105],
                      height=280, bargap=0.45)
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  DASHBOARD PAGE
# ════════════════════════════════════════════════════════════════════════════

def page_dashboard(df: pd.DataFrame):
    # Page title
    st.markdown("""
    <div class="title-row">
        <div>
            <div class="page-title">📊 Dashboard</div>
            <div class="page-subtitle">Real-time sentiment intelligence overview</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_ticker()

    stats = compute_stats(df)

    # ── KPI Cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "total",    "📦", f"{stats['total']:,}",            "Total Reviews"),
        (c2, "positive", "😊", f"{stats['pos_pct']:.1f}%",       "Positive"),
        (c3, "negative", "😠", f"{stats['neg_pct']:.1f}%",       "Negative"),
        (c4, "neutral",  "😐", f"{stats['neu_pct']:.1f}%",       "Neutral"),
        (c5, "blue",     "⭐", f"{stats['avg_rating']:.2f}/5",   "Avg Rating"),
    ]
    for col, cls, icon, val, lbl in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card {cls}">
                <span class="metric-icon">{icon}</span>
                <div class="metric-value">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Charts Row 1 ───────────────────────────────────────────────────────
    col_l, col_r = st.columns([1.6, 1])
    with col_l:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig_time = chart_sentiment_over_time(df)
        if fig_time:
            st.plotly_chart(fig_time, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.plotly_chart(chart_sentiment_pie(df), use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Charts Row 2 ───────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.plotly_chart(chart_sentiment_bar(df), use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig_rating = chart_rating_distribution(df)
        if fig_rating:
            st.plotly_chart(fig_rating, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Word Cloud ─────────────────────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span class="accent-dot"></span> Word Cloud — Most Frequent Terms</div>', unsafe_allow_html=True)
    wc_filter = st.selectbox("Filter by sentiment", ['all', 'positive', 'negative', 'neutral'], key="wc_filter")
    wc_fig = chart_wordcloud(df, wc_filter)
    if wc_fig:
        st.pyplot(wc_fig, use_container_width=True)
    else:
        st.info("Not enough text data to generate word cloud.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Recent Reviews Panel ───────────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span class="accent-dot"></span> Recent Reviews</div>', unsafe_allow_html=True)
    recent = df.tail(5).iloc[::-1]
    for _, row in recent.iterrows():
        sent = row.get('sentiment', 'neutral')
        emoji = SENTIMENT_EMOJI.get(sent, '😐')
        badge_html = render_badge(sent)
        st.markdown(f"""
        <div class="review-table-row">
            <div style="font-size:1.5rem;padding-top:0.1rem;">{emoji}</div>
            <div style="flex:1;">
                <div style="font-size:0.88rem;color:#e8e8f0;line-height:1.4;">{str(row['review'])[:140]}{'…' if len(str(row['review']))>140 else ''}</div>
                <div style="margin-top:0.35rem;">{badge_html}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  ALL REVIEWS PAGE
# ════════════════════════════════════════════════════════════════════════════

def page_all_reviews(df: pd.DataFrame, vectorizer, model):
    st.markdown("""
    <div class="title-row">
        <div>
            <div class="page-title">📋 All Reviews</div>
            <div class="page-subtitle">Browse, filter, search, and analyze every review</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Filters ────────────────────────────────────────────────────────────
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="accent-dot"></span> Filters & Search</div>', unsafe_allow_html=True)
        fc1, fc2, fc3, fc4 = st.columns([2, 1, 1, 1])
        with fc1:
            search_q = st.text_input("🔍 Search reviews", placeholder="Type keywords…", key="review_search")
        with fc2:
            sent_filter = st.selectbox("Sentiment", ['All', 'Positive', 'Negative', 'Neutral'], key="sent_filter")
        with fc3:
            sort_by = st.selectbox("Sort by", ['Date ↓', 'Date ↑', 'Rating ↓', 'Rating ↑'], key="sort_by")
        with fc4:
            per_page = st.selectbox("Show", [10, 25, 50, 100], key="per_page")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Apply filters ──────────────────────────────────────────────────────
    view = df.copy()
    if sent_filter != 'All':
        view = view[view['sentiment'] == sent_filter.lower()]
    if search_q.strip():
        view = view[view['review'].str.contains(search_q, case=False, na=False)]

    sort_map = {
        'Date ↓' : ('date', False),
        'Date ↑' : ('date', True),
        'Rating ↓': ('rating', False),
        'Rating ↑': ('rating', True),
    }
    sort_col, sort_asc = sort_map.get(sort_by, ('date', False))
    if sort_col in view.columns:
        view = view.sort_values(sort_col, ascending=sort_asc)

    st.markdown(f"<div style='font-size:0.78rem;color:#8888aa;margin-bottom:0.8rem;'>Showing <b style='color:#e8e8f0'>{min(per_page, len(view))}</b> of <b style='color:#e8e8f0'>{len(view)}</b> reviews</div>", unsafe_allow_html=True)

    # ── CSV Upload & Batch Predict ─────────────────────────────────────────
    with st.expander("📂 Upload CSV for Batch Analysis", expanded=False):
        uploaded = st.file_uploader("Choose a CSV file (needs 'review' column)", type=['csv'], key="batch_csv")
        if uploaded:
            try:
                udf = pd.read_csv(uploaded)
                udf.columns = udf.columns.str.lower()
                if 'review' not in udf.columns:
                    st.error("CSV must have a 'review' column.")
                else:
                    n_limit = st.slider("Number of reviews to analyze", 5, min(500, len(udf)), min(50, len(udf)))
                    udf = udf.head(n_limit)
                    if st.button("🚀 Run Batch Analysis", key="run_batch"):
                        with st.spinner("Analyzing reviews…"):
                            results = predict_batch(udf['review'].astype(str).tolist(), vectorizer, model)
                            udf['predicted_sentiment'] = [r['sentiment'] for r in results]
                            udf['confidence_pct']      = [f"{r['confidence']*100:.1f}%" for r in results]
                            udf['emoji']               = [SENTIMENT_EMOJI.get(r['sentiment'],'') for r in results]

                        st.success(f"✅ Analyzed {len(udf)} reviews!")
                        st.dataframe(udf[['review', 'predicted_sentiment', 'confidence_pct', 'emoji']],
                                     use_container_width=True, height=300)

                        # Download
                        csv_out = udf.to_csv(index=False).encode('utf-8')
                        st.download_button("⬇️ Download Results CSV", csv_out,
                                           file_name="sentiment_results.csv",
                                           mime="text/csv", key="dl_results")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    # ── Review Cards ───────────────────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    display_df = view.head(per_page)

    for idx, row in display_df.iterrows():
        sent  = str(row.get('sentiment', 'neutral')).lower()
        emoji = SENTIMENT_EMOJI.get(sent, '😐')
        badge_html = render_badge(sent)
        rating = row.get('rating', '—')
        date_val = row.get('date', '')
        date_str = str(date_val)[:10] if date_val else ''
        stars = '⭐' * int(rating) if str(rating).isdigit() or isinstance(rating, (int, float)) else ''

        st.markdown(f"""
        <div class="review-table-row">
            <div style="font-size:1.6rem;padding-top:0.1rem;">{emoji}</div>
            <div style="flex:1;">
                <div style="font-size:0.88rem;color:#e8e8f0;line-height:1.45;">{str(row['review'])}</div>
                <div style="margin-top:0.45rem;display:flex;gap:0.7rem;align-items:center;flex-wrap:wrap;">
                    {badge_html}
                    <span style="font-size:0.73rem;color:#8888aa;">{stars} {date_str}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Download all visible
    dl_csv = view.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Filtered Reviews", dl_csv,
                       file_name="filtered_reviews.csv", mime="text/csv")


# ════════════════════════════════════════════════════════════════════════════
#  SUBMIT REVIEW PAGE
# ════════════════════════════════════════════════════════════════════════════

def page_submit_review(df: pd.DataFrame, vectorizer, model):
    st.markdown("""
    <div class="title-row">
        <div>
            <div class="page-title">✍️ Submit Review</div>
            <div class="page-subtitle">Add a new product review and get instant ML prediction</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_preview = st.columns([1.2, 1])

    with col_form:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="accent-dot"></span> Review Form</div>', unsafe_allow_html=True)

        review_text = st.text_area("Your Review", height=140,
                                   placeholder="Write your product review here… be as detailed as you like.",
                                   key="submit_review_text")
        rating_val  = st.slider("Star Rating", 1, 5, 3, key="submit_rating")
        prod_name   = st.text_input("Product Name (optional)", placeholder="e.g. Wireless Headphones Pro", key="submit_prod")

        stars_display = '⭐' * rating_val
        st.markdown(f"<div style='font-size:1.1rem;margin-bottom:0.6rem;'>{stars_display}</div>", unsafe_allow_html=True)

        submit_btn = st.button("🚀 Analyze & Submit", use_container_width=True, key="do_submit")

        if submit_btn:
            if not review_text.strip():
                st.warning("Please write a review before submitting.")
            else:
                with st.spinner("Analyzing sentiment…"):
                    result = predict_single(review_text, vectorizer, model)
                    time.sleep(0.4)

                sent  = result['sentiment']
                conf  = result['confidence']
                probs = result['probabilities']
                emoji = SENTIMENT_EMOJI.get(sent, '😐')

                # Append to session DataFrame
                new_row = pd.DataFrame([{
                    'review'   : review_text,
                    'sentiment': sent,
                    'rating'   : rating_val,
                    'date'     : pd.Timestamp.now(),
                }])
                st.session_state.df = pd.concat([df, new_row], ignore_index=True)

                st.success(f"{emoji} Review submitted! Predicted: **{sent.title()}** ({conf*100:.1f}% confidence)")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_preview:
        if 'submit_review_text' in st.session_state and st.session_state.submit_review_text.strip():
            text_preview = st.session_state.submit_review_text
            preview_result = predict_single(text_preview, vectorizer, model)
            sent  = preview_result['sentiment']
            conf  = preview_result['confidence']
            probs = preview_result['probabilities']
            emoji = SENTIMENT_EMOJI.get(sent, '😐')

            st.markdown('<div class="analyzer-result-box">', unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:2.5rem;margin-bottom:0.3rem;'>{emoji}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-sentiment-label {sent}'>{sent.upper()}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.82rem;color:#8888aa;margin-bottom:1rem;'>Confidence: <b style='color:#e8e8f0'>{conf*100:.1f}%</b></div>", unsafe_allow_html=True)

            st.markdown(render_confidence_bar(conf, sent), unsafe_allow_html=True)
            st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

            st.plotly_chart(chart_probability_bars(probs), use_container_width=True,
                            config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center;padding:3rem 1.5rem;color:#55556a;">
                <div style="font-size:2.5rem;margin-bottom:0.8rem;">✍️</div>
                <div style="font-size:0.88rem;">Start typing your review to see a live prediction</div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  ANALYTICS PAGE
# ════════════════════════════════════════════════════════════════════════════

def page_analytics(metrics: dict):
    st.markdown("""
    <div class="title-row">
        <div>
            <div class="page-title">📈 Model Analytics</div>
            <div class="page-subtitle">Performance metrics, evaluation curves, and classification report</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not metrics:
        st.info("No metrics available. The model may still be loading.")
        return

    acc  = metrics.get('accuracy',  0)
    prec = metrics.get('precision', 0)
    rec  = metrics.get('recall',    0)
    f1   = metrics.get('f1',        0)
    cv_m = metrics.get('cv_mean',   0)
    cv_s = metrics.get('cv_std',    0)

    # ── Top Metric Cards ───────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    metric_cards = [
        (m1, "positive", "🎯", f"{acc*100:.2f}%",  "Accuracy"),
        (m2, "blue",     "🔬", f"{prec*100:.2f}%", "Precision"),
        (m3, "neutral",  "📡", f"{rec*100:.2f}%",  "Recall"),
        (m4, "positive", "🏆", f"{f1*100:.2f}%",   "F1-Score"),
        (m5, "total",    "🔁", f"{cv_m*100:.1f}%", "5-Fold CV"),
    ]
    for col, cls, icon, val, lbl in metric_cards:
        with col:
            st.markdown(f"""
            <div class="metric-card {cls}">
                <span class="metric-icon">{icon}</span>
                <div class="metric-value">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Confusion Matrix + Classification Report ───────────────────────────
    col_cm, col_cr = st.columns([1, 1.2])

    with col_cm:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        cm      = metrics.get('confusion_matrix')
        classes = metrics.get('classes', [])
        if cm is not None and len(classes):
            st.plotly_chart(chart_confusion_matrix(cm, classes),
                            use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Confusion matrix not available.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_cr:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="accent-dot"></span> Classification Report</div>', unsafe_allow_html=True)
        report = metrics.get('report', {})
        if report:
            rows = []
            for label in ['positive', 'negative', 'neutral']:
                if label in report:
                    r = report[label]
                    rows.append({
                        'Class'    : label.title(),
                        'Precision': f"{r.get('precision',0)*100:.2f}%",
                        'Recall'   : f"{r.get('recall',0)*100:.2f}%",
                        'F1-Score' : f"{r.get('f1-score',0)*100:.2f}%",
                        'Support'  : int(r.get('support', 0)),
                    })
            cr_df = pd.DataFrame(rows)
            st.dataframe(cr_df, use_container_width=True, hide_index=True, height=180)

            # Weighted avg
            wa = report.get('weighted avg', {})
            if wa:
                st.markdown(f"""
                <div style="margin-top:0.8rem;padding:0.9rem;background:rgba(0,255,135,0.05);
                            border:1px solid rgba(0,255,135,0.12);border-radius:10px;">
                    <div style="font-size:0.75rem;color:#8888aa;text-transform:uppercase;
                                letter-spacing:0.1em;margin-bottom:0.5rem;">Weighted Average</div>
                    <div style="display:flex;gap:2rem;">
                        <div><span style="color:#8888aa;font-size:0.78rem;">Precision</span><br>
                             <span style="color:#00ff87;font-size:1.1rem;font-weight:700;">{wa.get('precision',0)*100:.2f}%</span></div>
                        <div><span style="color:#8888aa;font-size:0.78rem;">Recall</span><br>
                             <span style="color:#00ff87;font-size:1.1rem;font-weight:700;">{wa.get('recall',0)*100:.2f}%</span></div>
                        <div><span style="color:#8888aa;font-size:0.78rem;">F1-Score</span><br>
                             <span style="color:#00ff87;font-size:1.1rem;font-weight:700;">{wa.get('f1-score',0)*100:.2f}%</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Classification report not available.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Per-Class F1 Bar ───────────────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span class="accent-dot"></span> Per-Class Performance Breakdown</div>', unsafe_allow_html=True)
    report = metrics.get('report', {})
    if report:
        metric_names = ['precision', 'recall', 'f1-score']
        fig_perf = go.Figure()
        for label in ['positive', 'negative', 'neutral']:
            if label in report:
                r = report[label]
                fig_perf.add_trace(go.Bar(
                    name         = label.title(),
                    x            = [m.title() for m in metric_names],
                    y            = [r.get(m, 0) * 100 for m in metric_names],
                    marker_color = SENTIMENT_COLORS.get(label, '#aaa'),
                    marker_line_width = 0,
                    hovertemplate = f"<b>{label.title()}</b><br>%{{x}}: %{{y:.2f}}%<extra></extra>",
                ))
        fig_perf.update_layout(**PLOTLY_LAYOUT, barmode='group', height=320,
                               yaxis_title="Score (%)", yaxis_range=[0, 105], bargap=0.2)
        st.plotly_chart(fig_perf, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

    # ── CV Scores visualization ────────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span class="accent-dot"></span> Model Info</div>', unsafe_allow_html=True)
    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        st.markdown(f"""
        <div style="background:rgba(0,255,135,0.05);border:1px solid rgba(0,255,135,0.1);border-radius:10px;padding:1rem;">
            <div style="font-size:0.72rem;color:#8888aa;text-transform:uppercase;letter-spacing:0.1em;">Algorithm</div>
            <div style="color:#00ff87;font-weight:700;margin-top:0.3rem;">Logistic Regression</div>
        </div>
        """, unsafe_allow_html=True)
    with col_i2:
        st.markdown(f"""
        <div style="background:rgba(0,207,255,0.05);border:1px solid rgba(0,207,255,0.1);border-radius:10px;padding:1rem;">
            <div style="font-size:0.72rem;color:#8888aa;text-transform:uppercase;letter-spacing:0.1em;">Vectorizer</div>
            <div style="color:#00cfff;font-weight:700;margin-top:0.3rem;">TF-IDF (1,2)-gram</div>
        </div>
        """, unsafe_allow_html=True)
    with col_i3:
        st.markdown(f"""
        <div style="background:rgba(255,230,0,0.05);border:1px solid rgba(255,230,0,0.1);border-radius:10px;padding:1rem;">
            <div style="font-size:0.72rem;color:#8888aa;text-transform:uppercase;letter-spacing:0.1em;">5-Fold CV Accuracy</div>
            <div style="color:#ffe600;font-weight:700;margin-top:0.3rem;">{cv_m*100:.2f}% ± {cv_s*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  LIVE ANALYZER PAGE
# ════════════════════════════════════════════════════════════════════════════

def page_live_analyzer(vectorizer, model):
    st.markdown("""
    <div class="title-row">
        <div>
            <div class="page-title">⚡ Live Analyzer</div>
            <div class="page-subtitle">Type any review and get instant ML-powered sentiment prediction</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_result = st.columns([1.1, 1])

    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="accent-dot"></span> Enter Review Text</div>', unsafe_allow_html=True)

        review_input = st.text_area(
            "Review",
            height=160,
            placeholder="e.g. 'The product quality is outstanding and delivery was super fast. Highly recommend!'",
            label_visibility="collapsed",
            key="live_review",
        )

        example_reviews = {
            "🟢 Positive Example": "Absolutely love this product! The quality is outstanding and it arrived ahead of schedule. Highly recommend to anyone.",
            "🔴 Negative Example": "Terrible experience. The product broke after two days and customer support was completely useless. Total waste of money.",
            "🟡 Neutral Example" : "The product is okay. It does what it's supposed to do but nothing really stands out about it.",
        }
        st.markdown("<div style='margin-top:0.6rem;font-size:0.78rem;color:#8888aa;'>Try an example:</div>", unsafe_allow_html=True)
        for label, sample in example_reviews.items():
            if st.button(label, key=f"ex_{label}"):
                st.session_state.live_review = sample
                st.rerun()

        analyze_btn = st.button("⚡ Analyze Sentiment", use_container_width=True, key="live_analyze_btn")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        live_text = st.session_state.get('live_review', '')
        if live_text and live_text.strip():
            result = predict_single(live_text, vectorizer, model)
            sent  = result['sentiment']
            conf  = result['confidence']
            probs = result['probabilities']
            emoji = SENTIMENT_EMOJI.get(sent, '😐')

            pos_kw, neg_kw = extract_keywords(live_text)

            st.markdown('<div class="analyzer-result-box">', unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:3rem;margin-bottom:0.3rem;'>{emoji}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-sentiment-label {sent}'>{sent.upper()}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.85rem;color:#8888aa;'>Confidence</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:1.7rem;font-weight:700;color:#e8e8f0;font-family:JetBrains Mono,monospace;'>{conf*100:.1f}%</div>", unsafe_allow_html=True)
            st.markdown(render_confidence_bar(conf, sent), unsafe_allow_html=True)

            st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

            if pos_kw:
                pills = ''.join([f'<span class="keyword-pill kw-positive">{k}</span>' for k in pos_kw[:8]])
                st.markdown(f"<div style='font-size:0.75rem;color:#8888aa;margin-bottom:0.3rem;'>Positive Keywords</div>{pills}", unsafe_allow_html=True)
            if neg_kw:
                pills = ''.join([f'<span class="keyword-pill kw-negative">{k}</span>' for k in neg_kw[:8]])
                st.markdown(f"<div style='font-size:0.75rem;color:#8888aa;margin-bottom:0.3rem;margin-top:0.5rem;'>Negative Keywords</div>{pills}", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            st.plotly_chart(chart_probability_bars(probs), use_container_width=True,
                            config={'displayModeBar': False})
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center;padding:4rem 1.5rem;color:#55556a;">
                <div style="font-size:3rem;margin-bottom:0.8rem;">⚡</div>
                <div style="font-size:0.9rem;">Enter a review on the left to see the prediction</div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  ABOUT / DICTIONARY PAGE
# ════════════════════════════════════════════════════════════════════════════

def page_about():
    st.markdown("""
    <div class="title-row">
        <div>
            <div class="page-title">📖 About & Dictionary</div>
            <div class="page-subtitle">Project documentation, methodology, and ML glossary</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🏗 Project Overview", "🤖 ML Pipeline", "📚 Glossary", "📁 Project Structure"])

    with tab1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="color:#e8e8f0;line-height:1.7;font-size:0.92rem;">
        <h3 style="color:#00ff87;font-size:1.2rem;margin-bottom:0.8rem;">Project Title</h3>
        <p><b>Sentiment Analysis for Product Ratings using Machine Learning</b></p>
        <p>This application is developed as a BCA Final Year Project. It demonstrates the practical application
        of Natural Language Processing (NLP) and Machine Learning to classify product reviews into three sentiment
        categories: <b style="color:#00ff87">Positive</b>, <b style="color:#ff4d6d">Negative</b>, and
        <b style="color:#ffe600">Neutral</b>.</p>

        <h3 style="color:#00ff87;font-size:1.2rem;margin:1rem 0 0.8rem;">Objectives</h3>
        <ul style="color:#8888aa;padding-left:1.2rem;">
            <li>Collect and preprocess real product review data</li>
            <li>Train a machine learning model for sentiment classification</li>
            <li>Build an interactive web dashboard for real-time analysis</li>
            <li>Evaluate model performance with industry-standard metrics</li>
            <li>Enable batch analysis of large review datasets</li>
        </ul>

        <h3 style="color:#00ff87;font-size:1.2rem;margin:1rem 0 0.8rem;">Technologies Used</h3>
        <ul style="color:#8888aa;padding-left:1.2rem;">
            <li><b style="color:#e8e8f0">Python 3.10+</b> — Core programming language</li>
            <li><b style="color:#e8e8f0">Streamlit</b> — Web application framework</li>
            <li><b style="color:#e8e8f0">Scikit-learn</b> — Machine learning library</li>
            <li><b style="color:#e8e8f0">NLTK</b> — Natural language processing toolkit</li>
            <li><b style="color:#e8e8f0">Plotly</b> — Interactive data visualization</li>
            <li><b style="color:#e8e8f0">Pandas / NumPy</b> — Data manipulation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="color:#e8e8f0;line-height:1.7;font-size:0.92rem;">
        <h3 style="color:#00ff87;font-size:1.2rem;margin-bottom:0.8rem;">Machine Learning Pipeline</h3>

        <div style="display:flex;flex-direction:column;gap:0.8rem;">

        <div style="background:rgba(0,255,135,0.05);border-left:3px solid #00ff87;padding:0.8rem 1rem;border-radius:0 8px 8px 0;">
            <b style="color:#00ff87;">Step 1 — Data Collection</b>
            <div style="color:#8888aa;margin-top:0.3rem;font-size:0.85rem;">CSV dataset with labelled reviews (positive/negative/neutral). Each review is manually or programmatically annotated.</div>
        </div>

        <div style="background:rgba(0,207,255,0.05);border-left:3px solid #00cfff;padding:0.8rem 1rem;border-radius:0 8px 8px 0;">
            <b style="color:#00cfff;">Step 2 — Text Preprocessing</b>
            <div style="color:#8888aa;margin-top:0.3rem;font-size:0.85rem;">
                Lowercase → URL/HTML removal → Punctuation removal → Tokenization →
                Stopword removal (preserving negations) → Lemmatization via WordNet
            </div>
        </div>

        <div style="background:rgba(255,230,0,0.05);border-left:3px solid #ffe600;padding:0.8rem 1rem;border-radius:0 8px 8px 0;">
            <b style="color:#ffe600;">Step 3 — Feature Extraction (TF-IDF)</b>
            <div style="color:#8888aa;margin-top:0.3rem;font-size:0.85rem;">
                TF-IDF Vectorizer with unigrams and bigrams (n-gram range 1–2), max 8,000 features,
                sublinear TF scaling, and document frequency filters.
            </div>
        </div>

        <div style="background:rgba(255,77,109,0.05);border-left:3px solid #ff4d6d;padding:0.8rem 1rem;border-radius:0 8px 8px 0;">
            <b style="color:#ff4d6d;">Step 4 — Model Training</b>
            <div style="color:#8888aa;margin-top:0.3rem;font-size:0.85rem;">
                Logistic Regression (multinomial, L2 regularization, C=1.5) trained on an 80/20
                train-test split with stratified sampling and balanced class weights.
            </div>
        </div>

        <div style="background:rgba(0,255,135,0.05);border-left:3px solid #00ff87;padding:0.8rem 1rem;border-radius:0 8px 8px 0;">
            <b style="color:#00ff87;">Step 5 — Evaluation</b>
            <div style="color:#8888aa;margin-top:0.3rem;font-size:0.85rem;">
                Accuracy, Precision, Recall, F1-Score, Confusion Matrix, 5-Fold Cross-Validation.
                Model serialized to model.pkl via pickle.
            </div>
        </div>

        </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        glossary = {
            "Sentiment Analysis"   : "The use of NLP and ML to identify and extract subjective information from text, such as opinions, emotions, and attitudes.",
            "TF-IDF"               : "Term Frequency–Inverse Document Frequency. A numerical statistic that reflects the importance of a word in a document relative to a corpus.",
            "Logistic Regression"  : "A statistical model used for binary or multi-class classification problems, outputting class probabilities via the softmax function.",
            "Precision"            : "The ratio of correctly predicted positive observations to all predicted positive observations. TP / (TP + FP).",
            "Recall"               : "The ratio of correctly predicted positive observations to all actual positives. TP / (TP + FN).",
            "F1-Score"             : "The harmonic mean of Precision and Recall. Balances both metrics into a single score.",
            "Confusion Matrix"     : "A table showing the counts of correct and incorrect predictions for each class.",
            "Lemmatization"        : "The process of reducing a word to its base or root form (e.g., 'running' → 'run').",
            "Stopwords"            : "Common words (the, is, at, which) that carry little meaning and are typically removed before NLP processing.",
            "Bigrams"              : "Sequences of two consecutive words treated as a single feature (e.g., 'not good', 'highly recommend').",
            "Cross-Validation"     : "A technique to evaluate model performance by training and testing on different data subsets to reduce overfitting.",
            "Confidence Score"     : "The probability assigned by the model to its predicted class, indicating how certain the prediction is.",
        }
        for term, definition in glossary.items():
            st.markdown(f"""
            <div style="margin-bottom:0.7rem;padding:0.8rem 1rem;background:rgba(255,255,255,0.03);
                        border-radius:8px;border-left:3px solid rgba(0,255,135,0.3);">
                <div style="color:#00ff87;font-weight:600;font-size:0.88rem;">{term}</div>
                <div style="color:#8888aa;font-size:0.82rem;margin-top:0.2rem;line-height:1.5;">{definition}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:#8888aa;
                    background:rgba(0,0,0,0.3);padding:1.5rem;border-radius:10px;
                    border:1px solid rgba(255,255,255,0.06);line-height:2;">
<span style="color:#00ff87;">sentiment_app/</span>
├── <span style="color:#e8e8f0;">app.py</span>           <span style="color:#55556a;"># Main Streamlit application</span>
├── <span style="color:#e8e8f0;">train_model.py</span>   <span style="color:#55556a;"># ML training pipeline</span>
├── <span style="color:#e8e8f0;">dataset.csv</span>      <span style="color:#55556a;"># Labelled review dataset</span>
├── <span style="color:#e8e8f0;">model.pkl</span>        <span style="color:#55556a;"># Saved model + vectorizer (auto-generated)</span>
├── <span style="color:#e8e8f0;">model_metrics.pkl</span> <span style="color:#55556a;"># Evaluation metrics (auto-generated)</span>
└── <span style="color:#e8e8f0;">requirements.txt</span> <span style="color:#55556a;"># Python dependencies</span>
        </div>

        <div style="margin-top:1.2rem;color:#e8e8f0;font-size:0.88rem;">
            <div style="color:#00cfff;font-weight:600;margin-bottom:0.6rem;">Quick Start Commands</div>
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:#8888aa;
                    background:rgba(0,0,0,0.3);padding:1.2rem;border-radius:10px;
                    border:1px solid rgba(255,255,255,0.06);line-height:2.2;">
<span style="color:#55556a;"># Install dependencies</span>
<span style="color:#00ff87;">pip install -r requirements.txt</span>

<span style="color:#55556a;"># (Optional) Train model manually</span>
<span style="color:#00ff87;">python train_model.py</span>

<span style="color:#55556a;"># Launch the dashboard</span>
<span style="color:#00ff87;">streamlit run app.py</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION ROUTER
# ════════════════════════════════════════════════════════════════════════════

def main():
    init_session()

    # Inject global CSS
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # ── Authentication gate ───────────────────────────────────────────────
    if not st.session_state.authenticated:
        page_login()
        return

    # ── Load resources ────────────────────────────────────────────────────
    with st.spinner("Loading SentiScope…"):
        vectorizer, model, metrics = load_ml_model()

    # Use session-updated DF if available, else load from disk
    if st.session_state.df is not None:
        df = st.session_state.df
    else:
        df = load_data()
        st.session_state.df = df

    # ── Sidebar ───────────────────────────────────────────────────────────
    render_sidebar()

    # ── Page routing ──────────────────────────────────────────────────────
    page = st.session_state.page

    if   page == "Dashboard"         : page_dashboard(df)
    elif page == "All Reviews"        : page_all_reviews(df, vectorizer, model)
    elif page == "Submit Review"      : page_submit_review(df, vectorizer, model)
    elif page == "Analytics"          : page_analytics(metrics)
    elif page == "Live Analyzer"      : page_live_analyzer(vectorizer, model)
    elif page == "About / Dictionary" : page_about()
    else:
        st.error(f"Unknown page: {page}")


if __name__ == '__main__':
    main()
