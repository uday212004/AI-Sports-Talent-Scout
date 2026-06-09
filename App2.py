import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import io
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Sports Talent Scout",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0rem; padding-bottom: 2rem; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 40%, #1a2332 70%, #0d2137 100%);
    border-bottom: 1px solid #21262d;
    padding: 2.2rem 2.5rem 1.8rem;
    margin: -1rem -1rem 0;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(0,168,255,0.07) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(255,107,53,0.05) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ffffff 0%, #58a6ff 60%, #00a8ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.3rem;
    letter-spacing: -0.5px;
}
.hero-sub {
    color: #8b949e;
    font-size: 0.9rem;
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.2px;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(90deg, #0ea5e9, #2563eb);
    color: #fff;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.7rem;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0 0.5rem;
}
.kpi-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease;
    cursor: default;
}
.kpi-card:hover { transform: translateY(-3px); border-color: #30363d; }
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}
.kpi-card.blue::before   { background: linear-gradient(90deg, #0ea5e9, #2563eb); }
.kpi-card.green::before  { background: linear-gradient(90deg, #10b981, #059669); }
.kpi-card.orange::before { background: linear-gradient(90deg, #f59e0b, #d97706); }
.kpi-card.purple::before { background: linear-gradient(90deg, #8b5cf6, #6d28d9); }
.kpi-icon { font-size: 1.6rem; margin-bottom: 0.5rem; display: block; }
.kpi-label {
    color: #8b949e;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.25rem;
}
.kpi-value {
    color: #e6edf3;
    font-size: 1.85rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.kpi-delta {
    font-size: 0.72rem;
    color: #10b981;
    font-weight: 500;
}

/* ── Section Headings ── */
.section-heading {
    color: #e6edf3;
    font-size: 1.1rem;
    font-weight: 700;
    margin: 1.8rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #21262d;
}

/* ── Prediction Result Cards ── */
.result-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 1.6rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    height: 100%;
}
.result-label {
    color: #8b949e;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.8rem;
}
.result-value {
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.result-badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.badge-elite  { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.badge-high   { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-medium { background: rgba(59,130,246,0.15); color: #3b82f6; border: 1px solid rgba(59,130,246,0.3); }
.badge-low    { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-value  { background: rgba(14,165,233,0.15); color: #0ea5e9; border: 1px solid rgba(14,165,233,0.3); }
.badge-rating { background: rgba(168,85,247,0.15); color: #a855f7; border: 1px solid rgba(168,85,247,0.3); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d !important;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* ── Sidebar Nav ── */
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.65rem 1rem;
    border-radius: 8px;
    color: #8b949e;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    margin-bottom: 4px;
    transition: all 0.15s ease;
    text-decoration: none;
}
.nav-item:hover { background: #161b22; color: #e6edf3; }
.nav-item.active { background: rgba(14,165,233,0.1); color: #0ea5e9; border-left: 3px solid #0ea5e9; }

/* ── Input Containers ── */
.input-group {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}
.input-group-title {
    color: #58a6ff;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Predict Button ── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.3px !important;
    padding: 0.65rem 2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(14,165,233,0.25) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(14,165,233,0.4) !important;
}

/* ── Leaderboard Table ── */
.lb-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
.lb-table th {
    background: #21262d;
    color: #8b949e;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 0.7rem 1rem;
    text-align: left;
    border-bottom: 1px solid #30363d;
}
.lb-table td {
    padding: 0.65rem 1rem;
    color: #e6edf3;
    border-bottom: 1px solid #21262d;
}
.lb-table tr:hover td { background: #161b22; }
.lb-table tr:first-child td { color: #fbbf24; font-weight: 700; }
.lb-table tr:nth-child(2) td { color: #9ca3af; }
.lb-table tr:nth-child(3) td { color: #92400e; }
.rank-medal { font-size: 1.1rem; }

/* ── Report Card ── */
.report-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 0.8rem;
}
.report-title {
    color: #58a6ff;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.8rem;
}
.insight-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 0.5rem 0;
    color: #c9d1d9;
    font-size: 0.875rem;
    border-bottom: 1px solid #21262d;
}
.insight-item:last-child { border-bottom: none; }
.insight-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #10b981;
    flex-shrink: 0;
    margin-top: 5px;
}

/* ── Tab Styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117 !important;
    border-bottom: 1px solid #21262d !important;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8b949e !important;
    border: none !important;
    padding: 0.7rem 1.2rem !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #e6edf3 !important;
    border-bottom: 2px solid #0ea5e9 !important;
    background: transparent !important;
}

/* ── Metric widgets ── */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 1.4rem !important; font-weight: 800 !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-weight: 600 !important;
}

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg, #0ea5e9, #2563eb) !important; }

/* ── Selectbox / Number input ── */
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}
.stSlider > div > div > div > div { background: #0ea5e9 !important; }

/* ── Divider ── */
hr { border-color: #21262d !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS  (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    clf             = joblib.load("models/future_potential_model.pkl")
    reg             = joblib.load("models/auction_value_model.pkl")
    role_encoder    = joblib.load("models/role_encoder.pkl")
    potential_encoder = joblib.load("models/potential_encoder.pkl")
    return clf, reg, role_encoder, potential_encoder


@st.cache_data
def load_data():
    return pd.read_csv("data/players_featured.csv")


clf, reg, role_encoder, potential_encoder = load_models()
df = load_data()


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
POTENTIAL_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Elite": 3}

POTENTIAL_COLOR = {
    "Elite":  "#fbbf24",
    "High":   "#10b981",
    "Medium": "#3b82f6",
    "Low":    "#ef4444",
}

def potential_badge_class(p):
    return {"Elite": "badge-elite", "High": "badge-high",
            "Medium": "badge-medium", "Low": "badge-low"}.get(p, "badge-medium")

def scout_rating_from_value(auction_val, potential):
    base = (auction_val / 20) * 10
    bonus = POTENTIAL_ORDER.get(potential, 1)
    return round(min(10, base + bonus * 0.3), 1)

def build_radar(batting_avg, strike_rate, wickets, economy, catches, fitness_score, matches):
    batting  = min(100, batting_avg * 1.8)
    bowling  = min(100, (wickets / max(matches, 1)) * 500)
    fielding = min(100, catches * 2)
    fitness  = fitness_score
    exp      = min(100, (matches / 300) * 100)
    aggress  = min(100, (strike_rate - 80) / 1.2)
    return [batting, bowling, fielding, fitness, exp, max(0, aggress)]

def generate_report_text(role, age, matches, runs, batting_avg, strike_rate,
                         fifties, hundreds, wickets, economy, catches,
                         fitness_score, future_potential, auction_value, scout_rating):
    lines = []
    if batting_avg > 45:
        lines.append("Elite batting average — consistent run-scorer at top level.")
    elif batting_avg > 35:
        lines.append("Above-average batting performance with solid consistency.")
    if strike_rate > 140:
        lines.append("Explosive strike rate — ideal for T20 and power-play phases.")
    if wickets > 100:
        lines.append("High wicket-taking ability — reliable match-winner with the ball.")
    elif wickets > 50:
        lines.append("Decent bowling threat contributing meaningful wickets.")
    if fitness_score > 88:
        lines.append("Outstanding fitness level — reduces injury risk significantly.")
    elif fitness_score > 75:
        lines.append("Good fitness profile suitable for modern professional cricket.")
    if catches > 40:
        lines.append("Exceptional fielder — reliable hands and strong ground fielding.")
    if hundreds > 5:
        lines.append("Match-winning hundreds demonstrate big-game temperament.")
    if future_potential == "Elite":
        lines.append("Elite projection — national and franchise franchise flagship material.")
    elif future_potential == "High":
        lines.append("High future potential — prime IPL auction candidate.")
    if not lines:
        lines.append("Balanced all-round profile — versatile squad player.")
    return lines


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.5rem; text-align:center;">
        <div style="font-size:2.2rem; margin-bottom:0.3rem;">🏏</div>
        <div style="color:#e6edf3; font-weight:800; font-size:1rem; letter-spacing:-0.3px;">AI Talent Scout</div>
        <div style="color:#8b949e; font-size:0.7rem; margin-top:2px;">Cricket Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🎯  Talent Prediction", "📊  Analytics", "🏆  Leaderboard", "ℹ️  About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="padding: 1rem; background:#161b22; border-radius:10px; border:1px solid #21262d;">
        <div style="color:#8b949e; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:0.8rem;">Model Status</div>
        <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;">
            <span style="color:#c9d1d9; font-size:0.78rem;">Classifier</span>
            <span style="color:#10b981; font-size:0.78rem; font-weight:600;">● Live</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;">
            <span style="color:#c9d1d9; font-size:0.78rem;">Regressor</span>
            <span style="color:#10b981; font-size:0.78rem; font-weight:600;">● Live</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="color:#c9d1d9; font-size:0.78rem;">Data Points</span>
            <span style="color:#58a6ff; font-size:0.78rem; font-weight:600;">3,000</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#8b949e; font-size:0.68rem; text-align:center; line-height:1.6;">
        Built with Random Forest · Scikit-Learn<br>
        Plotly · Streamlit
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO BANNER  (shown on all pages)
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🏆 AI-Powered Sports Analytics</div>
    <div class="hero-title">🏏 AI Sports Talent Scout</div>
    <p class="hero-sub">
        Cricket Talent Identification &amp; IPL Auction Value Prediction using Random Forest ML
        &nbsp;|&nbsp; 3,000 Players &nbsp;|&nbsp; 16 Features &nbsp;|&nbsp; 84% Classifier Accuracy &nbsp;|&nbsp; 96% R²
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
st.markdown("""
<div class="kpi-grid">
    <div class="kpi-card blue">
        <span class="kpi-icon">👥</span>
        <div class="kpi-label">Total Players</div>
        <div class="kpi-value">3,000</div>
        <div class="kpi-delta">↑ Training dataset</div>
    </div>
    <div class="kpi-card green">
        <span class="kpi-icon">🎯</span>
        <div class="kpi-label">Classifier Accuracy</div>
        <div class="kpi-value">84%</div>
        <div class="kpi-delta">↑ Random Forest</div>
    </div>
    <div class="kpi-card orange">
        <span class="kpi-icon">📈</span>
        <div class="kpi-label">Regression R²</div>
        <div class="kpi-value">96%</div>
        <div class="kpi-delta">↑ Auction predictor</div>
    </div>
    <div class="kpi-card purple">
        <span class="kpi-icon">⚙️</span>
        <div class="kpi-label">Total Features</div>
        <div class="kpi-value">16</div>
        <div class="kpi-delta">↑ Engineered features</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE: TALENT PREDICTION
# ═══════════════════════════════════════════════
if "Talent Prediction" in page:

    st.markdown('<div class="section-heading">🎯 Player Input &amp; Talent Prediction</div>', unsafe_allow_html=True)

    # ── INPUT FORM ──
    with st.expander("📋  Player Profile — Fill Inputs Below", expanded=True):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown('<div class="input-group-title">👤 Player Info</div>', unsafe_allow_html=True)
            age = st.number_input("Age", min_value=18, max_value=40, value=22,
                                  help="Player's current age (18–40)")
            role = st.selectbox("Role", ["All-Rounder", "Batter", "Bowler", "Wicketkeeper"],
                                help="Primary playing role")
            matches = st.number_input("Matches Played", min_value=0, max_value=500, value=100,
                                      help="Total career matches")

        with col_b:
            st.markdown('<div class="input-group-title">🏏 Batting Stats</div>', unsafe_allow_html=True)
            runs = st.number_input("Career Runs", min_value=0, value=3000)
            batting_average = st.number_input("Batting Average", min_value=0.0, value=40.0, step=0.5,
                                              format="%.2f")
            strike_rate = st.number_input("Strike Rate", min_value=0.0, value=130.0, step=0.5,
                                          format="%.2f")
            fifties  = st.number_input("Fifties", min_value=0, value=10)
            hundreds = st.number_input("Hundreds", min_value=0, value=2)

        with col_c:
            st.markdown('<div class="input-group-title">⚾ Bowling &amp; Fielding</div>', unsafe_allow_html=True)
            wickets = st.number_input("Wickets", min_value=0, value=50)
            economy = st.number_input("Economy Rate", min_value=0.1, value=6.0, step=0.1, format="%.2f")
            catches = st.number_input("Catches", min_value=0, value=20)
            fitness_score = st.slider("Fitness Score", min_value=50, max_value=100, value=80,
                                      help="Physical fitness score (50–100)")

    predict_clicked = st.button("🚀  Predict Talent & Generate Report", use_container_width=True)

    # ── PREDICTION LOGIC ──
    if predict_clicked:
        role_encoded      = role_encoder.transform([role])[0]
        batting_impact    = batting_average * strike_rate
        bowling_impact    = wickets / economy
        experience_score  = matches * age
        consistency_score = fifties + (2 * hundreds)

        input_data = pd.DataFrame([{
            "age": age, "role": role_encoded, "matches": matches,
            "runs": runs, "batting_average": batting_average,
            "strike_rate": strike_rate, "fifties": fifties, "hundreds": hundreds,
            "wickets": wickets, "economy": economy, "catches": catches,
            "fitness_score": fitness_score,
            "batting_impact": batting_impact, "bowling_impact": bowling_impact,
            "experience_score": experience_score, "consistency_score": consistency_score,
        }])

        future_pred     = clf.predict(input_data)[0]
        future_potential = potential_encoder.inverse_transform([future_pred])[0]
        auction_value   = reg.predict(input_data)[0]
        scout_rating    = scout_rating_from_value(auction_value, future_potential)
        proba           = clf.predict_proba(input_data)[0]
        proba_labels    = potential_encoder.inverse_transform(list(range(len(proba))))
        badge_cls       = potential_badge_class(future_potential)

        # ── RESULT CARDS ──
        st.markdown('<div class="section-heading">📊 Prediction Results</div>', unsafe_allow_html=True)

        rc1, rc2, rc3 = st.columns(3)

        with rc1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">🌟 Future Potential</div>
                <div class="result-value" style="color:{POTENTIAL_COLOR.get(future_potential,'#58a6ff')};">
                    {future_potential}
                </div>
                <span class="result-badge {badge_cls}">{future_potential} Grade</span>
            </div>
            """, unsafe_allow_html=True)

        with rc2:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">💰 IPL Auction Value</div>
                <div class="result-value" style="color:#0ea5e9;">₹ {auction_value:.2f} Cr</div>
                <span class="result-badge badge-value">Estimated Bid</span>
            </div>
            """, unsafe_allow_html=True)

        with rc3:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">🎖️ Scout Rating</div>
                <div class="result-value" style="color:#a855f7;">{scout_rating}/10</div>
                <span class="result-badge badge-rating">Overall Score</span>
            </div>
            """, unsafe_allow_html=True)

        # Progress bar for scout rating
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.progress(scout_rating / 10)

        # ── PROBABILITY + RADAR (side by side) ──
        st.markdown('<div class="section-heading">📉 Probability Analysis &amp; Skill Radar</div>',
                    unsafe_allow_html=True)

        viz1, viz2 = st.columns(2)

        with viz1:
            sorted_pairs = sorted(zip(proba_labels, proba), key=lambda x: POTENTIAL_ORDER.get(x[0], 0))
            labels_sorted = [p[0] for p in sorted_pairs]
            probs_sorted  = [p[1] * 100 for p in sorted_pairs]
            bar_colors    = [POTENTIAL_COLOR.get(l, "#58a6ff") for l in labels_sorted]

            fig_bar = go.Figure(go.Bar(
                x=probs_sorted, y=labels_sorted, orientation="h",
                marker=dict(
                    color=bar_colors,
                    opacity=0.85,
                    line=dict(color="rgba(255,255,255,0.1)", width=1),
                ),
                text=[f"{v:.1f}%" for v in probs_sorted],
                textposition="outside",
                textfont=dict(color="#e6edf3", size=12),
            ))
            fig_bar.update_layout(
                title=dict(text="Potential Class Probabilities", font=dict(color="#e6edf3", size=14)),
                paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                           range=[0, 115], color="#8b949e"),
                yaxis=dict(color="#c9d1d9", tickfont=dict(size=13, color="#c9d1d9")),
                margin=dict(l=10, r=30, t=45, b=10),
                height=260,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with viz2:
            radar_vals = build_radar(batting_average, strike_rate, wickets,
                                     economy, catches, fitness_score, matches)
            radar_cats = ["Batting", "Bowling", "Fielding", "Fitness", "Experience", "Aggression"]
            radar_vals_closed = radar_vals + [radar_vals[0]]
            radar_cats_closed = radar_cats + [radar_cats[0]]

            fig_radar = go.Figure(go.Scatterpolar(
                r=radar_vals_closed,
                theta=radar_cats_closed,
                fill="toself",
                fillcolor="rgba(14,165,233,0.12)",
                line=dict(color="#0ea5e9", width=2),
                marker=dict(color="#0ea5e9", size=6),
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="#1a2230",
                    radialaxis=dict(visible=True, range=[0, 100],
                                   color="#30363d", gridcolor="#21262d",
                                   tickfont=dict(color="#8b949e", size=9)),
                    angularaxis=dict(color="#8b949e", gridcolor="#21262d",
                                     tickfont=dict(color="#c9d1d9", size=12)),
                ),
                paper_bgcolor="#161b22",
                title=dict(text="Player Skill Radar", font=dict(color="#e6edf3", size=14)),
                margin=dict(l=30, r=30, t=50, b=30),
                height=280,
                showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── SCOUTING REPORT ──
        st.markdown('<div class="section-heading">📝 AI Scouting Report</div>', unsafe_allow_html=True)

        insights = generate_report_text(
            role, age, matches, runs, batting_average, strike_rate,
            fifties, hundreds, wickets, economy, catches,
            fitness_score, future_potential, auction_value, scout_rating
        )

        r1, r2 = st.columns([1, 1])
        with r1:
            st.markdown(f"""
            <div class="report-card">
                <div class="report-title">🔎 Key Insights</div>
                {"".join(f'<div class="insight-item"><div class="insight-dot"></div><span>{i}</span></div>' for i in insights)}
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="report-card">
                <div class="report-title">📋 Player Snapshot</div>
                <div class="insight-item"><div class="insight-dot" style="background:#58a6ff"></div><span>Age: {age} yrs &nbsp;|&nbsp; Role: {role}</span></div>
                <div class="insight-item"><div class="insight-dot" style="background:#58a6ff"></div><span>Matches: {matches} &nbsp;|&nbsp; Runs: {runs:,}</span></div>
                <div class="insight-item"><div class="insight-dot" style="background:#58a6ff"></div><span>Avg: {batting_average:.1f} &nbsp;|&nbsp; SR: {strike_rate:.1f}</span></div>
                <div class="insight-item"><div class="insight-dot" style="background:#58a6ff"></div><span>Wickets: {wickets} &nbsp;|&nbsp; Economy: {economy:.1f}</span></div>
                <div class="insight-item"><div class="insight-dot" style="background:#58a6ff"></div><span>50s: {fifties} &nbsp;|&nbsp; 100s: {hundreds} &nbsp;|&nbsp; Catches: {catches}</span></div>
                <div class="insight-item"><div class="insight-dot" style="background:#58a6ff"></div><span>Fitness Score: {fitness_score}/100</span></div>
            </div>
            """, unsafe_allow_html=True)

        # ── DOWNLOAD REPORT ──
        st.markdown('<div class="section-heading">⬇️ Download Scouting Report</div>',
                    unsafe_allow_html=True)

        report_text = f"""
╔══════════════════════════════════════════════════════════════════╗
║            AI SPORTS TALENT SCOUT — SCOUTING REPORT            ║
╚══════════════════════════════════════════════════════════════════╝
Generated : {datetime.now().strftime("%d %B %Y, %I:%M %p")}

── PLAYER PROFILE ──────────────────────────────────────────────────
Age         : {age} years
Role        : {role}
Matches     : {matches}
Career Runs : {runs:,}

── BATTING STATS ───────────────────────────────────────────────────
Batting Average : {batting_average:.2f}
Strike Rate     : {strike_rate:.2f}
Fifties         : {fifties}
Hundreds        : {hundreds}

── BOWLING STATS ───────────────────────────────────────────────────
Wickets   : {wickets}
Economy   : {economy:.2f}

── FIELDING & FITNESS ──────────────────────────────────────────────
Catches       : {catches}
Fitness Score : {fitness_score}/100

── PREDICTION RESULTS ──────────────────────────────────────────────
Future Potential : {future_potential}
Auction Value    : ₹ {auction_value:.2f} Crores
Scout Rating     : {scout_rating} / 10

── KEY INSIGHTS ────────────────────────────────────────────────────
{chr(10).join("• " + i for i in insights)}

── ENGINEERED METRICS ──────────────────────────────────────────────
Batting Impact    : {batting_average * strike_rate:.1f}
Bowling Impact    : {wickets / economy:.2f}
Experience Score  : {matches * age}
Consistency Score : {fifties + 2 * hundreds}

────────────────────────────────────────────────────────────────────
AI Sports Talent Scout | Powered by Random Forest ML
────────────────────────────────────────────────────────────────────
"""
        st.download_button(
            label="📄  Download Full Report (.txt)",
            data=report_text,
            file_name=f"scouting_report_{role}_{age}yrs.txt",
            mime="text/plain",
            use_container_width=True,
        )

    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem 2rem; background:#161b22;
                    border:1px dashed #30363d; border-radius:14px; margin-top:1rem;">
            <div style="font-size:3rem; margin-bottom:1rem;">🏏</div>
            <div style="color:#e6edf3; font-size:1.05rem; font-weight:600; margin-bottom:0.5rem;">
                Ready to Scout Talent?
            </div>
            <div style="color:#8b949e; font-size:0.85rem;">
                Fill in the player stats above and click <strong style="color:#0ea5e9;">Predict Talent</strong>
                to get AI predictions, radar chart, probability analysis and scouting report.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE: ANALYTICS
# ═══════════════════════════════════════════════
elif "Analytics" in page:

    st.markdown('<div class="section-heading">📊 Dataset Analytics Dashboard</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📌 Distributions", "📉 Correlations", "💹 Value Analysis"])

    with tab1:
        c1, c2 = st.columns(2)

        with c1:
            role_counts = df["role"].value_counts().reset_index()
            role_counts.columns = ["Role", "Count"]
            fig_role = px.pie(
                role_counts, values="Count", names="Role",
                hole=0.55, color_discrete_sequence=["#0ea5e9", "#10b981", "#f59e0b", "#8b5cf6"],
                title="Role Distribution"
            )
            fig_role.update_traces(textposition="outside", textinfo="label+percent",
                                   textfont=dict(color="#e6edf3"))
            fig_role.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                                   title_font=dict(color="#e6edf3", size=14),
                                   font_color="#c9d1d9",
                                   legend=dict(font=dict(color="#c9d1d9")),
                                   margin=dict(t=50, b=20, l=20, r=20))
            st.plotly_chart(fig_role, use_container_width=True)

        with c2:
            pot_order  = ["Low", "Medium", "High", "Elite"]
            pot_counts = df["future_potential"].value_counts().reindex(pot_order).reset_index()
            pot_counts.columns = ["Potential", "Count"]
            bar_colors = [POTENTIAL_COLOR[p] for p in pot_counts["Potential"]]

            fig_pot = px.bar(
                pot_counts, x="Potential", y="Count", color="Potential",
                color_discrete_map=POTENTIAL_COLOR, title="Potential Distribution",
                text="Count",
            )
            fig_pot.update_traces(textposition="outside", textfont=dict(color="#e6edf3"),
                                  marker_line_width=0)
            fig_pot.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                                  title_font=dict(color="#e6edf3", size=14),
                                  font_color="#c9d1d9", showlegend=False,
                                  xaxis=dict(color="#8b949e", gridcolor="#21262d"),
                                  yaxis=dict(color="#8b949e", gridcolor="#21262d"),
                                  margin=dict(t=50, b=20, l=20, r=20))
            st.plotly_chart(fig_pot, use_container_width=True)

        # Fitness + Age distributions
        c3, c4 = st.columns(2)

        with c3:
            fig_fit = px.histogram(
                df, x="fitness_score", nbins=30,
                color_discrete_sequence=["#0ea5e9"], title="Fitness Score Distribution",
                opacity=0.85,
            )
            fig_fit.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                                  title_font=dict(color="#e6edf3", size=14),
                                  font_color="#c9d1d9",
                                  xaxis=dict(color="#8b949e", gridcolor="#21262d"),
                                  yaxis=dict(color="#8b949e", gridcolor="#21262d"),
                                  bargap=0.05, margin=dict(t=50, b=20))
            st.plotly_chart(fig_fit, use_container_width=True)

        with c4:
            fig_age = px.histogram(
                df, x="age", nbins=22, color_discrete_sequence=["#10b981"],
                title="Age Distribution", opacity=0.85,
            )
            fig_age.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                                  title_font=dict(color="#e6edf3", size=14),
                                  font_color="#c9d1d9",
                                  xaxis=dict(color="#8b949e", gridcolor="#21262d"),
                                  yaxis=dict(color="#8b949e", gridcolor="#21262d"),
                                  bargap=0.05, margin=dict(t=50, b=20))
            st.plotly_chart(fig_age, use_container_width=True)

    with tab2:
        num_cols = ["batting_average", "strike_rate", "wickets", "economy",
                    "catches", "fitness_score", "auction_value_cr", "talent_score"]
        corr = df[num_cols].corr().round(2)

        fig_heat = px.imshow(
            corr,
            color_continuous_scale=[[0, "#1e3a5f"], [0.5, "#21262d"], [1, "#0ea5e9"]],
            title="Feature Correlation Heatmap",
            text_auto=True,
            aspect="auto",
        )
        fig_heat.update_traces(textfont=dict(size=10, color="#e6edf3"))
        fig_heat.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            title_font=dict(color="#e6edf3", size=14),
            font_color="#c9d1d9",
            coloraxis_colorbar=dict(tickfont=dict(color="#c9d1d9")),
            margin=dict(t=60, b=30),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Batting avg vs Strike rate scatter
        fig_scat = px.scatter(
            df, x="batting_average", y="strike_rate", color="future_potential",
            color_discrete_map=POTENTIAL_COLOR, opacity=0.65,
            title="Batting Average vs Strike Rate (coloured by Potential)",
            hover_data=["role", "auction_value_cr"],
            size_max=8,
        )
        fig_scat.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            title_font=dict(color="#e6edf3", size=14),
            font_color="#c9d1d9",
            xaxis=dict(color="#8b949e", gridcolor="#21262d"),
            yaxis=dict(color="#8b949e", gridcolor="#21262d"),
            legend=dict(font=dict(color="#c9d1d9")),
            margin=dict(t=60, b=30),
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    with tab3:
        c5, c6 = st.columns(2)

        with c5:
            fig_hist = px.histogram(
                df, x="auction_value_cr", nbins=40, color="future_potential",
                color_discrete_map=POTENTIAL_COLOR, barmode="overlay", opacity=0.7,
                title="Auction Value Distribution by Potential",
            )
            fig_hist.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                                   title_font=dict(color="#e6edf3", size=14),
                                   font_color="#c9d1d9",
                                   xaxis=dict(color="#8b949e", gridcolor="#21262d"),
                                   yaxis=dict(color="#8b949e", gridcolor="#21262d"),
                                   legend=dict(font=dict(color="#c9d1d9")),
                                   bargap=0.02, margin=dict(t=60, b=30))
            st.plotly_chart(fig_hist, use_container_width=True)

        with c6:
            fig_box = px.box(
                df, x="future_potential", y="auction_value_cr", color="future_potential",
                color_discrete_map=POTENTIAL_COLOR,
                category_orders={"future_potential": ["Low", "Medium", "High", "Elite"]},
                title="Auction Value by Potential Class",
            )
            fig_box.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                                  title_font=dict(color="#e6edf3", size=14),
                                  font_color="#c9d1d9", showlegend=False,
                                  xaxis=dict(color="#8b949e", gridcolor="#21262d"),
                                  yaxis=dict(color="#8b949e", gridcolor="#21262d"),
                                  margin=dict(t=60, b=30))
            st.plotly_chart(fig_box, use_container_width=True)

        # Runs vs Auction Value
        fig_rv = px.scatter(
            df, x="runs", y="auction_value_cr", color="role",
            color_discrete_sequence=["#0ea5e9", "#10b981", "#f59e0b", "#8b5cf6"],
            opacity=0.6, title="Career Runs vs Auction Value",
            trendline="ols",
            hover_data=["age", "future_potential"],
        )
        fig_rv.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                             title_font=dict(color="#e6edf3", size=14),
                             font_color="#c9d1d9",
                             xaxis=dict(color="#8b949e", gridcolor="#21262d"),
                             yaxis=dict(color="#8b949e", gridcolor="#21262d"),
                             legend=dict(font=dict(color="#c9d1d9")),
                             margin=dict(t=60, b=30))
        st.plotly_chart(fig_rv, use_container_width=True)

        # Role-wise average auction value
        role_avg = df.groupby("role")["auction_value_cr"].mean().sort_values(ascending=False).reset_index()
        role_avg.columns = ["Role", "Avg Auction Value (Cr)"]

        fig_role_val = px.bar(
            role_avg, x="Role", y="Avg Auction Value (Cr)",
            color="Role",
            color_discrete_sequence=["#0ea5e9", "#10b981", "#f59e0b", "#8b5cf6"],
            title="Average Auction Value by Role",
            text_auto=".2f",
        )
        fig_role_val.update_traces(textfont=dict(color="#e6edf3"), textposition="outside",
                                   marker_line_width=0)
        fig_role_val.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                                   title_font=dict(color="#e6edf3", size=14),
                                   font_color="#c9d1d9", showlegend=False,
                                   xaxis=dict(color="#8b949e", gridcolor="#21262d"),
                                   yaxis=dict(color="#8b949e", gridcolor="#21262d"),
                                   margin=dict(t=60, b=30))
        st.plotly_chart(fig_role_val, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE: LEADERBOARD
# ═══════════════════════════════════════════════
elif "Leaderboard" in page:

    st.markdown('<div class="section-heading">🏆 Top Talent Leaderboard</div>', unsafe_allow_html=True)

    lf1, lf2 = st.columns([2, 1])
    with lf1:
        selected_role = st.selectbox("Filter by Role", ["All"] + sorted(df["role"].unique().tolist()))
    with lf2:
        top_n = st.selectbox("Show Top N", [10, 25, 50, 100], index=0)

    lb_df = df.copy()
    if selected_role != "All":
        lb_df = lb_df[lb_df["role"] == selected_role]

    lb_df = lb_df.sort_values("talent_score", ascending=False).head(top_n).reset_index(drop=True)
    lb_df["Rank"] = lb_df.index + 1
    lb_df["Scout Rating"] = lb_df.apply(
        lambda r: scout_rating_from_value(r["auction_value_cr"], r["future_potential"]), axis=1
    )

    medals = {1: "🥇", 2: "🥈", 3: "🥉"}

    rows_html = ""
    for _, row in lb_df.iterrows():
        rank   = int(row["Rank"])
        medal  = medals.get(rank, f"#{rank}")
        badge  = potential_badge_class(row["future_potential"])
        pot_color = POTENTIAL_COLOR.get(row["future_potential"], "#58a6ff")
        rows_html += f"""
        <tr>
            <td><span class="rank-medal">{medal}</span></td>
            <td>P-{int(row['player_id']):04d}</td>
            <td>{row['role']}</td>
            <td style="color:{pot_color}; font-weight:700;">{row['future_potential']}</td>
            <td style="color:#fbbf24; font-weight:700;">{row['talent_score']:.1f}</td>
            <td style="color:#0ea5e9; font-weight:600;">₹ {row['auction_value_cr']:.2f} Cr</td>
            <td style="color:#a855f7;">{row['Scout Rating']}</td>
        </tr>"""

    st.markdown(f"""
    <div style="background:#161b22; border:1px solid #21262d; border-radius:12px; overflow:hidden; margin-top:0.5rem;">
        <table class="lb-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Player ID</th>
                    <th>Role</th>
                    <th>Potential</th>
                    <th>Talent Score</th>
                    <th>Auction Value</th>
                    <th>Scout Rating</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Download leaderboard
    csv_buf = io.StringIO()
    lb_df[["Rank", "player_id", "role", "future_potential", "talent_score",
           "auction_value_cr", "Scout Rating"]].to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️  Export Leaderboard as CSV",
        data=csv_buf.getvalue(),
        file_name="talent_leaderboard.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Summary metrics row
    st.markdown('<div class="section-heading">📈 Leaderboard Summary</div>', unsafe_allow_html=True)
    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("Avg Auction Value", f"₹ {lb_df['auction_value_cr'].mean():.2f} Cr")
    sm2.metric("Avg Talent Score",  f"{lb_df['talent_score'].mean():.1f}")
    sm3.metric("Elite Players",     int((lb_df["future_potential"] == "Elite").sum()))
    sm4.metric("High Potential",    int((lb_df["future_potential"] == "High").sum()))


# ═══════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════
elif "About" in page:

    st.markdown('<div class="section-heading">ℹ️ About this Project</div>', unsafe_allow_html=True)

    a1, a2 = st.columns([3, 2])

    with a1:
        st.markdown("""
        <div class="report-card">
            <div class="report-title">🚀 Project Overview</div>
            <div class="insight-item"><div class="insight-dot"></div>
                <span>AI-powered cricket talent identification &amp; IPL auction value prediction system.</span>
            </div>
            <div class="insight-item"><div class="insight-dot"></div>
                <span>Trained on a synthetic dataset of 3,000 cricket players with 16 features.</span>
            </div>
            <div class="insight-item"><div class="insight-dot"></div>
                <span>Random Forest Classifier predicts Future Potential with 84% accuracy.</span>
            </div>
            <div class="insight-item"><div class="insight-dot"></div>
                <span>Random Forest Regressor estimates Auction Value with 96% R² score.</span>
            </div>
            <div class="insight-item"><div class="insight-dot"></div>
                <span>Feature engineering produces batting impact, bowling impact, experience &amp; consistency scores.</span>
            </div>
        </div>

        <div class="report-card" style="margin-top:1rem;">
            <div class="report-title">⚙️ Tech Stack</div>
            <div class="insight-item"><div class="insight-dot" style="background:#f59e0b"></div>
                <span>Python 3.10+ · Streamlit · Pandas · NumPy</span>
            </div>
            <div class="insight-item"><div class="insight-dot" style="background:#f59e0b"></div>
                <span>Scikit-Learn (Random Forest Classifier + Regressor)</span>
            </div>
            <div class="insight-item"><div class="insight-dot" style="background:#f59e0b"></div>
                <span>Plotly (Interactive Charts, Radar, Heatmap)</span>
            </div>
            <div class="insight-item"><div class="insight-dot" style="background:#f59e0b"></div>
                <span>Joblib (Model Serialisation)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="report-card">
            <div class="report-title">🎯 Features</div>
            <div class="insight-item"><div class="insight-dot" style="background:#8b5cf6"></div><span>age</span></div>
            <div class="insight-item"><div class="insight-dot" style="background:#8b5cf6"></div><span>role (encoded)</span></div>
            <div class="insight-item"><div class="insight-dot" style="background:#8b5cf6"></div><span>matches</span></div>
            <div class="insight-item"><div class="insight-dot" style="background:#8b5cf6"></div><span>runs</span></div>
            <div class="insight-item"><div class="insight-dot" style="background:#8b5cf6"></div><span>batting_average</span></div>
            <div class="insight-item"><div class="insight-dot" style="background:#8b5cf6"></div><span>strike_rate</span></div>
            <div class="insight-item"><div class="insight-dot" style="background:#8b5cf6"></div><span>fifties / hundreds</span></div>
            <div class="insight-item"><div class="insight-dot" style="background:#8b5cf6"></div><span>wickets / economy</span></div>
            <div class="insight-item"><div class="insight-dot" style="background:#8b5cf6"></div><span>catches / fitness_score</span></div>
            <div class="insight-item"><div class="insight-dot" style="background:#8b5cf6"></div><span>batting_impact ✦</span></div>
            <div class="insight-item"><div class="insight-dot" style="background:#8b5cf6"></div><span>bowling_impact ✦</span></div>
            <div class="insight-item"><div class="insight-dot" style="background:#8b5cf6"></div><span>experience_score ✦</span></div>
            <div class="insight-item"><div class="insight-dot" style="background:#8b5cf6"></div><span>consistency_score ✦</span></div>
        </div>
        <div style="color:#8b949e; font-size:0.72rem; margin-top:0.5rem;">✦ Engineered features</div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding:2rem; margin-top:1rem; background:#161b22;
                border:1px solid #21262d; border-radius:12px;">
        <div style="color:#8b949e; font-size:0.8rem;">
            Built as a portfolio-quality AI/ML project showcasing end-to-end machine learning,
            feature engineering, and professional dashboard design.
        </div>
        <div style="color:#58a6ff; font-size:0.75rem; margin-top:0.5rem; font-weight:600;">
            AI Sports Talent Scout — 2025
        </div>
    </div>
    """, unsafe_allow_html=True)