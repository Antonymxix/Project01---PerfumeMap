import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="ScentExplorer AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;600;700;900&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {
        font-family: 'DM Sans', system-ui, sans-serif !important;
    }

    .stApp {
        background: #eef0f8;
    }

    /* Animated background blobs */
    .stApp::before {
        content: '';
        position: fixed;
        width: 700px; height: 700px;
        top: -200px; left: -150px;
        background: rgba(149,100,255,0.35);
        border-radius: 50%;
        filter: blur(180px);
        z-index: 0;
        pointer-events: none;
    }
    .stApp::after {
        content: '';
        position: fixed;
        width: 580px; height: 580px;
        top: 10%; right: -120px;
        background: rgba(56,189,248,0.28);
        border-radius: 50%;
        filter: blur(160px);
        z-index: 0;
        pointer-events: none;
    }

    /* Hide streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding: 2.5rem 3rem 3rem 3rem !important;
        max-width: 1400px !important;
        position: relative;
        z-index: 1;
    }

    /* Glass card base */
    .glass-card {
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(255,255,255,0.70);
        border-radius: 24px;
        box-shadow: 0 4px 32px rgba(0,0,0,0.07), 0 1px 4px rgba(0,0,0,0.04), inset 0 1.5px 0 rgba(255,255,255,0.95);
        padding: 2rem 2.5rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0; left: 4%; right: 4%; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.95), transparent);
    }

    /* Hero section */
    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 999px;
        background: rgba(124,58,237,0.09);
        border: 1px solid rgba(124,58,237,0.22);
        color: #7c3aed;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        line-height: 1.05;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    .hero-title span {
        background: linear-gradient(135deg, #7c3aed, #6366f1, #0ea5e9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #64748b;
        font-weight: 400;
        line-height: 1.6;
        max-width: 560px;
    }

    /* Stat pills */
    .stat-row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 1.5rem;
    }
    .stat-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 16px;
        border-radius: 16px;
        background: rgba(255,255,255,0.7);
        border: 1px solid rgba(148,163,184,0.25);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.9);
        font-size: 13px;
        font-weight: 600;
        color: #334155;
    }
    .stat-pill .dot {
        width: 8px; height: 8px;
        border-radius: 50%;
    }

    /* Section label */
    .section-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }

    /* Filter chips */
    .stMultiSelect > div {
        border-radius: 16px !important;
    }
    div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.7) !important;
        border: 1px solid rgba(148,163,184,0.30) !important;
        border-radius: 16px !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.9) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Plotly chart container */
    .js-plotly-plot {
        border-radius: 20px;
        overflow: hidden;
    }

    /* Search input */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.75) !important;
        border: 1px solid rgba(148,163,184,0.28) !important;
        border-radius: 16px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14px !important;
        padding: 0.6rem 1rem !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.9) !important;
        color: #1e293b !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: rgba(124,58,237,0.45) !important;
        box-shadow: 0 0 0 3px rgba(124,58,237,0.10), inset 0 1px 0 rgba(255,255,255,0.9) !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.82) !important;
        border-right: 1px solid rgba(255,255,255,0.60) !important;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(148,163,184,0.25), transparent);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data():
    df = pd.read_csv('perfumes.csv')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    descriptions = df['Brand'] + " " + df['Main_Notes'] + " " + df['Accords']
    embeddings = model.encode(descriptions.tolist(), show_progress_bar=False)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings_scaled)
    df['x'] = coords[:, 0]
    df['y'] = coords[:, 1]
    return df, pca.explained_variance_ratio_


df, variance_ratio = load_data()

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="glass-card">
  <div class="hero-eyebrow">◈ &nbsp;AI-Powered · Sentence Transformers + PCA</div>
  <div class="hero-title">Scent<span>Explorer</span> AI</div>
  <div class="hero-sub">
    100 designer fragrances mapped by olfactory similarity — an AI model analyzes each scent's notes and accords, then projects them into 2D space. Similar fragrances cluster together.
  </div>
  <div class="stat-row">
    <div class="stat-pill"><span class="dot" style="background:#7c3aed"></span> 100 Fragrances</div>
    <div class="stat-pill"><span class="dot" style="background:#ec4899"></span> {accords} Accord Groups</div>
    <div class="stat-pill"><span class="dot" style="background:#10b981"></span> {var:.0f}% Variance explained</div>
  </div>
</div>
""".format(
    accords=df['Accords'].nunique(),
    var=(variance_ratio[0] + variance_ratio[1]) * 100
), unsafe_allow_html=True)

# ── FILTERS ───────────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Filter & Explore</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    all_accords = sorted(df['Accords'].unique())
    selected_accords = st.multiselect("", options=all_accords, placeholder="Filter by accord…", label_visibility="collapsed")
with col2:
    st.markdown("""
    <style>
    div[data-testid="stToggle"] label {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #475569 !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    div[data-testid="stToggle"] p {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }
    div[data-testid="stToggle"] [data-checked="true"] {
        background-color: #7c3aed !important;
    }
    </style>
    """, unsafe_allow_html=True)
    show_labels = st.toggle("Fragrance Labels", value=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── FILTER LOGIC ──────────────────────────────────────────────────────────────
filtered = df.copy()
if selected_accords:
    filtered = filtered[filtered['Accords'].isin(selected_accords)]

# ── COLOR MAP ─────────────────────────────────────────────────────────────────
accord_list = sorted(df['Accords'].unique())
palette = [
    "#7c3aed","#0ea5e9","#ec4899","#10b981","#f59e0b","#6366f1","#14b8a6",
    "#f43f5e","#8b5cf6","#06b6d4","#a855f7","#22c55e","#ef4444","#3b82f6",
    "#d946ef","#84cc16","#fb923c","#e879f9","#2dd4bf","#facc15",
]
color_map = {acc: palette[i % len(palette)] for i, acc in enumerate(accord_list)}

# ── CHART ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card" style="padding: 1.5rem;">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Fragrance Space</div>', unsafe_allow_html=True)

fig = go.Figure()

for accord in filtered['Accords'].unique():
    group = filtered[filtered['Accords'] == accord]
    color = color_map.get(accord, "#94a3b8")

    fig.add_trace(go.Scatter(
        x=group['x'],
        y=group['y'],
        mode='markers+text' if show_labels else 'markers',
        name=accord,
        text=group['Name'] if show_labels else None,
        textposition='top center',
        textfont=dict(family='DM Sans, system-ui', size=10, color='#475569'),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "<span style='color:#94a3b8'>%{customdata[1]}</span><br>"
            "<span style='color:#7c3aed'>%{customdata[2]}</span>"
            "<extra></extra>"
        ),
        customdata=group[['Name', 'Brand', 'Accords']].values,
        marker=dict(
            size=10,
            color=color,
            opacity=0.85,
            line=dict(width=1.5, color='rgba(255,255,255,0.8)'),
        ),
    ))

fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(248,250,252,0.6)',
    font=dict(family='DM Sans, system-ui', color='#475569'),
    height=580,
    margin=dict(l=20, r=20, t=20, b=40),
    xaxis=dict(
        title=dict(text="← More Masculine · More Feminine →", font=dict(size=11, color='#94a3b8')),
        showgrid=True,
        gridcolor='rgba(148,163,184,0.15)',
        zeroline=True,
        zerolinecolor='rgba(148,163,184,0.30)',
        zerolinewidth=1,
        tickfont=dict(size=10, color='#94a3b8'),
        showline=False,
    ),
    yaxis=dict(
        title=dict(text="← Lighter · Heavier →", font=dict(size=11, color='#94a3b8')),
        showgrid=True,
        gridcolor='rgba(148,163,184,0.15)',
        zeroline=True,
        zerolinecolor='rgba(148,163,184,0.30)',
        zerolinewidth=1,
        tickfont=dict(size=10, color='#94a3b8'),
        showline=False,
    ),
    legend=dict(
        bgcolor='rgba(255,255,255,0.80)',
        bordercolor='rgba(148,163,184,0.25)',
        borderwidth=1,
        font=dict(size=11, color='#475569'),
        itemsizing='constant',
        title=dict(text='Accords', font=dict(size=11, color='#94a3b8', weight=700)),
        x=1.01, y=1,
    ),
    hoverlabel=dict(
        bgcolor='rgba(255,255,255,0.95)',
        bordercolor='rgba(124,58,237,0.25)',
        font=dict(family='DM Sans, system-ui', size=12, color='#1e293b'),
    ),
    showlegend=True,
    dragmode='pan',
)

st.plotly_chart(fig, use_container_width=True, config={
    'displayModeBar': True,
    'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'autoScale2d'],
    'displaylogo': False,
    'scrollZoom': True,
})

if len(filtered) == 0:
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #94a3b8; font-size: 14px;">
        No fragrances found — try a different search or filter.
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 1.5rem 0 0.5rem; color: #94a3b8; font-size: 12px; font-family: 'DM Sans', sans-serif;">
    Built by <strong style="color:#7c3aed">Anton Schindler</strong> &nbsp;·&nbsp;
    AI & Data Science &nbsp;·&nbsp;
    <a href="https://github.com/Antonymxix" style="color:#7c3aed; text-decoration:none;">GitHub ↗</a>
</div>
""", unsafe_allow_html=True)