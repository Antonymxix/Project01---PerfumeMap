import streamlit as st
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

st.set_page_config(
    page_title="ScentExplorer AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load CSS from file — avoids any string escaping issues
css = Path("styles.css").read_text()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_data():
    df = pd.read_csv("perfumes.csv")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    descriptions = df["Brand"] + " " + df["Main_Notes"] + " " + df["Accords"]
    embeddings = model.encode(descriptions.tolist(), show_progress_bar=False)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings_scaled)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    return df, pca.explained_variance_ratio_


df, variance_ratio = load_data()
n_accords = int(df["Accords"].nunique())
var_pct = int((variance_ratio[0] + variance_ratio[1]) * 100)

PILL = (
    "display:inline-flex;align-items:center;gap:6px;padding:8px 16px;"
    "border-radius:16px;background:rgba(255,255,255,0.7);"
    "border:1px solid rgba(148,163,184,0.25);"
    "box-shadow:inset 0 1px 0 rgba(255,255,255,0.9);"
    "font-size:13px;font-weight:600;color:#334155;"
)
DOT = "width:8px;height:8px;border-radius:50%;display:inline-block;"

# ── HERO ──────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown(
        "<div style='display:inline-flex;align-items:center;gap:8px;padding:6px 16px;"
        "border-radius:999px;background:rgba(124,58,237,0.09);border:1px solid rgba(124,58,237,0.22);"
        "color:#7c3aed;font-size:12px;font-weight:600;letter-spacing:0.04em;margin-bottom:1.2rem;'>"
        "&#9672;&nbsp;AI-Powered &middot; Sentence Transformers + PCA</div>"

        "<div style='font-size:3rem;font-weight:900;letter-spacing:-0.02em;line-height:1.05;"
        "color:#0f172a;margin-bottom:0.5rem;'>Scent"
        "<span style='background:linear-gradient(135deg,#7c3aed,#6366f1,#0ea5e9);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>Explorer</span> AI</div>"

        "<div style='font-size:0.95rem;color:#64748b;line-height:1.6;max-width:560px;margin-bottom:1.5rem;'>"
        "100 designer fragrances mapped by olfactory similarity &mdash; an AI model analyzes each "
        "scent&#39;s notes and accords, then projects them into 2D space. "
        "Similar fragrances cluster together.</div>"

        "<div style='display:flex;flex-wrap:wrap;gap:8px;'>"
        f"<div style='{PILL}'><span style='{DOT}background:#7c3aed;margin-right:4px;'></span>100 Fragrances</div>"
        f"<div style='{PILL}'><span style='{DOT}background:#ec4899;margin-right:4px;'></span>{n_accords} Accord Groups</div>"
        f"<div style='{PILL}'><span style='{DOT}background:#10b981;margin-right:4px;'></span>{var_pct}% Variance explained</div>"
        "</div>",
        unsafe_allow_html=True,
    )

# ── FILTERS ───────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown(
        "<p style='font-size:11px;font-weight:700;letter-spacing:0.12em;"
        "text-transform:uppercase;color:#94a3b8;margin-bottom:0.8rem;margin-top:0;'>"
        "Filter &amp; Explore</p>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([4, 1])
    with col1:
        all_accords = sorted(df["Accords"].unique())
        selected_accords = st.multiselect(
            "", options=all_accords,
            placeholder="Filter by accord…",
            label_visibility="collapsed",
        )
    with col2:
        if "show_labels" not in st.session_state:
            st.session_state.show_labels = True
        btn_label = "◉ Labels" if st.session_state.show_labels else "◎ Labels"
        btn_style = (
            "background:rgba(124,58,237,0.12);border:1.5px solid rgba(124,58,237,0.35);"
            "color:#7c3aed;font-weight:700;font-size:13px;border-radius:14px;"
            "padding:8px 18px;cursor:pointer;width:100%;font-family:'DM Sans',sans-serif;"
        ) if st.session_state.show_labels else (
            "background:rgba(148,163,184,0.10);border:1.5px solid rgba(148,163,184,0.28);"
            "color:#94a3b8;font-weight:700;font-size:13px;border-radius:14px;"
            "padding:8px 18px;cursor:pointer;width:100%;font-family:'DM Sans',sans-serif;"
        )
        if st.button(btn_label, key="labels_btn", use_container_width=True):
            st.session_state.show_labels = not st.session_state.show_labels
            st.rerun()
        show_labels = st.session_state.show_labels

# ── FILTER LOGIC ──────────────────────────────────────────────────────────────
filtered = df.copy()
if selected_accords:
    filtered = filtered[filtered["Accords"].isin(selected_accords)]

# ── COLOR MAP ─────────────────────────────────────────────────────────────────
accord_list = sorted(df["Accords"].unique())
palette = [
    "#7c3aed","#0ea5e9","#ec4899","#10b981","#f59e0b","#6366f1","#14b8a6",
    "#f43f5e","#8b5cf6","#06b6d4","#a855f7","#22c55e","#ef4444","#3b82f6",
    "#d946ef","#84cc16","#fb923c","#e879f9","#2dd4bf","#facc15",
]
color_map = {acc: palette[i % len(palette)] for i, acc in enumerate(accord_list)}

# ── CHART ─────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown(
        "<p style='font-size:11px;font-weight:700;letter-spacing:0.12em;"
        "text-transform:uppercase;color:#94a3b8;margin-bottom:0.5rem;margin-top:0;'>"
        "Fragrance Space</p>",
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    for accord in filtered["Accords"].unique():
        group = filtered[filtered["Accords"] == accord]
        color = color_map.get(accord, "#94a3b8")
        fig.add_trace(go.Scatter(
            x=group["x"], y=group["y"],
            mode="markers+text" if show_labels else "markers",
            name=accord,
            text=group["Name"] if show_labels else None,
            textposition="top center",
            textfont=dict(family="DM Sans, system-ui", size=10, color="#475569"),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "<span style='color:#94a3b8'>%{customdata[1]}</span><br>"
                "<span style='color:#7c3aed'>%{customdata[2]}</span>"
                "<extra></extra>"
            ),
            customdata=group[["Name", "Brand", "Accords"]].values,
            marker=dict(size=10, color=color, opacity=0.85,
                        line=dict(width=1.5, color="rgba(255,255,255,0.8)")),
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.5)",
        font=dict(family="DM Sans, system-ui", color="#475569"),
        height=600,
        margin=dict(l=20, r=20, t=10, b=40),
        xaxis=dict(
            title=dict(text="← More Masculine · More Feminine →", font=dict(size=11, color="#94a3b8")),
            showgrid=True, gridcolor="rgba(148,163,184,0.15)",
            zeroline=True, zerolinecolor="rgba(148,163,184,0.30)", zerolinewidth=1,
            tickfont=dict(size=10, color="#94a3b8"), showline=False,
        ),
        yaxis=dict(
            title=dict(text="← Lighter · Heavier →", font=dict(size=11, color="#94a3b8")),
            showgrid=True, gridcolor="rgba(148,163,184,0.15)",
            zeroline=True, zerolinecolor="rgba(148,163,184,0.30)", zerolinewidth=1,
            tickfont=dict(size=10, color="#94a3b8"), showline=False,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.82)", bordercolor="rgba(148,163,184,0.20)",
            borderwidth=1, font=dict(size=11, color="#475569"), itemsizing="constant",
            title=dict(text="Accords", font=dict(size=11, color="#94a3b8")),
            x=1.01, y=1,
        ),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.95)", bordercolor="rgba(124,58,237,0.25)",
            font=dict(family="DM Sans, system-ui", size=12, color="#1e293b"),
        ),
        dragmode="pan",
    )

    if len(filtered) == 0:
        st.info("No fragrances match — try a different filter.")
    else:
        st.plotly_chart(fig, use_container_width=True, config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
            "displaylogo": False,
            "scrollZoom": True,
        })

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;padding:1.5rem 0 0.5rem;color:#94a3b8;font-size:12px;'>"
    "Built by <strong style='color:#7c3aed'>Anton Schindler</strong> &nbsp;&middot;&nbsp; "
    "AI &amp; Data Science &nbsp;&middot;&nbsp;"
    "<a href='https://github.com/Antonymxix' style='color:#7c3aed;text-decoration:none;'>GitHub &#8599;</a>"
    "</div>",
    unsafe_allow_html=True,
)