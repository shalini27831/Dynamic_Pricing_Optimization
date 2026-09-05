import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Dynamic Pricing Optimization Engine",
    page_icon="₹",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PROFESSIONAL THEME — TYPOGRAPHY, COLOR SYSTEM, LAYOUT
# ============================================================

# Design tokens (colors only — no logic below this point is changed)
PLOTLY_COLORWAY = [
    "#6C5CE7",   # primary line (purple accent — matches navbar/brand)
    "#5B7FFF",   # secondary series (blue, for contrast without noise)
    "#F5C451",   # tertiary / callout marker (gold, used sparingly)
    "#8A93A6",   # quaternary series (neutral slate)
]
PLOTLY_FONT = "Manrope, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
PLOTLY_TEXT_COLOR = "#E7EAEF"
PLOTLY_GRID_COLOR = "#22253F"
PLOTLY_PAPER_COLOR = "#14162B"

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@500;600;700&display=swap" rel="stylesheet">

    <style>

    :root {
        /* Text/surfaces — unified dark navy/purple theme */
        --ink-900: #F5F7FA;
        --ink-700: #C9CCE3;
        --ink-500: #9195B8;
        --ink-300: #696E93;
        --surface-0: #14162B;
        --surface-1: #0A0B1A;
        --surface-2: #1C1F3D;
        --line: #262A45;
        --line-bright: #3A3F6690;
        --accent-700: #5647C7;
        --accent-600: #6C5CE7;
        --accent-500: #8B7CF6;
        --accent-100: #221F45;
        --gold-600: #F5C451;
        --success-600: #2ED47A;
        --warning-600: #FF5C6C;
        --blue-600: #5B7FFF;
        --pink-600: #F06FA0;
        --glow: 0 0 0 1px var(--line), 0 8px 24px -8px rgba(108, 92, 231, 0.28);

        /* Same palette — page is dark everywhere now, no separate light-page text needed */
        --page-ink-900: #F5F7FA;
        --page-ink-500: #9195B8;
        --page-ink-300: #696E93;
    }

    html, body, [class*="css"] {
        font-family: 'Manrope', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* Kill Streamlit's default top header bar/whitespace */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    div[data-testid="stDecoration"] {
        display: none !important;
    }

    div[data-testid="stStatusWidget"] {
        display: none !important;
    }

    div[data-testid="stAppViewContainer"] > .main {
        padding-top: 0rem;
    }

    div[data-testid="stToolbar"] {
        display: none !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(108, 92, 231, 0.10), transparent 45%),
            radial-gradient(circle at 85% 10%, rgba(91, 127, 255, 0.07), transparent 40%),
            var(--surface-1);
    }

    .block-container {
        max-width: 1320px;
        padding-top: 0.4rem;
        padding-bottom: 4rem;
    }

    /* ---------------- Top navbar ---------------- */

    .app-navbar {
        position: sticky;
        top: 0;
        z-index: 999;
        background: rgba(20, 22, 43, 0.92);
        backdrop-filter: blur(10px);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0 20px;
        margin-bottom: 1.8rem;
    }

    .app-navbar-inner {
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
        flex-wrap: nowrap;
    }

    .app-navbar-brand {
        display: flex;
        align-items: center;
        gap: 9px;
        font-weight: 800;
        font-size: 1.05rem;
        color: var(--ink-900);
        letter-spacing: -0.01em;
        white-space: nowrap;
    }

    .app-navbar-brand .brand-text {
        color: var(--ink-900);
    }

    .navbar-status-pill .pill-label {
        color: var(--accent-500);
    }

    .app-navbar-brand .logo-dot {
        width: 26px;
        height: 26px;
        border-radius: 8px;
        background: linear-gradient(135deg, var(--accent-600), var(--blue-600));
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        box-shadow: 0 4px 14px -4px rgba(108, 92, 231, 0.6);
    }

    .app-navbar-links {
        display: flex;
        align-items: center;
        gap: 26px;
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--ink-500);
        flex: 1;
        padding-left: 12px;
    }

    .app-navbar-links a {
        color: var(--ink-500);
        text-decoration: none;
        cursor: pointer;
        padding-bottom: 20px;
        border-bottom: 2px solid transparent;
        transition: color 0.15s ease, border-color 0.15s ease;
    }

    .app-navbar-links a:hover {
        color: var(--ink-700);
    }

    .app-navbar-links a.active {
        color: var(--ink-900);
        border-bottom: 2px solid var(--accent-600);
    }

    .app-navbar-right {
        display: flex;
        align-items: center;
        gap: 16px;
        white-space: nowrap;
    }

    .navbar-icon {
        width: 34px;
        height: 34px;
        border-radius: 9px;
        background: var(--surface-2);
        border: 1px solid var(--line);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        color: var(--ink-500);
    }

    .navbar-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: var(--accent-100);
        border: 1px solid var(--line-bright);
        color: var(--accent-500);
        font-weight: 700;
        font-size: 0.78rem;
        padding: 7px 14px 7px 12px;
        border-radius: 999px;
    }

    .navbar-status-pill::before {
        content: "";
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--success-600);
        box-shadow: 0 0 0 3px rgba(46, 212, 122, 0.2);
        margin-right: 2px;
    }

    .app-navbar-profile {
        display: flex;
        align-items: center;
        gap: 9px;
        padding-left: 14px;
        border-left: 1px solid var(--line);
    }

    .app-navbar-profile .avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--pink-600), var(--accent-600));
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.78rem;
        font-weight: 800;
        color: white;
    }

    .app-navbar-profile .who .name {
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--ink-900);
        line-height: 1.15;
    }

    .app-navbar-profile .who .role {
        font-size: 0.72rem;
        color: var(--ink-300);
        line-height: 1.15;
    }

    @media (max-width: 900px) {
        .app-navbar-links { display: none; }
        .app-navbar-profile .who { display: none; }
    }

    /* ---------------- Dashboard header row ---------------- */

    .dash-header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 16px;
        margin: 0.2rem 0 1.6rem 0;
    }

    .dash-header-row .dash-heading h1 {
        margin-bottom: 4px !important;
    }

    .dash-header-badges {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }

    .dash-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: var(--surface-0);
        border: 1px solid var(--line);
        color: var(--ink-700);
        font-weight: 700;
        font-size: 0.82rem;
        padding: 8px 15px;
        border-radius: 999px;
    }

    .dash-pill.accent {
        background: linear-gradient(135deg, var(--accent-700), var(--accent-600));
        border: none;
        color: white;
        box-shadow: 0 6px 18px -6px rgba(108, 92, 231, 0.55);
    }

    section[data-testid="stSidebar"] {
        display: none;
    }

    /* ---------------- Hero header ---------------- */

    .hero-banner {
        background: linear-gradient(135deg, var(--surface-0) 0%, #141A24 100%);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 28px 32px;
        margin-bottom: 1.8rem;
        box-shadow: var(--glow);
        position: relative;
        overflow: hidden;
    }

    .hero-banner::before {
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 4px;
        height: 100%;
        background: var(--accent-600);
    }

    .hero-eyebrow {
        color: var(--accent-500);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .hero-title {
        font-family: 'Manrope', -apple-system, sans-serif;
        color: var(--ink-900);
        font-weight: 800;
        letter-spacing: -0.02em;
        font-size: 2.15rem;
        margin: 0 0 8px 0;
        line-height: 1.15;
    }

    .hero-subtitle {
        color: var(--ink-500);
        font-size: 0.97rem;
        line-height: 1.5;
        max-width: 780px;
        margin: 0;
    }

    /* ---------------- Typography ---------------- */

    h1 {
        font-family: 'Manrope', -apple-system, sans-serif !important;
        color: var(--ink-900) !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        font-size: 2.05rem !important;
        margin-bottom: 0.15rem !important;
    }

    h2 {
        color: var(--ink-900) !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }

    h3 {
        color: var(--ink-900) !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        letter-spacing: -0.005em;
        margin-top: 0.6rem !important;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--line);
    }

    p, span, label, li {
        color: var(--ink-500);
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--page-ink-300) !important;
        font-size: 0.9rem !important;
    }

    /* ---------------- Metric cards ---------------- */

    div[data-testid="stMetric"] {
        background: linear-gradient(160deg, var(--surface-0) 0%, #181B38 100%);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 1px 4px rgba(4, 5, 14, 0.35);
        transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 6px;
        position: relative;
        overflow: hidden;
    }

    div[data-testid="stMetric"]::before {
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 100%;
        height: 2px;
        border-radius: 14px 14px 0 0;
        background: linear-gradient(90deg, var(--accent-600), var(--blue-600) 60%, transparent 100%);
        opacity: 0.85;
    }

    div[data-testid="stMetric"]:hover {
        border-color: var(--accent-700);
        box-shadow: 0 10px 26px -10px rgba(108, 92, 231, 0.35);
        transform: translateY(-1px);
    }

    /* Small rounded icon badge, top-right of each metric card — cycles per column */
    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(4n+1) div[data-testid="stMetric"]::after,
    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(4n+2) div[data-testid="stMetric"]::after,
    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(4n+3) div[data-testid="stMetric"]::after,
    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(4n+4) div[data-testid="stMetric"]::after {
        position: absolute;
        top: 16px;
        right: 16px;
        width: 30px;
        height: 30px;
        border-radius: 8px;
        background: var(--surface-2);
        border: 1px solid var(--line);
        text-align: center;
        line-height: 30px;
        font-size: 0.82rem;
    }

    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(4n+1) div[data-testid="stMetric"]::after { content: "\1F4B3"; }
    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(4n+2) div[data-testid="stMetric"]::after { content: "\1F4E6"; }
    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(4n+3) div[data-testid="stMetric"]::after { content: "\1F3F7"; }
    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(4n+4) div[data-testid="stMetric"]::after { content: "\1F4C8"; }

    div[data-testid="stMetricLabel"] {
        color: var(--ink-300) !important;
        font-weight: 700 !important;
        font-size: 0.76rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    div[data-testid="stMetricValue"] {
        color: var(--ink-900) !important;
        font-weight: 800 !important;
        font-size: 1.65rem !important;
        letter-spacing: -0.01em;
    }

    div[data-testid="stMetricDelta"] {
        font-weight: 700 !important;
        font-size: 0.8rem !important;
        display: inline-flex !important;
        width: fit-content;
        padding: 2px 8px 2px 6px;
        border-radius: 999px;
        background: rgba(145, 149, 184, 0.12);
    }

    div[data-testid="stHorizontalBlock"] {
        align-items: stretch;
    }

    /* ---------------- Inputs ---------------- */

    div[data-baseweb="select"] > div {
        background-color: var(--surface-0);
        border-radius: 8px;
        border-color: var(--line) !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: var(--accent-500) !important;
    }

    div[data-baseweb="select"]:focus-within > div {
        border-color: var(--accent-600) !important;
        box-shadow: 0 0 0 3px rgba(108, 92, 231, 0.22) !important;
    }

    /* Selected value text + dropdown arrow icon */
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input {
        color: var(--ink-900) !important;
        opacity: 1 !important;
    }

    div[data-baseweb="select"] svg {
        fill: var(--ink-500) !important;
    }

    /* Dropdown option list (menu that pops open) */
    ul[data-testid="stSelectboxVirtualDropdown"] {
        background-color: var(--surface-2) !important;
        border: 1px solid var(--line) !important;
    }

    ul[data-testid="stSelectboxVirtualDropdown"] li {
        color: var(--ink-900) !important;
    }

    ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
        background-color: var(--accent-100) !important;
    }

    label[data-testid="stWidgetLabel"] p {
        color: var(--page-ink-500) !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    /* ---------------- Alerts ---------------- */

    div[data-testid="stAlert"] {
        border-radius: 10px;
        border: 1px solid transparent;
        font-size: 0.95rem;
        box-shadow: 0 2px 10px rgba(16, 21, 28, 0.15);
    }

    /* ---------------- Buttons ---------------- */

    .stButton > button {
        background: linear-gradient(135deg, var(--accent-700), var(--accent-600));
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 700;
        padding: 0.5rem 1.1rem;
        letter-spacing: 0.01em;
        transition: filter 0.15s ease, transform 0.15s ease;
    }

    .stButton > button:hover {
        filter: brightness(1.1);
        color: white;
        transform: translateY(-1px);
    }

    /* ---------------- Tables ---------------- */

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(16, 21, 28, 0.15);
        font-size: 0.86rem;
    }

    div[data-testid="stDataFrame"] div[data-testid="stDataFrameResizable"] {
        font-size: 0.86rem;
    }

    div[data-testid="stDataFrame"] [role="row"] {
        min-height: 30px !important;
    }

    div[data-testid="stDataFrame"] [role="gridcell"],
    div[data-testid="stDataFrame"] [role="columnheader"] {
        padding-top: 2px !important;
        padding-bottom: 2px !important;
    }

    /* ---------------- Pricing Actions Table (custom) ---------------- */

    .pricing-table-wrap {
        border: 1px solid var(--line);
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(16, 21, 28, 0.15);
        background: var(--surface-0);
    }

    table.pricing-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
    }

    table.pricing-table thead tr {
        background: linear-gradient(180deg, #241F4D 0%, #1B1740 100%);
    }

    table.pricing-table thead th {
        text-align: left;
        padding: 10px 16px;
        font-weight: 700;
        font-size: 0.82rem;
        letter-spacing: 0.02em;
        color: #FFFFFF;
        border-bottom: 1px solid var(--line-bright);
        white-space: nowrap;
    }

    table.pricing-table tbody td {
        padding: 8px 16px;
        color: var(--page-ink-900);
        border-bottom: 1px solid var(--line);
        white-space: nowrap;
    }

    table.pricing-table tbody tr:nth-child(even):not(.reco-row) {
        background: rgba(108, 92, 231, 0.04);
    }

    table.pricing-table tbody tr:hover:not(.reco-row) {
        background: rgba(108, 92, 231, 0.10);
    }

    table.pricing-table tbody tr:last-child td {
        border-bottom: none;
    }

    table.pricing-table tbody tr.reco-row td {
        background: #6C5CE7;
        color: #FFFFFF;
        font-weight: 700;
    }

    /* ---------------- Expanders ---------------- */

    details {
        background: linear-gradient(160deg, var(--surface-0) 0%, #181B38 100%);
        border: 1px solid var(--line) !important;
        border-radius: 10px !important;
    }

    details:hover {
        border-color: var(--line-bright) !important;
    }

    summary {
        font-weight: 700 !important;
        color: var(--accent-500) !important;
    }

    /* ---------------- Dividers ---------------- */

    hr {
        border-color: var(--line);
        margin: 1.6rem 0;
    }

    /* ---------------- Section label (icon + text) ---------------- */

    .section-label {
        scroll-margin-top: 84px;
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 700;
        font-size: 1.02rem;
        color: var(--page-ink-900);
        margin: 0.4rem 0 0.9rem 0;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--line);
    }

    .section-label .icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 26px;
        border-radius: 7px;
        background: var(--accent-100);
        font-size: 0.9rem;
    }

    /* ---------------- Recommendation card ---------------- */

    .reco-card {
        background: var(--surface-0);
        border-radius: 12px;
        border: 1px solid var(--line);
        padding: 22px 26px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 18px;
        position: relative;
        overflow: hidden;
    }

    .reco-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 5px;
        height: 100%;
    }

    .reco-card.increase {
        background: linear-gradient(135deg, rgba(46, 212, 122, 0.10), rgba(46, 212, 122, 0.02)), var(--surface-0);
    }
    .reco-card.increase::before { background: var(--success-600); }

    .reco-card.decrease {
        background: linear-gradient(135deg, rgba(255, 92, 108, 0.10), rgba(255, 92, 108, 0.02)), var(--surface-0);
    }
    .reco-card.decrease::before { background: var(--warning-600); }

    .reco-card.keep {
        background: linear-gradient(135deg, rgba(108, 92, 231, 0.10), rgba(108, 92, 231, 0.02)), var(--surface-0);
    }
    .reco-card.keep::before { background: var(--accent-600); }

    .reco-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 999px;
        margin-bottom: 8px;
    }

    .reco-card.increase .reco-badge { background: rgba(46, 212, 122, 0.16); color: var(--success-600); }
    .reco-card.decrease .reco-badge { background: rgba(255, 92, 108, 0.16); color: var(--warning-600); }
    .reco-card.keep .reco-badge { background: rgba(108, 92, 231, 0.16); color: var(--accent-500); }

    .reco-action {
        font-family: 'IBM Plex Mono', 'Manrope', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: var(--ink-900);
        margin: 0;
    }

    .reco-stats {
        display: flex;
        gap: 34px;
        flex-wrap: wrap;
    }

    .reco-stat-label {
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--ink-300);
        margin-bottom: 3px;
    }

    .reco-stat-value {
        font-size: 1.28rem;
        font-weight: 800;
        color: var(--ink-900);
        letter-spacing: -0.01em;
    }

    /* ---------------- Info grid (Model Information) ---------------- */

    .info-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 14px 28px;
        padding: 6px 2px 2px 2px;
    }

    .info-item {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }

    .info-item .info-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--ink-300);
    }

    .info-item .info-value {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--ink-900);
    }

    .info-item .info-value code {
        background: var(--surface-2);
        color: var(--accent-500);
        padding: 1px 6px;
        border-radius: 5px;
        font-size: 0.85rem;
    }

    /* ---------------- Recommended row highlight ---------------- */

    .best-price-note {
        color: var(--page-ink-300);
        font-size: 0.85rem;
        margin-top: -4px;
        margin-bottom: 0.6rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TOP NAVBAR (visual only — no routing/logic attached)
# ============================================================

navbar_html = (
    '<div class="app-navbar"><div class="app-navbar-inner">'
    '<div class="app-navbar-brand"><span class="logo-dot">₹</span><span class="brand-text">Dynamic Pricing Engine</span></div>'
    '<div class="app-navbar-links">'
    '<a href="#nav-forecast" class="active">Forecast</a>'
    '<a href="#nav-recommendation">Recommendation</a>'
    '<a href="#nav-optimization">Optimization</a>'
    '<a href="#nav-actions">Actions</a>'
    '</div>'
    '<div class="app-navbar-right">'
    '<span class="navbar-icon" title="LightGBM V3 High-Demand">🧠</span>'
    '<span class="navbar-icon" title="Currency: INR">₹</span>'
    '<span class="navbar-status-pill"><span class="pill-label">Model Live</span></span>'
    '<div class="app-navbar-profile"><span class="avatar">V3</span>'
    '<div class="who"><div class="name">LightGBM V3</div><div class="role">High-Demand Model</div></div>'
    '</div>'
    '</div>'
    '</div></div>'
)

st.markdown(navbar_html, unsafe_allow_html=True)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "lightgbm_v3_high_demand.pkl"
)

DATA_PATH = (
    BASE_DIR
    / "data"
    / "forecast_model_data_deploy.csv"
)

ELASTICITY_PATH = (
    BASE_DIR
    / "reports"
    / "reliable_price_elasticity.csv"
)


# ============================================================
# LOAD V3 MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        st.error(
            f"V3 model not found:\n\n{MODEL_PATH}"
        )
        st.stop()

    with open(MODEL_PATH, "rb") as file:
        package = pickle.load(file)

    if isinstance(package, dict):

        model = package.get("model")
        features = package.get("features")

        if model is None:
            st.error(
                "The V3 model file does not contain a valid model."
            )
            st.stop()

        if features is None:
            st.error(
                "The V3 model file does not contain feature names."
            )
            st.stop()

    else:

        model = package
        features = None

    return model, features


# ============================================================
# LOAD FORECAST DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        st.error(
            f"Forecast dataset not found:\n\n{DATA_PATH}"
        )
        st.stop()

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "Date",
        "StockCode",
        "Demand",
        "AveragePrice_INR"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        st.error(
            "Required columns are missing from "
            "forecast_model_data.csv:\n\n"
            + ", ".join(missing_columns)
        )
        st.stop()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["StockCode"] = (
        df["StockCode"]
        .astype(str)
        .str.strip()
    )

    df = df.dropna(
        subset=["Date", "StockCode"]
    )

    return df


# ============================================================
# LOAD ELASTICITY
# ============================================================

@st.cache_data
def load_elasticity():

    if not ELASTICITY_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        ELASTICITY_PATH
    )

    if "StockCode" in df.columns:
        df["StockCode"] = (
            df["StockCode"]
            .astype(str)
            .str.strip()
        )

    return df


# ============================================================
# LOAD MODEL / DATA
# ============================================================

model, model_features = load_model()

data = load_data()

elasticity_data = load_elasticity()


# ============================================================
# V3 FEATURE LIST
# ============================================================

V3_FEATURES = [
    "Demand",
    "Demand_Lag_1",
    "Demand_Lag_7",
    "Demand_Lag_14",
    "Demand_Lag_28",
    "Demand_RollingMean_7",
    "Demand_RollingMean_14",
    "Demand_RollingMean_28",
    "Demand_RollingStd_7",
    "Demand_RollingStd_28",
    "Demand_Trend_7_28",
    "AveragePrice_INR",
    "Price_Lag_1",
    "Price_Change_Pct",
    "Price_RollingMean_7",
    "Revenue_INR",
    "Revenue_Lag_1",
    "TransactionCount",
    "Transactions_RollingMean_7",
    "DayOfWeek",
    "WeekOfYear",
    "Month",
    "Quarter",
    "IsWeekend",
    "DayOfWeek_Sin",
    "DayOfWeek_Cos",
    "WeekOfYear_Sin",
    "WeekOfYear_Cos",
    "Month_Sin",
    "Month_Cos",
    "Product_Age_Days",
    "Observed",
    "Product_HistoricalMean",
    "Product_HistoricalMax",
    "Product_HistoricalNonZeroRate",
    "Product_HistoricalDays",
    "Demand_vs_ProductHistory"
]


# ============================================================
# PRODUCT DESCRIPTION
# ============================================================

def get_description(stock_code):

    stock_code = str(stock_code).strip()

    if "Description" not in data.columns:
        return "Product"

    rows = data[
        data["StockCode"] == stock_code
    ]

    if rows.empty:
        return "Product"

    descriptions = (
        rows["Description"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    descriptions = descriptions[
        descriptions != ""
    ]

    if descriptions.empty:
        return "Product"

    return descriptions.iloc[0]


# ============================================================
# GET ELASTICITY
# ============================================================

def get_elasticity(stock_code):

    default_elasticity = -0.692337

    if elasticity_data.empty:
        return default_elasticity, "Fallback"

    rows = elasticity_data[
        elasticity_data["StockCode"]
        == str(stock_code)
    ]

    if rows.empty:
        return default_elasticity, "Fallback"

    row = rows.iloc[0]

    value = row.get(
        "Elasticity",
        np.nan
    )

    if pd.isna(value):
        return default_elasticity, "Fallback"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return default_elasticity, "Fallback"

    value = np.clip(
        value,
        -3.0,
        -0.10
    )

    confidence = row.get(
        "Confidence",
        "Available"
    )

    return float(value), str(confidence)


# ============================================================
# CREATE V3 FEATURES
# ============================================================

def create_features(product_data):

    df = product_data.copy()

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Demand Lags
    # --------------------------------------------------------

    df["Demand_Lag_1"] = (
        df["Demand"].shift(1)
    )

    df["Demand_Lag_7"] = (
        df["Demand"].shift(7)
    )

    df["Demand_Lag_14"] = (
        df["Demand"].shift(14)
    )

    df["Demand_Lag_28"] = (
        df["Demand"].shift(28)
    )

    # --------------------------------------------------------
    # Rolling Demand
    # --------------------------------------------------------

    previous_demand = (
        df["Demand"].shift(1)
    )

    df["Demand_RollingMean_7"] = (
        previous_demand
        .rolling(
            7,
            min_periods=1
        )
        .mean()
    )

    df["Demand_RollingMean_14"] = (
        previous_demand
        .rolling(
            14,
            min_periods=1
        )
        .mean()
    )

    df["Demand_RollingMean_28"] = (
        previous_demand
        .rolling(
            28,
            min_periods=1
        )
        .mean()
    )

    df["Demand_RollingStd_7"] = (
        previous_demand
        .rolling(
            7,
            min_periods=2
        )
        .std()
    )

    df["Demand_RollingStd_28"] = (
        previous_demand
        .rolling(
            28,
            min_periods=2
        )
        .std()
    )

    df["Demand_Trend_7_28"] = (
        df["Demand_RollingMean_7"]
        -
        df["Demand_RollingMean_28"]
    )

    # --------------------------------------------------------
    # Price Features
    # --------------------------------------------------------

    df["Price_Lag_1"] = (
        df["AveragePrice_INR"].shift(1)
    )

    df["Price_Change_Pct"] = (
        df["AveragePrice_INR"]
        .pct_change()
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
        * 100
    )

    df["Price_RollingMean_7"] = (
        df["AveragePrice_INR"]
        .shift(1)
        .rolling(
            7,
            min_periods=1
        )
        .mean()
    )

    # --------------------------------------------------------
    # Revenue
    # --------------------------------------------------------

    if "Revenue_INR" not in df.columns:

        df["Revenue_INR"] = (
            df["Demand"]
            *
            df["AveragePrice_INR"]
        )

    df["Revenue_Lag_1"] = (
        df["Revenue_INR"].shift(1)
    )

    # --------------------------------------------------------
    # Transactions
    # --------------------------------------------------------

    if "TransactionCount" not in df.columns:

        df["TransactionCount"] = 0

    df["Transactions_RollingMean_7"] = (
        df["TransactionCount"]
        .shift(1)
        .rolling(
            7,
            min_periods=1
        )
        .mean()
    )

    # --------------------------------------------------------
    # Calendar Features
    # --------------------------------------------------------

    df["DayOfWeek"] = (
        df["Date"].dt.dayofweek
    )

    df["WeekOfYear"] = (
        df["Date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["Month"] = (
        df["Date"].dt.month
    )

    df["Quarter"] = (
        df["Date"].dt.quarter
    )

    df["IsWeekend"] = (
        df["DayOfWeek"] >= 5
    ).astype(int)

    df["DayOfWeek_Sin"] = np.sin(
        2
        * np.pi
        * df["DayOfWeek"]
        / 7
    )

    df["DayOfWeek_Cos"] = np.cos(
        2
        * np.pi
        * df["DayOfWeek"]
        / 7
    )

    df["WeekOfYear_Sin"] = np.sin(
        2
        * np.pi
        * df["WeekOfYear"]
        / 52
    )

    df["WeekOfYear_Cos"] = np.cos(
        2
        * np.pi
        * df["WeekOfYear"]
        / 52
    )

    df["Month_Sin"] = np.sin(
        2
        * np.pi
        * df["Month"]
        / 12
    )

    df["Month_Cos"] = np.cos(
        2
        * np.pi
        * df["Month"]
        / 12
    )

    # --------------------------------------------------------
    # Product Age
    # --------------------------------------------------------

    first_date = df["Date"].min()

    df["Product_Age_Days"] = (
        df["Date"] - first_date
    ).dt.days

    # --------------------------------------------------------
    # Observed
    # --------------------------------------------------------

    if "Observed" not in df.columns:

        df["Observed"] = (
            df["Demand"] > 0
        ).astype(int)

    # --------------------------------------------------------
    # Historical Product Features
    # --------------------------------------------------------

    previous_demand = (
        df["Demand"].shift(1)
    )

    df["Product_HistoricalMean"] = (
        previous_demand
        .expanding(
            min_periods=1
        )
        .mean()
    )

    df["Product_HistoricalMax"] = (
        previous_demand
        .expanding(
            min_periods=1
        )
        .max()
    )

    df["Product_HistoricalNonZeroRate"] = (
        previous_demand
        .gt(0)
        .expanding(
            min_periods=1
        )
        .mean()
    )

    df["Product_HistoricalDays"] = (
        np.arange(len(df))
    )

    df["Demand_vs_ProductHistory"] = (
        df["Demand"]
        /
        df["Product_HistoricalMean"]
        .replace(
            0,
            np.nan
        )
    )

    # --------------------------------------------------------
    # Fill Historical Missing Values
    # --------------------------------------------------------

    global_demand_mean = (
        df["Demand"].mean()
    )

    global_demand_max = (
        df["Demand"].max()
    )

    global_nonzero_rate = (
        (df["Demand"] > 0).mean()
    )

    df["Product_HistoricalMean"] = (
        df["Product_HistoricalMean"]
        .fillna(global_demand_mean)
    )

    df["Product_HistoricalMax"] = (
        df["Product_HistoricalMax"]
        .fillna(global_demand_max)
    )

    df["Product_HistoricalNonZeroRate"] = (
        df["Product_HistoricalNonZeroRate"]
        .fillna(global_nonzero_rate)
    )

    df["Demand_vs_ProductHistory"] = (
        df["Demand_vs_ProductHistory"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(1.0)
    )

    # --------------------------------------------------------
    # Fill Remaining Numeric Missing Values
    # --------------------------------------------------------

    numeric_columns = (
        df
        .select_dtypes(
            include=[np.number]
        )
        .columns
    )

    for column in numeric_columns:

        df[column] = (
            df[column]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
        )

    return df


# ============================================================
# PREDICT NEXT-DAY DEMAND
# ============================================================

def predict_demand(
    product_data,
    selected_date
):

    feature_data = create_features(
        product_data
    )

    selected_rows = feature_data[
        feature_data["Date"]
        ==
        pd.Timestamp(selected_date)
    ]

    if selected_rows.empty:
        return None, feature_data

    row = selected_rows.iloc[-1]

    if model_features is not None:

        features = list(
            model_features
        )

    else:

        features = V3_FEATURES

    missing_features = [
        feature
        for feature in features
        if feature not in row.index
    ]

    if missing_features:

        st.error(
            "Model feature mismatch.\n\n"
            f"Missing features: {missing_features}"
        )

        st.stop()

    X = pd.DataFrame(
        [
            [
                row[feature]
                for feature in features
            ]
        ],
        columns=features
    )

    prediction = model.predict(X)[0]

    prediction = max(
        0.0,
        float(prediction)
    )

    return (
        prediction,
        feature_data
    )


# ============================================================
# RL-STYLE PRICE OPTIMIZATION
# ============================================================

# Grid of candidate price changes evaluated by the optimizer.
# Defined once at module level so the header badge below can
# report the true candidate count instead of a hardcoded number.
PRICE_CHANGE_GRID = [
    -0.10,
    -0.075,
    -0.05,
    -0.025,
    0.0,
    0.025,
    0.05,
    0.075,
    0.10
]


def optimize_price(
    current_price,
    forecast_demand,
    elasticity
):

    price_changes = PRICE_CHANGE_GRID

    candidates = []

    base_revenue = (
        current_price
        *
        forecast_demand
    )

    for change in price_changes:

        candidate_price = (
            current_price
            *
            (1 + change)
        )

        # ----------------------------------------------------
        # Elasticity-Based Demand Response
        # ----------------------------------------------------

        demand_multiplier = (
            candidate_price
            /
            current_price
        ) ** elasticity

        expected_demand = (
            forecast_demand
            *
            demand_multiplier
        )

        expected_demand = max(
            0.0,
            float(expected_demand)
        )

        # ----------------------------------------------------
        # Expected Revenue
        # ----------------------------------------------------

        expected_revenue = (
            candidate_price
            *
            expected_demand
        )

        # ----------------------------------------------------
        # Price Movement Penalty
        # ----------------------------------------------------

        price_penalty = (
            abs(change)
            *
            base_revenue
            *
            0.10
        )

        # ----------------------------------------------------
        # Demand Loss Penalty
        # ----------------------------------------------------

        demand_penalty = (
            max(
                0,
                forecast_demand
                -
                expected_demand
            )
            *
            candidate_price
            *
            0.05
        )

        # ----------------------------------------------------
        # Reward
        # ----------------------------------------------------

        reward = (
            expected_revenue
            -
            price_penalty
            -
            demand_penalty
        )

        candidates.append(
            {
                "Price Change": change * 100,
                "Candidate Price": candidate_price,
                "Expected Demand": expected_demand,
                "Expected Revenue": expected_revenue,
                "Reward": reward
            }
        )

    candidate_df = pd.DataFrame(
        candidates
    )

    best_index = (
        candidate_df["Reward"]
        .idxmax()
    )

    best = (
        candidate_df
        .loc[best_index]
    )

    price_change = float(
        best["Price Change"]
    )

    if price_change > 0.01:

        action = "Increase Price"

    elif price_change < -0.01:

        action = "Decrease Price"

    else:

        action = "Keep Price"

    return (
        best,
        candidate_df,
        action
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
    <div class="dash-header-row">
        <div class="dash-heading">
            <div class="hero-eyebrow">Pricing Intelligence</div>
            <div class="hero-title">Dynamic Pricing Optimization Engine</div>
            <p class="hero-subtitle">
                Next-day demand forecasting and intelligent price
                recommendation using LightGBM, time-series signals,
                price elasticity and RL-style pricing optimization.
            </p>
        </div>
        <div class="dash-header-badges">
            <span class="dash-pill">📅 Next-Day Horizon</span>
            <span class="dash-pill">🧮 Adaptive Candidate Actions</span>
            <span class="dash-pill accent">🤖 AI Recommendation</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PRODUCT AND DATE SELECTION
# ============================================================

st.markdown(
    """
    <div class="section-label">
        <span class="icon">🔎</span> Product &amp; Analysis Date
    </div>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Product Lookup
# ------------------------------------------------------------

product_columns = ["StockCode"]

if "Description" in data.columns:
    product_columns.append("Description")


product_lookup = (
    data[product_columns]
    .dropna(
        subset=["StockCode"]
    )
    .copy()
)


product_lookup["StockCode"] = (
    product_lookup["StockCode"]
    .astype(str)
    .str.strip()
)


# ------------------------------------------------------------
# Product Description
# ------------------------------------------------------------

if "Description" in product_lookup.columns:

    product_lookup["Description"] = (
        product_lookup["Description"]
        .fillna("Product")
        .astype(str)
        .str.strip()
    )

    product_lookup.loc[
        product_lookup["Description"] == "",
        "Description"
    ] = "Product"

else:

    product_lookup["Description"] = "Product"


# ------------------------------------------------------------
# Remove Duplicate Products
# ------------------------------------------------------------

product_lookup = (
    product_lookup
    .drop_duplicates(
        subset=["StockCode"]
    )
    .sort_values(
        "StockCode"
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# Display Label
# ------------------------------------------------------------

product_lookup["Display"] = (
    product_lookup["StockCode"]
    +
    " — "
    +
    product_lookup["Description"]
)


# ------------------------------------------------------------
# Product Dropdown
# ------------------------------------------------------------

sel_col1, sel_col2 = st.columns(2)

with sel_col1:

    selected_display = st.selectbox(
        "Select Product",
        product_lookup["Display"].tolist(),
        index=0
    )


# ------------------------------------------------------------
# Get Actual StockCode
# ------------------------------------------------------------

selected_stock = product_lookup.loc[
    product_lookup["Display"]
    ==
    selected_display,
    "StockCode"
].iloc[0]


# ============================================================
# FILTER SELECTED PRODUCT
# ============================================================

product_data = data[
    data["StockCode"]
    ==
    selected_stock
].copy()


if product_data.empty:

    st.error(
        "No data found for the selected product."
    )

    st.stop()


# Sort by date
product_data = (
    product_data
    .sort_values("Date")
    .reset_index(drop=True)
)


# ============================================================
# DATE SELECTION
# ============================================================

available_dates = (
    product_data["Date"]
    .dt.date
    .drop_duplicates()
    .tolist()
)


if not available_dates:

    st.error(
        "No dates are available for the selected product."
    )

    st.stop()


with sel_col2:

    selected_date = st.selectbox(
        "Select Analysis Date",
        available_dates,
        index=len(available_dates) - 1,
        format_func=lambda x:
            x.strftime("%d %b %Y")
    )


# Convert to Timestamp
selected_date = pd.Timestamp(
    selected_date
)


# ============================================================
# DEMAND FORECAST
# ============================================================

forecast_demand, feature_data = (
    predict_demand(
        product_data,
        selected_date
    )
)


if forecast_demand is None:

    st.error(
        "Prediction could not be generated "
        "for the selected date."
    )

    st.stop()


# ============================================================
# SELECTED ROW
# ============================================================

selected_rows = feature_data[
    feature_data["Date"]
    ==
    pd.Timestamp(selected_date)
]


if selected_rows.empty:

    st.error(
        "Selected date data could not be found."
    )

    st.stop()


selected_row = (
    selected_rows.iloc[-1]
)


# ============================================================
# CURRENT PRICE
# ============================================================

current_price = float(
    selected_row[
        "AveragePrice_INR"
    ]
)


# ============================================================
# CURRENT DEMAND
# ============================================================

current_demand = float(
    selected_row[
        "Demand"
    ]
)


# ============================================================
# NEXT-DAY DATE
# ============================================================

next_day = (
    pd.Timestamp(selected_date)
    +
    pd.Timedelta(days=1)
)


# ============================================================
# ELASTICITY
# ============================================================

elasticity, confidence = (
    get_elasticity(
        selected_stock
    )
)


# ============================================================
# PRICE OPTIMIZATION
# ============================================================

best, candidate_df, action = (
    optimize_price(
        current_price,
        forecast_demand,
        elasticity
    )
)


# ============================================================
# RECOMMENDED VALUES
# ============================================================

recommended_price = float(
    best["Candidate Price"]
)

expected_demand = float(
    best["Expected Demand"]
)

expected_revenue = float(
    best["Expected Revenue"]
)

price_change_pct = float(
    best["Price Change"]
)


# ============================================================
# REVENUE CHANGE
# ============================================================

base_forecast_revenue = (
    current_price
    *
    forecast_demand
)


if base_forecast_revenue > 0:

    revenue_change_pct = (
        (
            expected_revenue
            -
            base_forecast_revenue
        )
        /
        base_forecast_revenue
        *
        100
    )

else:

    revenue_change_pct = 0.0


# ============================================================
# DEMAND CHANGE
# ============================================================

if forecast_demand > 0:

    demand_change_pct = (
        (
            expected_demand
            -
            forecast_demand
        )
        /
        forecast_demand
        *
        100
    )

else:

    demand_change_pct = 0.0


# ============================================================
# NEXT-DAY FORECAST
# ============================================================

st.markdown(
    '<div class="section-label" id="nav-forecast"><span class="icon">📈</span> Next-Day Demand Forecast</div>',
    unsafe_allow_html=True
)

st.caption(
    f"Prediction for **{next_day.strftime('%d %b %Y')}** "
    f"using information available on "
    f"**{selected_date.strftime('%d %b %Y')}**."
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Current Price",
        f"₹{current_price:,.2f}"
    )


with c2:

    st.metric(
        "Next-Day Demand",
        f"{forecast_demand:,.2f}"
    )


with c3:

    st.metric(
        "Recommended Price",
        f"₹{recommended_price:,.2f}",
        f"{price_change_pct:+.2f}%"
    )


with c4:

    st.metric(
        "Expected Demand",
        f"{expected_demand:,.2f}",
        f"{demand_change_pct:+.2f}%"
    )


st.divider()


# ============================================================
# AI PRICING RECOMMENDATION
# ============================================================

st.markdown(
    '<div class="section-label" id="nav-recommendation"><span class="icon">🤖</span> AI Pricing Recommendation</div>',
    unsafe_allow_html=True
)


if action == "Increase Price":
    reco_class = "increase"
    reco_icon = "▲"

elif action == "Decrease Price":
    reco_class = "decrease"
    reco_icon = "▼"

else:
    reco_class = "keep"
    reco_icon = "●"


st.markdown(
    f"""
    <div class="reco-card {reco_class}">
        <div>
            <div class="reco-badge">{reco_icon} {action}</div>
            <p class="reco-action">Recommended price: ₹{recommended_price:,.2f}</p>
        </div>
        <div class="reco-stats">
            <div>
                <div class="reco-stat-label">Price Change</div>
                <div class="reco-stat-value">{price_change_pct:+.2f}%</div>
            </div>
            <div>
                <div class="reco-stat-label">Revenue Impact</div>
                <div class="reco-stat-value">{revenue_change_pct:+.2f}%</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# EXPECTED BUSINESS IMPACT
# ============================================================

st.markdown('<div class="section-label"><span class="icon">💰</span> Expected Business Impact</div>', unsafe_allow_html=True)


b1, b2, b3, b4 = st.columns(4)


with b1:

    st.metric(
        "Expected Revenue",
        f"₹{expected_revenue:,.2f}"
    )


with b2:

    st.metric(
        "Revenue Change",
        f"{revenue_change_pct:+.2f}%"
    )


with b3:

    st.metric(
        "Price Elasticity",
        f"{elasticity:.3f}"
    )


with b4:

    st.metric(
        "Confidence",
        confidence
    )


# ============================================================
# PRICE OPTIMIZATION CHART
# ============================================================

st.markdown('<div class="section-label" id="nav-optimization"><span class="icon">⚙️</span> Price Optimization Analysis</div>', unsafe_allow_html=True)


fig_price = go.Figure()


fig_price.add_trace(
    go.Scatter(
        x=candidate_df[
            "Candidate Price"
        ],
        y=candidate_df[
            "Expected Revenue"
        ],
        mode="lines+markers",
        name="Expected Revenue",
        line=dict(color=PLOTLY_COLORWAY[0], width=2.5),
        marker=dict(color=PLOTLY_COLORWAY[0], size=7)
    )
)


fig_price.add_trace(
    go.Scatter(
        x=[recommended_price],
        y=[expected_revenue],
        mode="markers",
        marker=dict(
            size=14,
            symbol="diamond",
            color=PLOTLY_COLORWAY[1],
            line=dict(color=PLOTLY_PAPER_COLOR, width=1.5)
        ),
        name="Recommended Price"
    )
)


fig_price.update_layout(
    title="Expected Revenue at Different Price Levels",
    xaxis_title="Price (₹)",
    yaxis_title="Expected Revenue (₹)",
    template="plotly_dark",
    height=430,
    hovermode="x unified",
    font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT_COLOR, size=13),
    title_font=dict(size=15, color="#F5F7FA"),
    plot_bgcolor=PLOTLY_PAPER_COLOR,
    paper_bgcolor=PLOTLY_PAPER_COLOR,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=60, l=10, r=10, b=10)
)
fig_price.update_xaxes(gridcolor=PLOTLY_GRID_COLOR, zeroline=False)
fig_price.update_yaxes(gridcolor=PLOTLY_GRID_COLOR, zeroline=False)


st.plotly_chart(
    fig_price,
    use_container_width=True,
    theme=None
)


# ============================================================
# HISTORICAL DEMAND TREND
# ============================================================

st.markdown('<div class="section-label"><span class="icon">📊</span> Historical Demand Trend</div>', unsafe_allow_html=True)


history = feature_data[
    feature_data["Date"]
    <=
    pd.Timestamp(selected_date)
].tail(120)


fig_demand = go.Figure()


fig_demand.add_trace(
    go.Scatter(
        x=history["Date"],
        y=history["Demand"],
        mode="lines",
        name="Daily Demand",
        line=dict(color=PLOTLY_COLORWAY[0], width=2)
    )
)


if "Demand_RollingMean_7" in history.columns:

    fig_demand.add_trace(
        go.Scatter(
            x=history["Date"],
            y=history[
                "Demand_RollingMean_7"
            ],
            mode="lines",
            name="7-Day Rolling Mean",
            line=dict(
                dash="dot",
                color=PLOTLY_COLORWAY[1],
                width=2
            )
        )
    )


fig_demand.update_layout(
    title="Recent Product Demand",
    xaxis_title="Date",
    yaxis_title="Demand",
    template="plotly_dark",
    height=420,
    font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT_COLOR, size=13),
    title_font=dict(size=15, color="#F5F7FA"),
    plot_bgcolor=PLOTLY_PAPER_COLOR,
    paper_bgcolor=PLOTLY_PAPER_COLOR,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=60, l=10, r=10, b=10)
)
fig_demand.update_xaxes(gridcolor=PLOTLY_GRID_COLOR, zeroline=False)
fig_demand.update_yaxes(gridcolor=PLOTLY_GRID_COLOR, zeroline=False)


st.plotly_chart(
    fig_demand,
    use_container_width=True,
    theme=None
)


# ============================================================
# PRICE VS DEMAND
# ============================================================

st.markdown('<div class="section-label"><span class="icon">📉</span> Price vs Demand</div>', unsafe_allow_html=True)


fig_relationship = go.Figure()


fig_relationship.add_trace(
    go.Scatter(
        x=history[
            "AveragePrice_INR"
        ],
        y=history[
            "Demand"
        ],
        mode="markers",
        name="Historical Data",
        marker=dict(
            size=7,
            color=PLOTLY_COLORWAY[0],
            opacity=0.8,
            line=dict(color=PLOTLY_PAPER_COLOR, width=0.5)
        )
    )
)


fig_relationship.update_layout(
    title="Historical Price-Demand Relationship",
    xaxis_title="Average Price (₹)",
    yaxis_title="Demand",
    template="plotly_dark",
    height=420,
    font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT_COLOR, size=13),
    title_font=dict(size=15, color="#F5F7FA"),
    plot_bgcolor=PLOTLY_PAPER_COLOR,
    paper_bgcolor=PLOTLY_PAPER_COLOR,
    margin=dict(t=60, l=10, r=10, b=10)
)
fig_relationship.update_xaxes(gridcolor=PLOTLY_GRID_COLOR, zeroline=False)
fig_relationship.update_yaxes(gridcolor=PLOTLY_GRID_COLOR, zeroline=False)


st.plotly_chart(
    fig_relationship,
    use_container_width=True,
    theme=None
)


# ============================================================
# PRICING ACTION TABLE
# ============================================================

st.markdown('<div class="section-label" id="nav-actions"><span class="icon">📋</span> Evaluated Pricing Actions</div>', unsafe_allow_html=True)


action_table = candidate_df.copy()


action_table["Price Change"] = (
    action_table["Price Change"]
    .map(
        lambda x:
            f"{x:+.1f}%"
    )
)


action_table["Candidate Price"] = (
    action_table["Candidate Price"]
    .map(
        lambda x:
            f"₹{x:,.2f}"
    )
)


action_table["Expected Demand"] = (
    action_table["Expected Demand"]
    .map(
        lambda x:
            f"{x:,.2f}"
    )
)


action_table["Expected Revenue"] = (
    action_table["Expected Revenue"]
    .map(
        lambda x:
            f"₹{x:,.2f}"
    )
)


action_table["Reward"] = (
    action_table["Reward"]
    .map(
        lambda x:
            f"₹{x:,.2f}"
    )
)


action_table = action_table[
    [
        "Price Change",
        "Candidate Price",
        "Expected Demand",
        "Expected Revenue",
        "Reward"
    ]
]


st.markdown(
    '<p class="best-price-note">The highlighted row is the recommended pricing action.</p>',
    unsafe_allow_html=True
)

_table_rows_html = ""

for idx, row in action_table.iterrows():

    row_class = "reco-row" if idx == best.name else ""

    _table_rows_html += (
        f'<tr class="{row_class}">'
        f'<td>{row["Price Change"]}</td>'
        f'<td>{row["Candidate Price"]}</td>'
        f'<td>{row["Expected Demand"]}</td>'
        f'<td>{row["Expected Revenue"]}</td>'
        f'<td>{row["Reward"]}</td>'
        f'</tr>'
    )

st.markdown(
    f"""
    <div class="pricing-table-wrap">
        <table class="pricing-table">
            <thead>
                <tr>
                    <th>Price Change</th>
                    <th>Candidate Price</th>
                    <th>Expected Demand</th>
                    <th>Expected Revenue</th>
                    <th>Reward</th>
                </tr>
            </thead>
            <tbody>
                {_table_rows_html}
            </tbody>
        </table>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "ℹ️ Model Information"
):

    feature_count = (
        len(model_features)
        if model_features is not None
        else len(V3_FEATURES)
    )

    st.markdown(
        f"""
        <div class="info-grid">
            <div class="info-item">
                <div class="info-label">Model</div>
                <div class="info-value">LightGBM V3 High-Demand</div>
            </div>
            <div class="info-item">
                <div class="info-label">Model File</div>
                <div class="info-value"><code>lightgbm_v3_high_demand.pkl</code></div>
            </div>
            <div class="info-item">
                <div class="info-label">Model Features</div>
                <div class="info-value">{feature_count}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Currency</div>
                <div class="info-value">INR (₹)</div>
            </div>
            <div class="info-item">
                <div class="info-label">Forecast Horizon</div>
                <div class="info-value">Next Day</div>
            </div>
            <div class="info-item">
                <div class="info-label">Pricing Actions</div>
                <div class="info-value">9</div>
            </div>
            <div class="info-item">
                <div class="info-label">Optimization Method</div>
                <div class="info-value">RL-Style Reward Evaluation</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "⚡ Dynamic Pricing Optimization Engine — "
    "LightGBM V3 High-Demand · Time-Series Signals · "
    "Price Elasticity · RL-Style Pricing Optimization"
)
