"""
AgroVox - Centralized design system for the Streamlit UI.

Streamlit has no JSX-style component model, so "reusable components" here
means: a single source of truth for colors/typography/icons (this module),
plus small Python functions that render HTML snippets consistently across
every page of app.py. Nothing in this module touches models, datasets, the
database, or any NLP logic - it is presentation only.
"""
import re


def flatten_html(html: str) -> str:
    """Collapse newline+indentation between tags (but not a plain inline
    single space, which some snippets rely on for visual gap). Streamlit's
    markdown parser can misinterpret indented multi-line HTML - especially
    once a blank-ish or oddly-indented line appears - as the start of a
    Markdown code block, leaking literal tag text into the rendered page.
    Every raw HTML block passed to st.markdown(..., unsafe_allow_html=True)
    anywhere in this app should be wrapped in this before rendering."""
    return re.sub(r">\s*\n\s*<", "><", html.strip())


_flatten = flatten_html  # internal alias used by the helpers below

# ---------------------------------------------------------------- PALETTE --
LIGHT_COLORS = {
    "primary_dark": "#163D27",
    "secondary": "#2F6041",
    "accent": "#6FAE45",
    "accent_light": "#EAF4E3",
    "bg": "#F7F4EA",
    "surface": "#FFFFFF",
    "text": "#13231A",
    "text_muted": "#6B776F",
    "border": "#DDE5DA",
    "warning": "#F4B942",
    "danger": "#E96A5B",
    "info": "#4C8BF5",
    "ai_purple": "#8067C9",
    "shadow": "0 1px 2px rgba(19,35,26,0.04), 0 6px 20px rgba(19,35,26,0.06)",
    "shadow_hover": "0 4px 10px rgba(19,35,26,0.08), 0 14px 32px rgba(19,35,26,0.10)",
}

DARK_COLORS = {
    "primary_dark": "#0E2818",
    "secondary": "#2F6041",
    "accent": "#7FC257",
    "accent_light": "#1B3324",
    "bg": "#0F1912",
    "surface": "#16241A",
    "text": "#EAF1E7",
    "text_muted": "#93A398",
    "border": "#28392E",
    "warning": "#F4B942",
    "danger": "#F08A7E",
    "info": "#6FA0F7",
    "ai_purple": "#9C87DE",
    "shadow": "0 1px 2px rgba(0,0,0,0.25), 0 6px 20px rgba(0,0,0,0.35)",
    "shadow_hover": "0 4px 10px rgba(0,0,0,0.30), 0 14px 32px rgba(0,0,0,0.40)",
}


def get_colors(theme: str) -> dict:
    return DARK_COLORS if theme == "dark" else LIGHT_COLORS


# ------------------------------------------------------------------ ICONS --
# Minimal line icons, Lucide-inspired (viewBox 0 0 24 24, stroke=currentColor).
_ICON_BODIES = {
    "home": '<path d="M3 11.5 12 4l9 7.5"/><path d="M5 10v9a1 1 0 0 0 1 1h4v-6h4v6h4a1 1 0 0 0 1-1v-9"/>',
    "bot": '<rect x="4" y="8" width="16" height="11" rx="3"/><path d="M12 3v5"/><circle cx="9" cy="13.5" r="1.2"/><circle cx="15" cy="13.5" r="1.2"/><path d="M2 13h2M20 13h2"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>',
    "book-open": '<path d="M3 5.5C4.5 4.5 7 4 12 5.5V19c-5-1.5-7.5-1-9-0.2V5.5Z"/><path d="M21 5.5C19.5 4.5 17 4 12 5.5V19c5-1.5 7.5-1 9-0.2V5.5Z"/>',
    "flask": '<path d="M9 3h6"/><path d="M10 3v6.2L4.8 18a2 2 0 0 0 1.7 3h11a2 2 0 0 0 1.7-3L14 9.2V3"/><path d="M7.5 15h9"/>',
    "bar-chart": '<path d="M4 20V10"/><path d="M12 20V4"/><path d="M20 20v-6"/><path d="M2 20h20"/>',
    "user": '<circle cx="12" cy="8" r="3.6"/><path d="M4.5 20c1.4-3.6 4.2-5.5 7.5-5.5s6.1 1.9 7.5 5.5"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M19.4 13.5a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 1 1-2.9 2.9l-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6V20a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1a2 2 0 1 1-2.9-2.9l.1-.1a1.7 1.7 0 0 0 .3-1.9 1.7 1.7 0 0 0-1.6-1H4a2 2 0 1 1 0-4h.1A1.7 1.7 0 0 0 5.7 9.5a1.7 1.7 0 0 0-.3-1.9l-.1-.1a2 2 0 1 1 2.9-2.9l.1.1a1.7 1.7 0 0 0 1.9.3H10a1.7 1.7 0 0 0 1-1.6V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1a2 2 0 1 1 2.9 2.9l-.1.1a1.7 1.7 0 0 0-.3 1.9V9c.4.3 1 .5 1.6.5H20a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.6 1Z"/>',
    "mic": '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0"/><path d="M12 18v3"/><path d="M9 21h6"/>',
    "volume": '<path d="M4 9v6h4l5 4V5L8 9H4Z"/><path d="M17 8.5a5 5 0 0 1 0 7"/>',
    "message": '<path d="M4 4h16v12H8l-4 4V4Z"/>',
    "trend-down": '<path d="m3 7 7 7 4-4 7 7"/><path d="M15 17h6v-6"/>',
    "trend-up": '<path d="m3 17 7-7 4 4 7-7"/><path d="M15 7h6v6"/>',
    "check-circle": '<circle cx="12" cy="12" r="9"/><path d="m8.5 12.5 2.5 2.5 5-5.5"/>',
    "alert": '<path d="M12 3 2 20h20L12 3Z"/><path d="M12 10v4"/><circle cx="12" cy="17" r="0.6" fill="currentColor" stroke="none"/>',
    "sun": '<circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v2.3M12 19.2v2.3M4.4 4.4l1.6 1.6M18 18l1.6 1.6M2.5 12h2.3M19.2 12h2.3M4.4 19.6 6 18M18 6l1.6-1.6"/>',
    "moon": '<path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5Z"/>',
    "bell": '<path d="M6 9a6 6 0 0 1 12 0c0 4 1.5 5.5 1.5 5.5H4.5S6 13 6 9Z"/><path d="M10 18a2 2 0 0 0 4 0"/>',
    "copy": '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/>',
    "send": '<path d="m3 11 18-8-8 18-2.5-7.5L3 11Z"/>',
    "trash": '<path d="M4 7h16"/><path d="M9 7V4h6v3"/><path d="M6 7l1 13h10l1-13"/>',
    "thumbs-up": '<path d="M7 11v9H4v-9h3Z"/><path d="M7 11l3.5-7c1.2 0 2 1 1.7 2.2L11.5 9H18a1.8 1.8 0 0 1 1.7 2.5l-2.3 6.3A2 2 0 0 1 15.5 20H7"/>',
    "thumbs-down": '<path d="M17 13V4h3v9h-3Z"/><path d="M17 13l-3.5 7c-1.2 0-2-1-1.7-2.2l.7-2.8H6a1.8 1.8 0 0 1-1.7-2.5l2.3-6.3A2 2 0 0 1 8.5 4H17"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6"/><circle cx="12" cy="7.5" r="0.6" fill="currentColor" stroke="none"/>',
    "lightbulb": '<path d="M9 18h6"/><path d="M10 21h4"/><path d="M12 3a6 6 0 0 0-3.6 10.8c.6.5 1 1.2 1.1 2.2h5c.1-1 .5-1.7 1.1-2.2A6 6 0 0 0 12 3Z"/>',
    "download": '<path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M4 19h16"/>',
    "filter": '<path d="M4 5h16l-6 8v6l-4-2v-4L4 5Z"/>',
    "chevron-down": '<path d="m6 9 6 6 6-6"/>',
    "leaf": '<path d="M4 20c8-1 15-8 16-16C12 5 5 12 4 20Z"/><path d="M4 20 15 9"/>',
    "sprout": '<path d="M7 20h10"/><path d="M12 20v-8"/><path d="M12 12C7 12 5 9 5 5c4 0 7 2 7 7Z"/><path d="M12 9c1-3 3-4 6-4 0 3-1 5-4 6"/>',
    "droplet": '<path d="M12 3s6 7 6 11.5A6 6 0 0 1 6 14.5C6 10 12 3 12 3Z"/>',
    "cloud-sun": '<circle cx="8" cy="7" r="2.8"/><path d="M15 19H7a4 4 0 0 1-.4-8 5 5 0 0 1 9.6 1.6A3.6 3.6 0 0 1 15 19Z"/>',
    "map-pin": '<path d="M12 21s7-7.2 7-12a7 7 0 0 0-14 0c0 4.8 7 12 7 12Z"/><circle cx="12" cy="9" r="2.4"/>',
    "sparkles": '<path d="M12 3v4M12 17v4M3 12h4M17 12h4"/><path d="m6 6 2 2M16 16l2 2M6 18l2-2M16 8l2-2"/>',
    "waveform": '<path d="M2 12h2v3H2zM6 8h2v11H6zM10 4h2v19h-2zM14 8h2v11h-2zM18 12h2v3h-2z"/>',
    "database": '<ellipse cx="12" cy="5.5" rx="8" ry="2.7"/><path d="M4 5.5V18c0 1.5 3.6 2.7 8 2.7s8-1.2 8-2.7V5.5"/><path d="M4 12c0 1.5 3.6 2.7 8 2.7s8-1.2 8-2.7"/>',
    "x": '<path d="m6 6 12 12M18 6 6 18"/>',
    "bug": '<rect x="8" y="7" width="8" height="11" rx="4"/><path d="M8 11H4M20 11h-4M9 5l-1.5-2M15 5l1.5-2M9 18l-2 2M15 18l2 2M8 14H5M19 14h-3"/>',
    "soil-layers": '<path d="M3 6h18M3 12h18M3 18h18"/><path d="M6 6v12M12 6v12M18 6v12" stroke-opacity="0.4"/>',
}


def icon(name: str, size: int = 18, color: str = "currentColor", stroke_width: float = 2) -> str:
    body = _ICON_BODIES.get(name, _ICON_BODIES["info"])
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" '
            f'stroke-linejoin="round" style="flex-shrink:0;">{body}</svg>')


# -------------------------------------------------------------------- CSS --
def build_css(colors: dict) -> str:
    c = colors
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
    --av-primary-dark: {c['primary_dark']};
    --av-secondary: {c['secondary']};
    --av-accent: {c['accent']};
    --av-accent-light: {c['accent_light']};
    --av-bg: {c['bg']};
    --av-surface: {c['surface']};
    --av-text: {c['text']};
    --av-text-muted: {c['text_muted']};
    --av-border: {c['border']};
    --av-warning: {c['warning']};
    --av-danger: {c['danger']};
    --av-info: {c['info']};
    --av-purple: {c['ai_purple']};
    --av-shadow: {c['shadow']};
    --av-shadow-hover: {c['shadow_hover']};
    --av-radius: 16px;
}}

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: var(--av-bg); }}
h1, h2, h3 {{ font-family: 'Playfair Display', serif !important; color: var(--av-text) !important; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.5rem !important; margin-top: 0.2rem !important; }}
h3 {{ font-size: 1.15rem !important; }}
p, span, div, label {{ color: var(--av-text); }}
code, .stCodeBlock, pre {{ font-family: 'JetBrains Mono', monospace !important; }}

[data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
[data-testid="stAppViewBlockContainer"] {{ padding-top: 1.4rem; }}
[data-testid="stAppViewContainer"] {{ background: var(--av-bg); }}

/* ---------------- Sidebar ---------------- */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {c['primary_dark']} 0%, {c['primary_dark']} 100%);
    border-right: none;
    min-width: 272px !important;
}}
[data-testid="stSidebar"] * {{ color: #EAF1E7 !important; }}
[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.10); }}

/* nav list styled from st.radio - hide native circle, style like nav items */
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 2px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
    padding: 9px 12px; border-radius: 10px; width: 100%;
    transition: background 0.15s ease; cursor: pointer; position: relative;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: rgba(255,255,255,0.06); }}
[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"],
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
    background: rgba(111,174,69,0.18);
}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked)::before {{
    content: ""; position: absolute; left: -2px; top: 20%; bottom: 20%; width: 3px;
    background: var(--av-accent); border-radius: 3px;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{
    color: #FFFFFF !important; font-weight: 700 !important;
}}
[data-testid="stSidebar"] [role="radiogroup"] input {{ display: none; }}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{ display: none; }}
[data-testid="stSidebar"] .stSelectbox label, [data-testid="stSidebar"] .stTextInput label {{
    color: #9FB397 !important; font-size: 0.78rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.04em;
}}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] input {{
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 9px !important;
}}

/* ---------------- Top header ---------------- */
.av-topbar {{ display:flex; align-items:center; justify-content:space-between; padding:0.2rem 0 1rem 0;
    border-bottom:1px solid var(--av-border); margin-bottom:1.2rem; gap:16px; }}
.av-search {{ flex:1; max-width:520px; background:var(--av-surface); border:1px solid var(--av-border);
    border-radius:999px; padding:9px 18px; display:flex; align-items:center; gap:10px; font-size:0.86rem;
    color:var(--av-text-muted); }}
.av-icon-btn {{ width:36px; height:36px; border-radius:50%; background:var(--av-surface);
    border:1px solid var(--av-border); display:flex; align-items:center; justify-content:center;
    color:var(--av-text-muted); flex-shrink:0; }}
.av-user-chip {{ display:flex; align-items:center; gap:8px; padding:4px 12px 4px 6px; border-radius:999px;
    background:var(--av-surface); border:1px solid var(--av-border); }}
.av-user-avatar {{ width:28px; height:28px; border-radius:50%; background:var(--av-primary-dark); color:#fff;
    display:flex; align-items:center; justify-content:center; font-weight:700; font-size:0.8rem; }}

/* ---------------- Hero ---------------- */
.av-hero {{
    border-radius: 22px; position: relative; overflow: hidden; margin-bottom: 1.6rem;
    background: linear-gradient(120deg, {c['accent_light']} 0%, {c['bg']} 55%, {c['accent_light']} 100%);
    border: 1px solid var(--av-border);
    box-shadow: var(--av-shadow-hover);
    padding: 2.2rem 2.6rem;
}}
.av-hero-inner {{ display:flex; align-items:center; justify-content:space-between; gap:24px; position:relative; z-index:1; }}
.av-hero h1 {{ font-size: 2.1rem !important; margin: 0 0 0.3rem !important; }}
.av-hero p.av-hero-sub {{ color: var(--av-text-muted); font-size: 1rem; font-weight: 600; margin:0; }}
.av-hero .av-tagline {{ font-family:'Playfair Display',serif; font-style:italic; font-size:0.92rem;
    color: var(--av-secondary); text-align:right; line-height:1.4; white-space:nowrap; }}
.av-hero-badge-row {{ display:flex; gap:8px; margin-bottom:0.7rem; flex-wrap:wrap; }}
.av-badge {{ display:inline-block; background: var(--av-accent-light); border:1px solid var(--av-border);
    color: var(--av-secondary); padding:0.26rem 0.7rem; border-radius:999px; font-size:0.74rem; font-weight:600; }}
.av-hero-art {{ position:absolute; right:0; bottom:0; opacity:0.9; pointer-events:none; }}

/* ---------------- Page header (non-hero pages) ---------------- */
.av-page-header {{ display:flex; align-items:center; gap:14px; margin-bottom:1.2rem; }}
.av-page-header .av-icon-tile {{ width:44px; height:44px; background:var(--av-accent-light);
    color:var(--av-secondary); }}
.av-page-header h2 {{ margin:0 !important; }}
.av-page-header p {{ margin:0; color:var(--av-text-muted); font-size:0.9rem; }}

/* ---------------- Cards ---------------- */
.av-card {{
    background: var(--av-surface); border: 1px solid var(--av-border); border-radius: var(--av-radius);
    padding: 1.15rem 1.35rem; margin-bottom: 0.9rem; box-shadow: var(--av-shadow);
    transition: box-shadow 0.18s ease, transform 0.18s ease;
}}
.av-card:hover {{ box-shadow: var(--av-shadow-hover); transform: translateY(-1px); }}
.av-icon-tile {{
    width: 38px; height: 38px; border-radius: 11px; display: inline-flex; align-items: center;
    justify-content: center; margin-bottom: 0.5rem;
}}
.av-metric-num {{ font-family:'Playfair Display',serif; font-size:1.9rem; color:var(--av-text); font-weight:700; line-height:1.05; animation: av-fade-in 0.4s ease; }}
.av-metric-label {{ color:var(--av-text-muted); font-size:0.74rem; text-transform:uppercase;
    letter-spacing:0.06em; font-weight:700; margin-top:0.15rem; }}
.av-metric-sub {{ font-size:0.78rem; color:var(--av-text-muted); margin-top:0.35rem; }}
@keyframes av-fade-in {{ from {{ opacity:0; transform:translateY(3px); }} to {{ opacity:1; transform:translateY(0); }} }}

.av-concept-card {{ background:var(--av-surface); border:1px solid var(--av-border); border-radius:var(--av-radius);
    padding:1.1rem 1.3rem; height:100%; box-shadow:var(--av-shadow); transition:transform 0.18s ease, box-shadow 0.18s ease;
    border-top:3px solid var(--av-accent); }}
.av-concept-card:hover {{ transform:translateY(-3px); box-shadow:var(--av-shadow-hover); }}
.av-concept-card b {{ font-family:'Playfair Display',serif; font-size:1.02rem; color:var(--av-text); }}

/* ---------------- Chips / entities / badges ---------------- */
.av-chip {{ display:inline-block; background:var(--av-bg); border:1px solid var(--av-border); color:var(--av-text);
    padding:0.18rem 0.65rem; border-radius:999px; font-size:0.78rem; margin:0.12rem 0.28rem 0.12rem 0; font-weight:500; }}
.av-entity-CROP {{ background:#E7EFD9; border-color:#B9CE94; color:#1F3313; }}
.av-entity-DISEASE {{ background:#FBE4E0; border-color:#E3A99C; color:#5C231A; }}
.av-entity-SYMPTOM {{ background:#FBE9CB; border-color:#EFC98C; color:#4A340C; }}
.av-entity-PEST {{ background:#EBE0F7; border-color:#C6ACE6; color:#3B215C; }}
.av-entity-FERTILIZER {{ background:#DCEAFB; border-color:#A6CBDA; color:#173A55; }}
.av-entity-LOCATION {{ background:#FBF0D9; border-color:#DFC488; color:#4A3611; }}
.av-entity-SOIL_TYPE {{ background:#E9E2D3; border-color:#C4B294; color:#3B3117; }}
.av-entity-FARMING_ACTIVITY {{ background:#DCEAE3; border-color:#A9CFBC; color:#173B29; }}
.av-entity-WEATHER_CONDITION {{ background:#DCE7FB; border-color:#A9C1DE; color:#152E56; }}

.av-status-badge {{ display:inline-flex; align-items:center; gap:4px; font-size:0.74rem; font-weight:700;
    padding:3px 10px; border-radius:999px; }}
.av-status-exact {{ background:#DCEAFB; color:#2563EB; }}
.av-status-good {{ background:#E1F3E1; color:#1E8E3E; }}
.av-status-review {{ background:#FDF0DC; color:#B4740E; }}
.av-status-poor {{ background:#FBE3E0; color:#C23B2E; }}
.av-status-success {{ background:#E1F3E1; color:#1E8E3E; }}
.av-status-danger {{ background:#FBE3E0; color:#C23B2E; }}
.av-status-info {{ background:#E4ECFC; color:#3161D1; }}
.av-status-purple {{ background:#EFEAFB; color:#6B4FC2; }}

/* ---------------- Chat ---------------- */
.av-chat-row {{ display:flex; gap:0.7rem; margin:0.9rem 0; align-items:flex-start; }}
.av-chat-row.user {{ flex-direction:row-reverse; }}
.av-avatar {{ width:36px; height:36px; border-radius:50%; flex-shrink:0; display:flex; align-items:center;
    justify-content:center; box-shadow:0 2px 6px rgba(0,0,0,0.12); }}
.av-avatar.user {{ background:var(--av-accent-light); color:var(--av-secondary); }}
.av-avatar.bot {{ background:linear-gradient(135deg, var(--av-accent), var(--av-secondary)); color:#fff; }}
.av-bubble {{ max-width:78%; padding:0.75rem 1.05rem; border-radius:16px; font-size:0.95rem; line-height:1.5;
    box-shadow:var(--av-shadow); }}
.av-bubble.user {{ background:var(--av-primary-dark); color:#F5F8F2; border-bottom-right-radius:4px; }}
.av-bubble.bot {{ background:var(--av-surface); border:1px solid var(--av-border); border-left:4px solid var(--av-accent);
    border-bottom-left-radius:4px; color:var(--av-text); }}
.av-meta-row {{ display:flex; gap:0.5rem; flex-wrap:wrap; margin:0.35rem 0 0.2rem 46px; }}
.av-meta-pill {{ font-size:0.72rem; background:var(--av-bg); border:1px solid var(--av-border); border-radius:999px;
    padding:0.12rem 0.55rem; color:var(--av-text-muted); font-weight:500; display:inline-flex; align-items:center; gap:4px; }}
.av-confidence-track {{ width:56px; height:5px; border-radius:3px; background:var(--av-border); display:inline-block;
    overflow:hidden; vertical-align:middle; margin-left:4px; }}
.av-confidence-fill {{ height:100%; background:linear-gradient(90deg, var(--av-accent), var(--av-secondary)); }}
.av-disclaimer {{ font-size:0.76rem; color:var(--av-text-muted); border-top:1px dashed var(--av-border);
    padding-top:0.55rem; margin:0.6rem 0 0.2rem 46px; }}

.av-suggested-chip {{ display:inline-block; background:var(--av-surface); border:1px solid var(--av-border);
    color:var(--av-text); padding:0.4rem 0.85rem; border-radius:999px; font-size:0.82rem; font-weight:500;
    margin:0.2rem 0.35rem 0.2rem 0; box-shadow:var(--av-shadow); }}

/* ---------------- Pipeline flow diagram ---------------- */
.av-flow {{ display:flex; flex-wrap:wrap; align-items:center; gap:0; }}
.av-flow-step {{ background:var(--av-surface); border:1px solid var(--av-border); border-radius:12px;
    padding:0.55rem 0.9rem; font-size:0.82rem; font-weight:600; color:var(--av-text); box-shadow:var(--av-shadow);
    white-space:nowrap; }}
.av-flow-arrow {{ color:var(--av-accent); font-size:1.1rem; margin:0 0.35rem; }}

/* ---------------- Knowledge Explorer cards ---------------- */
.av-kb-card {{ background:var(--av-surface); border:1px solid var(--av-border); border-radius:var(--av-radius);
    padding:1.1rem; text-align:center; box-shadow:var(--av-shadow); transition:transform 0.15s ease, box-shadow 0.15s ease;
    cursor:pointer; height:100%; }}
.av-kb-card:hover {{ transform:translateY(-3px); box-shadow:var(--av-shadow-hover); }}
.av-kb-card.active {{ border-color:var(--av-accent); background:var(--av-accent-light); }}
.av-kb-card .av-kb-icon {{ width:46px; height:46px; border-radius:13px; background:var(--av-accent-light);
    color:var(--av-secondary); display:flex; align-items:center; justify-content:center; margin:0 auto 0.6rem; }}
.av-kb-card b {{ font-family:'Playfair Display',serif; font-size:1rem; }}
.av-kb-card .av-kb-count {{ font-size:0.76rem; color:var(--av-text-muted); margin-top:2px; }}

/* ---------------- Tabs ---------------- */
.stTabs [data-baseweb="tab-list"] {{ gap:4px; background:var(--av-surface); padding:5px; border-radius:12px;
    border:1px solid var(--av-border); }}
.stTabs [data-baseweb="tab"] {{ border-radius:8px; padding:8px 16px; font-weight:600; color:var(--av-text-muted); }}
.stTabs [aria-selected="true"] {{ background:var(--av-primary-dark) !important; color:#FFFFFF !important;
    box-shadow: inset 0 -2px 0 var(--av-accent); }}

/* ---------------- Buttons & inputs ---------------- */
.stButton>button {{ border-radius:9px; border:1px solid var(--av-border); font-weight:600;
    background:var(--av-surface); color:var(--av-text);
    transition:transform 0.12s ease, box-shadow 0.12s ease, background 0.12s ease; }}
.stButton>button:hover {{ transform:translateY(-1px); box-shadow:0 3px 10px rgba(22,61,39,0.18);
    border-color:var(--av-secondary); }}
.stButton>button[kind="primary"] {{ background:linear-gradient(135deg, var(--av-secondary), var(--av-primary-dark));
    border-color:var(--av-secondary); color:#FFFFFF; }}
.stButton>button p {{ color:inherit !important; }}
[data-testid="stForm"] {{ border:1px solid var(--av-border); border-radius:14px; padding:1rem; background:var(--av-surface); }}
[data-testid="stDataFrame"] {{ border-radius:12px; overflow:hidden; border:1px solid var(--av-border); }}

/* ---------------- Insight card ---------------- */
.av-insight-card {{ background:var(--av-accent-light); border:1px solid var(--av-border); border-radius:14px;
    padding:1.1rem 1.4rem; margin-top:1.2rem; display:flex; justify-content:space-between; align-items:center;
    gap:16px; flex-wrap:wrap; }}
.av-insight-card .av-insight-icon {{ background:var(--av-surface); color:var(--av-secondary); width:40px; height:40px;
    border-radius:50%; display:flex; align-items:center; justify-content:center; flex-shrink:0; }}
.av-insight-card b {{ font-family:'Playfair Display',serif; font-size:0.98rem; }}
.av-insight-card .av-quote {{ font-family:'Playfair Display',serif; font-style:italic; font-size:0.86rem;
    color:var(--av-secondary); white-space:nowrap; }}

/* ---------------- Info card ---------------- */
.av-info-card {{ background:var(--av-surface); border:1px solid var(--av-border); border-radius:14px;
    padding:1.1rem 1.4rem; margin:1rem 0; display:flex; gap:14px; align-items:flex-start; }}
.av-info-card .av-info-icon {{ background:#E4ECFC; color:var(--av-info); width:36px; height:36px; border-radius:50%;
    display:flex; align-items:center; justify-content:center; flex-shrink:0; }}
.av-info-card b {{ font-family:'Playfair Display',serif; font-size:1.02rem; }}

/* ---------------- Misc ---------------- */
.av-section-label {{ text-transform:uppercase; letter-spacing:0.08em; font-size:0.76rem; font-weight:700;
    color:var(--av-secondary); margin-bottom:0.2rem; }}
.av-divider {{ height:1px; background:linear-gradient(90deg, transparent, var(--av-border), transparent);
    margin:1.1rem 0; border:none; }}
.av-etl-stage {{ display:flex; align-items:center; gap:0.8rem; padding:0.55rem 0; border-bottom:1px dashed var(--av-border); }}
.av-etl-stage:last-child {{ border-bottom:none; }}
.av-etl-bar-bg {{ flex:1; height:10px; background:var(--av-bg); border-radius:6px; overflow:hidden; }}
.av-etl-bar-fill {{ height:100%; border-radius:6px; }}

/* ---------------- Responsive ---------------- */
@media (max-width: 640px) {{
    .av-hero-inner {{ flex-direction:column; align-items:flex-start; gap:12px; }}
    .av-hero .av-tagline {{ text-align:left; }}
    .av-hero h1 {{ font-size:1.6rem !important; }}
}}
</style>
"""


# --------------------------------------------------------------- HELPERS --
def metric_card(icon_name: str, label: str, value: str, sublabel: str = None,
                 badge: str = None, badge_kind: str = "success", icon_bg: str = None) -> str:
    bg_style = f"background:{icon_bg};" if icon_bg else ""
    badge_html = (f'<div style="margin-top:8px;"><span class="av-status-badge av-status-{badge_kind}">'
                  f'{badge}</span></div>') if badge else ""
    sub_html = f'<div class="av-metric-sub">{sublabel}</div>' if sublabel else ""
    return _flatten(f"""<div class="av-card">
        <div style="display:flex; align-items:flex-start; gap:12px;">
            <div class="av-icon-tile" style="{bg_style}">{icon(icon_name, 20)}</div>
            <div>
                <div class="av-metric-label">{label}</div>
                <div class="av-metric-num" style="font-size:1.7rem;">{value}</div>
            </div>
        </div>
        {sub_html}{badge_html}
    </div>""")


def status_badge(text: str, kind: str = "info") -> str:
    return f'<span class="av-status-badge av-status-{kind}">{text}</span>'


def info_card(title: str, body: str, icon_name: str = "info") -> str:
    return _flatten(f"""<div class="av-info-card">
        <div class="av-info-icon">{icon(icon_name, 18)}</div>
        <div><b>{title}</b><p style="font-size:0.86rem; color:var(--av-text-muted); margin:0.25rem 0 0; line-height:1.45;">{body}</p></div>
    </div>""")


def insight_card(title: str, body: str, quote: str = None) -> str:
    quote_html = f'<div class="av-quote">{quote}</div>' if quote else ""
    return _flatten(f"""<div class="av-insight-card">
        <div style="display:flex; align-items:center; gap:14px;">
            <div class="av-insight-icon">{icon("lightbulb", 20)}</div>
            <div><b>{title}</b><p style="font-size:0.84rem; color:var(--av-text-muted); margin:0.15rem 0 0; line-height:1.4;">{body}</p></div>
        </div>
        {quote_html}
    </div>""")


def page_header(icon_name: str, title: str, subtitle: str = "") -> str:
    return _flatten(f"""<div class="av-page-header">
        <div class="av-icon-tile">{icon(icon_name, 22)}</div>
        <div><h2>{title}</h2><p>{subtitle}</p></div>
    </div>""")
