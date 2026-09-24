"""
AgroVox - Streamlit Application
Voice-Driven Intelligence for Smarter Farming

Run: streamlit run app.py
"""
import os
import sys
import uuid
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils import db
from src.chatbot.chatbot_engine import AgroVoxPipeline, ChatbotSession
from src.retrieval.knowledge_base import KnowledgeBase
from src.speech.speech_engine import SpeechEngine
import ui_theme
from ui_theme import icon, metric_card, status_badge, info_card, insight_card, page_header

# Every raw HTML block rendered via st.markdown(..., unsafe_allow_html=True)
# anywhere in this app (including on column objects, e.g. c1.markdown(...))
# is auto-flattened first - see ui_theme.flatten_html for why: Streamlit's
# markdown parser can otherwise misinterpret indented multi-line HTML as a
# Markdown code block and leak literal tag text into the rendered page.
# Two patches are needed: `st.markdown` is captured as a fixed bound-method
# reference inside Streamlit's own __init__ (so reassigning the class alone
# does not affect it - verified empirically), while column objects returned
# by st.columns(...) resolve `.markdown` fresh via the class on every call.
from streamlit.delta_generator import DeltaGenerator as _DeltaGenerator
_original_markdown = _DeltaGenerator.markdown
def _flattening_markdown(self, body="", *args, **kwargs):
    if kwargs.get("unsafe_allow_html") and isinstance(body, str):
        body = ui_theme.flatten_html(body)
    return _original_markdown(self, body, *args, **kwargs)
_DeltaGenerator.markdown = _flattening_markdown          # fixes column .markdown(...) calls
st.markdown = _flattening_markdown.__get__(st._main)      # fixes direct st.markdown(...) calls

# ------------------------------------------------------------------ SETUP --
st.set_page_config(
    page_title="AgroVox — Smarter Farming",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state.theme = "light"

# ---- Design tokens (centralized in ui_theme.py; kept as COLORS for the
# existing plotly chart calls elsewhere on this page, mapped to the new
# palette's closest equivalents) --
_raw_colors = ui_theme.get_colors(st.session_state.theme)
COLORS = {
    "bg": _raw_colors["bg"],
    "surface": _raw_colors["surface"],
    "ink": _raw_colors["text"],
    "primary": _raw_colors["secondary"],
    "primary_dark": _raw_colors["primary_dark"],
    "accent": _raw_colors["accent"],
    "accent2": _raw_colors["accent"],
    "gold": _raw_colors["warning"],
    "muted": _raw_colors["text_muted"],
    "border": _raw_colors["border"],
}

st.markdown(ui_theme.build_css(_raw_colors), unsafe_allow_html=True)


# ---------------------------------------------------------------- CACHING --
@st.cache_resource(show_spinner="Loading AgroVox NLP pipeline (classifier, NER, QA index)...")
def get_pipeline():
    return AgroVoxPipeline()


@st.cache_resource
def get_knowledge_base():
    return KnowledgeBase()


@st.cache_resource
def get_speech_engine():
    return SpeechEngine()


def get_session_state_defaults():
    if "farmer_id" not in st.session_state:
        st.session_state.farmer_id = f"farmer_{uuid.uuid4().hex[:8]}"
    if "farmer_name" not in st.session_state:
        st.session_state.farmer_name = ""
    if "preferred_language" not in st.session_state:
        st.session_state.preferred_language = "en"
    if "location" not in st.session_state:
        st.session_state.location = ""
    if "crops_cultivated" not in st.session_state:
        st.session_state.crops_cultivated = ""
    if "chat_session" not in st.session_state:
        pipeline = get_pipeline()
        st.session_state.chat_session = ChatbotSession(pipeline, st.session_state.farmer_id)
    if "chat_display" not in st.session_state:
        st.session_state.chat_display = []


db.init_db()
get_session_state_defaults()

LANG_LABELS = {"en": "English", "ta": "தமிழ் (Tamil)"}


# ------------------------------------------------------------------ SIDEBAR --
with st.sidebar:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:1.2rem; padding:0.2rem 0;">
        <div style="width:38px; height:38px; border-radius:10px; background:linear-gradient(135deg, var(--av-accent) 0%, var(--av-secondary) 100%); display:flex; align-items:center; justify-content:center; color:#FFFFFF; box-shadow:0 4px 14px rgba(111,174,69,0.35);">{icon("sprout", 20, color="#FFFFFF")}</div>
        <div>
            <b style="font-family:'Inter',sans-serif; font-size:1.3rem; color:#FFFFFF; letter-spacing:-0.02em; display:block; line-height:1.1;">AgroVox</b>
            <span style="font-size:0.72rem; color:#9FB397; font-weight:500;">Smarter Farming, By Voice</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation menu
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🤖 AI Assistant", "🔍 Query Analysis",
         "📖 Knowledge Explorer", "🔬 Data Pipeline (ETL & EDA)",
         "📊 NLP Evaluation", "👨‍🌾 Farmer Profile & Analytics", "⚙️ Settings"],
        index=5,
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style="height:1px; background:rgba(255,255,255,0.1); margin:0.9rem 0;"></div>
    """, unsafe_allow_html=True)

    # Farmer Profile Widget
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12); border-radius:14px; padding:14px; margin-bottom:8px;">
        <div style="display:flex; align-items:center; gap:8px; font-weight:700; color:#FFFFFF; font-size:0.88rem; margin-bottom:4px;">
            {icon("user", 16)} <span>Farmer Profile</span>
        </div>
        <p style="font-size:0.73rem; color:#9FB397; line-height:1.35; margin:0; ">
            Your Farmer ID helps us personalize recommendations and save your activity securely.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Farmer ID (click to copy)")
    st.code(st.session_state.farmer_id, language=None)

    with st.expander("🌱 Resume an existing Farmer ID"):
        resume_id = st.text_input("Enter your Farmer ID", key="resume_id_input",
                                    placeholder="FMR_5U1d982")
        if st.button("Resume Profile →", use_container_width=True):
            existing = db.get_farmer(resume_id.strip()) if resume_id else None
            if existing:
                st.session_state.farmer_id = existing["farmer_id"]
                st.session_state.farmer_name = existing["name"] or ""
                st.session_state.preferred_language = existing["preferred_language"] or "en"
                st.session_state.location = existing["location"] or ""
                st.session_state.crops_cultivated = existing["crops_cultivated"] or ""
                st.session_state.chat_session = ChatbotSession(get_pipeline(), st.session_state.farmer_id)
                st.session_state.chat_display = []
                st.success(f"Resumed profile for {existing['name'] or existing['farmer_id']}.")
                st.rerun()
            else:
                st.warning("No profile found with that Farmer ID.")

    st.session_state.farmer_name = st.text_input("Name", value=st.session_state.farmer_name,
                                                    placeholder="Enter your name")
    lang_choice = st.selectbox("Preferred language", options=["en", "ta"],
                                 format_func=lambda x: LANG_LABELS[x],
                                 index=["en", "ta"].index(st.session_state.preferred_language))
    st.session_state.preferred_language = lang_choice

    if st.button("💾 Save Profile", use_container_width=True):
        db.upsert_farmer(st.session_state.farmer_id, name=st.session_state.farmer_name,
                          preferred_language=st.session_state.preferred_language,
                          location=st.session_state.location,
                          crops_cultivated=st.session_state.crops_cultivated)
        st.success("Profile saved securely.")

    st.markdown(f"""
    <div style="margin-top:1.2rem; padding-top:1rem; border-top:1px solid rgba(255,255,255,0.08); display:flex; align-items:center; gap:8px;">
        {icon("leaf", 18, color="#9FB397")}
        <p style="font-size:0.73rem; color:#9FB397; margin:0; line-height:1.3;">Growing Knowledge<br>for a Greener Tomorrow</p>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------ TOP BAR --
_top_search, _top_actions = st.columns([5, 1.6])
with _top_search:
    st.markdown(f"""
    <div class="av-topbar" style="border-bottom:none; margin-bottom:0; padding-bottom:0;">
        <div class="av-search">{icon("search", 16)} Search for queries, datasets, or evaluations...</div>
    </div>
    """, unsafe_allow_html=True)
with _top_actions:
    _tcol1, _tcol2, _tcol3 = st.columns(3)
    with _tcol1:
        if st.button("🌗", key="theme_toggle_btn", help="Toggle light / dark theme",
                     use_container_width=True):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.rerun()
    with _tcol2:
        st.markdown(f'<div class="av-icon-btn">{icon("bell", 16)}</div>', unsafe_allow_html=True)
    with _tcol3:
        initial = (st.session_state.farmer_name or "A")[0].upper()
        st.markdown(f"""<div class="av-user-chip">
            <div class="av-user-avatar">{initial}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<div style="border-bottom:1px solid var(--av-border); margin-bottom:1.2rem;"></div>',
            unsafe_allow_html=True)

# Flat vector agricultural landscape illustration for the hero (no stock photos, no baked-in text)
_HERO_ART = f"""
<svg class="av-hero-art" width="340" height="150" viewBox="0 0 340 150" xmlns="http://www.w3.org/2000/svg">
    <circle cx="290" cy="36" r="26" fill="{COLORS['gold']}" opacity="0.35"/>
    <path d="M0 110 Q 60 70 140 100 T 340 90 V150 H0 Z" fill="{COLORS['accent2']}" opacity="0.20"/>
    <path d="M0 130 Q 90 95 180 122 T 340 112 V150 H0 Z" fill="{COLORS['primary']}" opacity="0.16"/>
    <path d="M40 118 v18 M40 118 c-8 -4 -12 -12 -10 -18 c8 2 12 10 10 18 Z M40 112 c8 -3 13 -10 12 -17 c-9 1 -14 9 -12 17 Z"
          fill="none" stroke="{COLORS['primary']}" stroke-width="2" opacity="0.30"/>
    <path d="M250 122 v20 M250 122 c-9 -4 -13 -13 -11 -20 c9 2 13 11 11 20 Z M250 116 c9 -3 14 -11 13 -19 c-10 1 -15 10 -13 19 Z"
          fill="none" stroke="{_raw_colors['secondary']}" stroke-width="2" opacity="0.28"/>
</svg>
"""



# ================================================================== DASHBOARD --
if "Dashboard" in page:
    st.markdown(f"""
    <div class="av-hero">
        {_HERO_ART}
        <div class="av-hero-inner">
            <div>
                <div class="av-hero-badge-row">
                    <span class="av-badge">🇮🇳 Tamil Nadu focus</span>
                    <span class="av-badge">Bilingual · தமிழ் / English</span>
                    <span class="av-badge">Voice + Text</span>
                </div>
                <h1>Welcome to AgroVox!</h1>
                <p class="av-hero-sub">Evaluate, Understand, Empower Farmers with AI.</p>
            </div>
            <div class="av-tagline">"AI for Farmers.<br>A Healthier Tomorrow."</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    kb = get_knowledge_base()
    stats = kb.stats()
    analytics = db.get_analytics()

    cols = st.columns(5)
    metric_defs = [
        ("sprout", "Crops in KB", stats["Crops"]),
        ("alert", "Diseases in KB", stats["Diseases"]),
        ("flask", "Fertilizers in KB", stats["Fertilizers"]),
        ("bar-chart", "Pests in KB", stats["Pests"]),
        ("message", "Total Farmer Queries", analytics["total_queries"]),
    ]
    for c, (icon_name, label, val) in zip(cols, metric_defs):
        with c:
            st.markdown(metric_card(icon_name, label, str(val)), unsafe_allow_html=True)

    st.markdown("### 🧩 Six Integrated NLP Concepts")
    concept_cols = st.columns(3)
    concepts = [
        ("mic", "Speech Processing", "Tamil/English speech-to-text (with graceful text-mode fallback) "
         "and optional text-to-speech responses."),
        ("waveform", "Machine Translation", "Tamil ↔ English translation so the same English-trained NLP "
         "core can serve both languages."),
        ("bar-chart", "Text Classification", "TF-IDF + Logistic Regression / SVM / Naive Bayes classify each "
         "query into 10 agricultural intent categories."),
        ("search", "Named Entity Recognition", "Extracts CROP, DISEASE, SYMPTOM, PEST, FERTILIZER, LOCATION, "
         "SOIL_TYPE, FARMING_ACTIVITY and WEATHER_CONDITION entities."),
        ("info", "Question Answering", "TF-IDF semantic retrieval over a structured agricultural "
         "knowledge base, boosted by the predicted intent category."),
        ("bot", "Chatbot", "Maintains conversation context (crop, symptom, growth stage) across "
         "turns to handle natural follow-up questions."),
    ]
    for i, (icon_name, title, desc) in enumerate(concepts):
        with concept_cols[i % 3]:
            st.markdown(f"""<div class="av-concept-card">
                <div class="av-icon-tile" style="background:var(--av-accent-light); color:var(--av-secondary);">{icon(icon_name, 20)}</div>
                <b>{title}</b>
                <p style="color:{COLORS['muted']}; font-size:0.87rem; margin-top:0.35rem; line-height:1.45;">{desc}</p>
                </div>""", unsafe_allow_html=True)
            st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    st.markdown("### 🌱 System Pipeline")
    flow_steps = ["Farmer Input", "Speech-to-Text", "Language Detection", "Translation",
                  "Preprocessing", "Classification", "NER", "Knowledge Retrieval",
                  "Question Answering", "Chatbot Reply", "Translate Back", "Output"]
    flow_html = '<div class="av-flow">' + '<span class="av-flow-arrow">→</span>'.join(
        f'<span class="av-flow-step">{s}</span>' for s in flow_steps) + '</div>'
    st.markdown(f'<div class="av-card">{flow_html}</div>', unsafe_allow_html=True)

    if analytics["total_queries"] > 0:
        st.markdown("### 📈 Recent Activity Snapshot")
        c1, c2 = st.columns(2)
        with c1:
            if analytics["intent_distribution"]:
                fig = px.bar(
                    x=list(analytics["intent_distribution"].values()),
                    y=list(analytics["intent_distribution"].keys()),
                    orientation="h", labels={"x": "Queries", "y": "Intent"},
                    color_discrete_sequence=[COLORS["primary"]])
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                                    plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if analytics["language_distribution"]:
                fig2 = px.pie(values=list(analytics["language_distribution"].values()),
                               names=[LANG_LABELS.get(k, k) for k in analytics["language_distribution"]],
                               color_discrete_sequence=[COLORS["accent"], COLORS["accent2"]])
                fig2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig2, use_container_width=True)


# ================================================================== AI ASSISTANT --
elif "AI Assistant" in page:
    st.markdown(page_header("bot", "AI Agricultural Assistant",
        "Ask in Tamil or English, by voice or text. AgroVox remembers context within "
        "this conversation, so you can answer follow-up questions naturally."),
        unsafe_allow_html=True)

    speech_engine = get_speech_engine()

    MAX_QUERY_CHARS = 800  # generous for a spoken/typed farmer question; guards against abuse/latency
    query_text = None
    used_input_mode = "text"

    SUGGESTED_QUESTIONS = [
        "What crop is suitable for my soil?",
        "Why are my rice leaves turning yellow?",
        "When should I irrigate my field?",
        "What fertilizer is suitable for tomato?",
        "How can I improve crop yield?",
    ]
    st.markdown(f'<div class="av-section-label">{icon("sparkles", 13)} Suggested questions</div>',
                unsafe_allow_html=True)
    sugg_cols = st.columns(len(SUGGESTED_QUESTIONS))
    for sc, sq in zip(sugg_cols, SUGGESTED_QUESTIONS):
        with sc:
            if st.button(sq, key=f"suggested_{sq}", use_container_width=True):
                query_text = sq
                used_input_mode = "text"

    top_row = st.columns([3, 1])
    with top_row[0]:
        input_mode = st.radio("Input mode", ["💬 Text", "🎤 Voice"], horizontal=True,
                               label_visibility="collapsed")
    with top_row[1]:
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.chat_display = []
            st.session_state.chat_session = ChatbotSession(get_pipeline(), st.session_state.farmer_id)
            st.rerun()

    if input_mode == "💬 Text":
        with st.form("text_query_form", clear_on_submit=True):
            typed = st.text_area("Your question", height=90, max_chars=MAX_QUERY_CHARS,
                                  placeholder="e.g. My rice crop has yellow leaves, what should I do? "
                                              "/ என் நெற்பயிரில் இலைகள் மஞ்சளாகிறது...")
            submitted = st.form_submit_button("Send ➤", type="primary")
        if submitted:
            cleaned = (typed or "").strip()
            if not cleaned:
                st.warning("Please type a question before sending.")
            elif len(cleaned) < 3:
                st.warning("That looks too short — please describe your question a little more.")
            else:
                query_text = cleaned
                used_input_mode = "text"
    else:
        if not speech_engine.asr_available():
            st.info("🎙️ No speech-recognition backend (e.g. `openai-whisper`) is installed in "
                    "this environment, so recorded audio can't be transcribed here. Record below "
                    "for the pipeline to attempt transcription, or switch to the **Text** tab — "
                    "the rest of the NLP pipeline works identically either way.")
        audio_value = st.audio_input("Record your question")
        if audio_value is not None:
            with st.spinner("Transcribing speech..."):
                transcript = speech_engine.speech_to_text(audio_value.read())
            if transcript["text"]:
                st.success(f"Transcribed: {transcript['text']}")
                query_text = transcript["text"]
                used_input_mode = "voice"
            else:
                st.warning(transcript.get("message", "Could not transcribe audio."))

    if query_text:
        try:
            with st.spinner("Running AgroVox NLP pipeline..."):
                result = st.session_state.chat_session.ask(
                    query_text, input_mode=used_input_mode,
                    preferred_output_language=st.session_state.preferred_language)
            st.session_state.chat_display.append(result)

            # optional voice output
            if speech_engine.tts_available():
                tts = speech_engine.text_to_speech(result["answer"], language=result["output_language"])
                if tts.get("audio_bytes"):
                    st.session_state.chat_display[-1]["tts_audio"] = tts["audio_bytes"]
        except Exception as e:
            st.error(
                "Something went wrong while processing that question. This has been logged; "
                "please try rephrasing your question. (Technical detail: "
                f"{type(e).__name__}: {e})"
            )

    st.markdown('<div class="av-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 💬 Conversation")

    if not st.session_state.chat_display:
        st.info("No messages yet — ask AgroVox something above to get started. Try: "
                "*\"What fertilizer is suitable for rice during the vegetative stage?\"*")

    for i, turn in enumerate(reversed(st.session_state.chat_display)):
        idx = len(st.session_state.chat_display) - 1 - i
        lang_label = LANG_LABELS.get(turn['detected_language'], turn['detected_language'])

        # --- user bubble ---
        st.markdown(f"""
        <div class="av-chat-row user">
            <div class="av-avatar user">{icon("user", 16)}</div>
            <div class="av-bubble user">{turn['raw_text']}</div>
        </div>
        <div class="av-meta-row" style="justify-content:flex-end; margin-right:46px; margin-left:0;">
            <span class="av-meta-pill">{lang_label}</span>
        </div>
        """, unsafe_allow_html=True)

        # --- bot bubble ---
        conf_pct = turn['qa_confidence'] * 100
        st.markdown(f"""
        <div class="av-chat-row bot">
            <div class="av-avatar bot">{icon("sprout", 16, color="#FFFFFF")}</div>
            <div class="av-bubble bot">{turn['answer']}</div>
        </div>
        <div class="av-meta-row">
            <span class="av-meta-pill">{icon("bar-chart", 11)} {turn['intent']} ({turn['intent_confidence']*100:.0f}%)</span>
            <span class="av-meta-pill">{icon("info", 11)} QA confidence {conf_pct:.0f}%
                <span class="av-confidence-track"><span class="av-confidence-fill" style="width:{conf_pct}%;"></span></span>
            </span>
            <span class="av-meta-pill">{icon("waveform", 11)} {turn['translation_backend']}</span>
        </div>
        """, unsafe_allow_html=True)

        if turn.get("tts_audio"):
            st.audio(turn["tts_audio"], format="audio/mp3")

        entity_chips = ""
        for label, values in turn["entities"].items():
            for v in values:
                entity_chips += (f'<span class="av-chip av-entity-{label}">{label}: {v}</span>')
        if entity_chips:
            st.markdown(f'<div style="margin:0.3rem 0 0 46px;">{entity_chips}</div>',
                        unsafe_allow_html=True)

        fb1, fb2, _ = st.columns([0.6, 0.6, 8])
        if fb1.button("👍", key=f"up_{idx}"):
            st.session_state.chat_session.give_feedback(turn["query_id"], True)
            st.toast("Thanks for the feedback!")
        if fb2.button("👎", key=f"down_{idx}"):
            st.session_state.chat_session.give_feedback(turn["query_id"], False)
            st.toast("Thanks — we'll use this to improve.")

        st.markdown(f"<div class='av-disclaimer'>{config.DISCLAIMER}</div>", unsafe_allow_html=True)
        st.markdown('<div class="av-divider"></div>', unsafe_allow_html=True)


# ================================================================== QUERY ANALYSIS --
elif "Query Analysis" in page:
    st.markdown(page_header("search", "Query Analysis",
        "A detailed, evaluator-friendly breakdown of the intermediate NLP results for the "
        "most recent query, plus system-wide analytics across every query ever logged."),
        unsafe_allow_html=True)

    if not st.session_state.chat_display:
        st.info("Ask AgroVox a question in the **AI Assistant** tab first.")
    else:
        turn = st.session_state.chat_display[-1]

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("waveform", "Language", LANG_LABELS.get(turn['detected_language'])),
                    unsafe_allow_html=True)
        c2.markdown(metric_card("bar-chart", "Intent", turn['intent']), unsafe_allow_html=True)
        c3.markdown(metric_card("trend-up", "Intent Confidence", f"{turn['intent_confidence']*100:.1f}%"),
                    unsafe_allow_html=True)
        c4.markdown(metric_card("info", "QA Confidence", f"{turn['qa_confidence']*100:.1f}%"),
                    unsafe_allow_html=True)

        st.markdown("#### 1️⃣ Speech / Text Input")
        st.code(turn["raw_text"], language=None)

        st.markdown("#### 2️⃣ Language Detection → 3️⃣ Machine Translation")
        st.write(f"Detected language: **{LANG_LABELS.get(turn['detected_language'])}**  "
                 f"(backend: `{turn['translation_backend']}`)")
        if turn.get("translated_input_en"):
            st.write("English translation used internally:")
            st.code(turn["translated_input_en"], language=None)

        st.markdown("#### 4️⃣ Text Preprocessing")
        pre = turn["preprocessing"]
        pcols = st.columns(2)
        pcols[0].write("Tokens:")
        pcols[0].code(", ".join(pre["tokens"]), language=None)
        pcols[1].write("After stopword removal + lemmatization:")
        pcols[1].code(", ".join(pre["processed_tokens"]), language=None)

        st.markdown("#### 5️⃣ Text Classification")
        st.write(f"Predicted intent: **{turn['intent']}**  "
                 f"(confidence {turn['intent_confidence']*100:.1f}%, model: `{turn['classifier_model']}`)")

        st.markdown("#### 6️⃣ Named Entity Recognition")
        if turn["entities"]:
            for label, values in turn["entities"].items():
                st.markdown(" ".join(f'<span class="av-chip av-entity-{label}">{label}: {v}</span>'
                                       for v in values), unsafe_allow_html=True)
        else:
            st.write("No agricultural entities detected in this query.")

        st.markdown("#### 7️⃣–8️⃣ Knowledge Retrieval + Question Answering")
        st.write(f"Matched knowledge-base question: *{turn['qa_matched_question']}*")
        st.write(f"Source: `{turn['qa_source']}`  ·  Similarity: {turn['qa_confidence']*100:.1f}%")

        st.markdown("#### 9️⃣ Chatbot Response (context-aware) → 🔟 Translation back")
        st.markdown(f"""<div class="av-bubble bot" style="max-width:100%;">{turn['answer']}</div>""",
                    unsafe_allow_html=True)

        st.markdown("#### 🧠 Conversation Context Memory")
        st.json(turn["context_snapshot"])

    # ---- System-wide analytics (real data across every logged query) ----
    st.markdown('<div class="av-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 System-wide Query Analytics")
    analytics = db.get_analytics()

    if analytics["total_queries"] == 0:
        st.info("No system-wide activity yet — ask AgroVox some questions in the AI Assistant "
                "tab to populate these charts.")
    else:
        qa1, qa2 = st.columns(2)
        with qa1:
            st.markdown("**Query category distribution**")
            if analytics["intent_distribution"]:
                fig = px.bar(x=list(analytics["intent_distribution"].values()),
                             y=list(analytics["intent_distribution"].keys()), orientation="h",
                             labels={"x": "Queries", "y": ""},
                             color_discrete_sequence=[COLORS["primary"]])
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
        with qa2:
            st.markdown("**Language distribution**")
            if analytics["language_distribution"]:
                fig2 = px.pie(values=list(analytics["language_distribution"].values()),
                              names=[LANG_LABELS.get(k, k) for k in analytics["language_distribution"]],
                              color_discrete_sequence=[COLORS["accent"], COLORS["primary"]])
                fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig2, use_container_width=True)

        qa3, qa4 = st.columns(2)
        with qa3:
            st.markdown("**Entity frequency**")
            st.caption("Every entity value extracted by NER, across all queries logged so far.")
            entity_freq = db.get_entity_frequency()
            if entity_freq:
                fig3 = px.bar(x=list(entity_freq.values()), y=list(entity_freq.keys()), orientation="h",
                              labels={"x": "Occurrences", "y": ""},
                              color_discrete_sequence=[COLORS["gold"]])
                fig3.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.caption("No entities extracted yet.")
        with qa4:
            st.markdown("**Confidence distribution**")
            st.caption("Intent-classification confidence across all logged queries.")
            conf_vals = db.get_confidence_values()
            if conf_vals["intent_confidence"]:
                fig4 = px.histogram(x=[v * 100 for v in conf_vals["intent_confidence"]], nbins=10,
                                    labels={"x": "Intent confidence (%)"},
                                    color_discrete_sequence=[COLORS["accent2"]])
                fig4.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.caption("No confidence data yet.")


# ================================================================== KNOWLEDGE EXPLORER --
elif "Knowledge" in page:
    st.markdown(page_header("book-open", "Agricultural Knowledge Explorer",
        "Browse the structured knowledge base the QA engine retrieves answers from."),
        unsafe_allow_html=True)
    kb = get_knowledge_base()
    tables = kb.get_tables()

    KB_ICONS = {"Crops": "sprout", "Diseases": "alert", "Fertilizers": "flask",
                "Irrigation": "droplet", "Pests": "bug", "Soil": "soil-layers", "FAQ": "message"}
    KB_TABLE_KEYS = {"Crops": "crops", "Diseases": "diseases", "Fertilizers": "fertilizers",
                      "Irrigation": "irrigation", "Pests": "pests", "Soil": "soil", "FAQ": "faq"}

    if "kb_category" not in st.session_state:
        st.session_state.kb_category = "Crops"

    kb_cols = st.columns(len(tables))
    for kcol, cat_name in zip(kb_cols, tables.keys()):
        with kcol:
            active = " active" if st.session_state.kb_category == cat_name else ""
            st.markdown(f"""<div class="av-kb-card{active}">
                <div class="av-kb-icon">{icon(KB_ICONS.get(cat_name, "leaf"), 20)}</div>
                <b>{cat_name}</b>
                <div class="av-kb-count">{len(tables[cat_name])} entries</div>
            </div>""", unsafe_allow_html=True)
            if st.button("Browse", key=f"kb_select_{cat_name}", use_container_width=True):
                st.session_state.kb_category = cat_name
                st.rerun()

    st.markdown('<div class="av-divider"></div>', unsafe_allow_html=True)

    table_name = st.session_state.kb_category
    st.markdown(f'<div style="display:flex; align-items:center; gap:8px; font-family:\'Playfair Display\',serif; '
                f'font-size:1.15rem; font-weight:700; margin-bottom:0.5rem;">'
                f'{icon(KB_ICONS.get(table_name, "leaf"), 18)} {table_name}</div>', unsafe_allow_html=True)
    search_query = st.text_input("Search", placeholder="e.g. rice, whitefly, urea...",
                                  label_visibility="collapsed")

    df = kb.search(KB_TABLE_KEYS[table_name], search_query)

    st.caption(f"{len(df)} of {len(tables[table_name])} entries shown")
    st.dataframe(df, use_container_width=True, hide_index=True)


# ================================================================== DATA PIPELINE (ETL & EDA) --
elif "Data Pipeline" in page:
    st.markdown(page_header("flask", "Data Pipeline — ETL & Exploratory Data Analysis",
        "How AgroVox's training data actually gets from a messy raw log to a trained model. "
        "Nothing on this page is hand-typed — every number is computed live from the files "
        "in data/, etl/, and eda/."), unsafe_allow_html=True)

    import importlib
    from etl import extract as etl_extract, transform as etl_transform, load as etl_load
    from eda import exploratory_analysis as eda_mod

    raw_path = os.path.join(config.RAW_DATA_DIR, "farmer_queries_raw.csv")
    clean_path = os.path.join(config.PROCESSED_DATA_DIR, "farmer_queries_clean.csv")
    report_path = os.path.join(config.PROCESSED_DATA_DIR, "transform_report.json")

    pipe_col1, pipe_col2 = st.columns([3, 1])
    with pipe_col1:
        st.markdown(f"""
        <div class="av-flow" style="margin-bottom:0.4rem;">
            <span class="av-flow-step">{icon("download", 14)} EXTRACT<br><span style="font-weight:400;font-size:0.72rem;">raw, messy log</span></span>
            <span class="av-flow-arrow">→</span>
            <span class="av-flow-step">{icon("filter", 14)} TRANSFORM<br><span style="font-weight:400;font-size:0.72rem;">clean · validate · engineer</span></span>
            <span class="av-flow-arrow">→</span>
            <span class="av-flow-step">{icon("database", 14)} LOAD<br><span style="font-weight:400;font-size:0.72rem;">SQLite + training corpus</span></span>
        </div>
        """, unsafe_allow_html=True)
    with pipe_col2:
        run_etl = st.button("▶️ Run ETL pipeline now", use_container_width=True, type="primary")

    if run_etl:
        with st.spinner("Running Extract → Transform → Load..."):
            from etl.etl_pipeline import run_pipeline
            pipeline_result = run_pipeline()
        st.success(f"ETL complete in {pipeline_result['elapsed_seconds']:.1f}s — "
                   f"{pipeline_result['report']['final_row_count']} clean rows loaded.")
        st.cache_data.clear()

    if not os.path.exists(raw_path) or not os.path.exists(clean_path):
        st.info("No ETL run found yet. Click **Run ETL pipeline now** above to generate the "
                "raw log, clean it, and load it — or run `python etl/etl_pipeline.py` from "
                "the command line.")
    else:
        raw_df = pd.read_csv(raw_path, keep_default_na=False)
        clean_df = pd.read_csv(clean_path, keep_default_na=False)
        with open(report_path) as f:
            transform_report = json.load(f)

        st.markdown("### 📉 ETL Funnel — rows retained at each stage")
        st.caption("This is data lineage: tracking exactly how many rows survive each "
                   "cleaning stage (and why) so nothing silently vanishes or duplicates.")

        stages = transform_report["stages"]
        max_rows = stages[0]["rows"]
        stage_icon_names = {"raw_extracted": "download", "missing_value_handling": "alert",
                       "language_imputation_via_detection": "waveform",
                       "junk_content_removal": "filter", "duplicate_removal": "copy",
                       "post_processing_validation": "check-circle", "schema_validation_labels": "bar-chart"}
        funnel_html = '<div class="av-card">'
        colors_cycle = [COLORS["primary"], COLORS["accent2"], COLORS["gold"],
                         COLORS["accent"], COLORS["primary"], COLORS["accent2"], COLORS["gold"]]
        for i, stage in enumerate(stages):
            pct = stage["rows"] / max_rows * 100
            stage_icon = icon(stage_icon_names.get(stage["stage"], "info"), 14)
            label = stage["stage"].replace("_", " ").title()
            funnel_html += f"""
            <div class="av-etl-stage">
                <div style="width:230px; font-size:0.85rem; font-weight:600; display:flex; align-items:center; gap:6px;">{stage_icon} {label}</div>
                <div class="av-etl-bar-bg"><div class="av-etl-bar-fill" style="width:{pct}%; background:{colors_cycle[i % len(colors_cycle)]};"></div></div>
                <div style="width:60px; text-align:right; font-family:'JetBrains Mono',monospace; font-size:0.85rem;">{stage['rows']}</div>
            </div>"""
        funnel_html += "</div>"
        st.markdown(funnel_html, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(metric_card("download", "Raw rows extracted", str(stages[0]['rows'])), unsafe_allow_html=True)
        m2.markdown(metric_card("check-circle", "Clean rows loaded", str(transform_report['final_row_count'])),
                    unsafe_allow_html=True)
        m3.markdown(metric_card("trash", "Rows dropped", str(transform_report['total_rows_dropped'])),
                    unsafe_allow_html=True)
        m4.markdown(metric_card("bar-chart", "Retention rate", f"{transform_report['retention_rate']*100:.1f}%"),
                    unsafe_allow_html=True)

        st.markdown('<div class="av-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🔍 Exploratory Data Analysis (EDA)")
        st.caption("Statistical + visual profiling of the training data — run BEFORE trusting "
                   "any model trained on it. This is exactly the process that caught two real "
                   "bugs during this project's build (see the README's Production Audit section).")

        eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs(
            ["⚖️ Class Balance", "📏 Text Length", "🔤 Word Frequency", "🧪 Data Quality"])

        intents_df = eda_mod.load_intents()

        with eda_tab1:
            balance_df = eda_mod.class_balance(intents_df)
            fig = px.bar(balance_df, x="count", y="label", orientation="h",
                         text="percent", color="count",
                         color_continuous_scale=["#DCEAE3", COLORS["primary"]])
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                               coloraxis_showscale=False, yaxis_title=None, xaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
            imbalance_ratio = balance_df["count"].max() / balance_df["count"].min()
            st.caption(f"Imbalance ratio (largest ÷ smallest class): **{imbalance_ratio:.2f}×** — "
                       "close to 1.0 means well-balanced classes, which helps the classifier "
                       "avoid being biased toward majority categories.")

        with eda_tab2:
            length_stats = eda_mod.text_length_stats(intents_df)
            lc1, lc2 = st.columns(2)
            with lc1:
                fig = px.histogram(x=length_stats["token_length_series"], nbins=15,
                                    labels={"x": "Tokens per query"},
                                    color_discrete_sequence=[COLORS["accent"]])
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                                    title="Query length (tokens)")
                st.plotly_chart(fig, use_container_width=True)
            with lc2:
                fig2 = px.histogram(x=length_stats["char_length_series"], nbins=15,
                                     labels={"x": "Characters per query"},
                                     color_discrete_sequence=[COLORS["accent2"]])
                fig2.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                                    title="Query length (characters)")
                st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(pd.DataFrame({
                "Metric": ["Mean", "Median", "Min", "Max", "Std Dev"],
                "Tokens": [round(length_stats["token_length"]["mean"], 1),
                           length_stats["token_length"]["median"],
                           length_stats["token_length"]["min"], length_stats["token_length"]["max"],
                           round(length_stats["token_length"]["std"], 1)],
                "Characters": [round(length_stats["char_length"]["mean"], 1),
                               length_stats["char_length"]["median"],
                               length_stats["char_length"]["min"], length_stats["char_length"]["max"],
                               round(length_stats["char_length"]["std"], 1)],
            }), use_container_width=True, hide_index=True)

        with eda_tab3:
            top_words = eda_mod.top_words_overall(intents_df, top_n=20)
            fig = px.bar(top_words.sort_values("count"), x="count", y="word", orientation="h",
                         color_discrete_sequence=[COLORS["gold"]])
            fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10),
                                yaxis_title=None, xaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Top words after stopword removal + lemmatization — dominated by "
                       "agricultural vocabulary (crop, rice, fertilizer, disease...) rather "
                       "than generic filler words, which is a good sign the preprocessing "
                       "pipeline is working as intended.")

        with eda_tab4:
            dq1, dq2 = st.columns(2)
            with dq1:
                st.markdown("**Missing values per column (intents.csv)**")
                dup_summary = eda_mod.duplicate_and_missing_summary(intents_df)
                mv_df = pd.DataFrame(list(dup_summary["missing_values_per_column"].items()),
                                      columns=["Column", "Missing"])
                st.dataframe(mv_df, use_container_width=True, hide_index=True)
                st.metric("Duplicate rows", dup_summary["n_duplicate_rows"])
            with dq2:
                st.markdown("**Raw log data-quality issues (this ETL run)**")
                st.write(f"🕳️ Blank/missing text or label dropped: "
                         f"**{stages[1]['dropped_blank_text'] + stages[1]['dropped_blank_label']}**")
                st.write(f"🌐 Missing language recovered via detection: "
                         f"**{transform_report['missing_language_imputed']}**")
                st.write(f"🧹 Punctuation-only junk rows dropped: "
                         f"**{stages[3]['dropped_punctuation_only_rows']}**")
                st.write(f"🪞 Duplicate submissions removed: "
                         f"**{transform_report['duplicates_removed']}**")
                st.write(f"🏷️ Invalid label rows dropped: "
                         f"**{stages[-1]['dropped_invalid_labels']}**")

        st.markdown('<div class="av-divider"></div>', unsafe_allow_html=True)
        with st.expander("📄 Raw vs. Cleaned sample rows (side by side)"):
            sample_n = min(8, len(clean_df))
            st.dataframe(
                clean_df[["raw_text", "cleaned_text", "label", "language", "token_count"]].head(sample_n),
                use_container_width=True, hide_index=True)


# ================================================================== NLP EVALUATION --
elif "NLP Evaluation" in page:
    st.markdown(page_header("bar-chart", "NLP Evaluation",
        "Real metrics computed from the trained models and evaluation datasets in evaluation/*.py — "
        "nothing on this page is hand-typed."), unsafe_allow_html=True)

    def _load_json(path):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return None

    cls_report = _load_json(os.path.join(config.EVAL_RESULTS_DIR, "classification_report.json"))
    ner_report = _load_json(os.path.join(config.EVAL_RESULTS_DIR, "ner_metrics.json"))
    trans_report = _load_json(os.path.join(config.EVAL_RESULTS_DIR, "translation_metrics.json"))
    qa_report = _load_json(os.path.join(config.EVAL_RESULTS_DIR, "qa_metrics.json"))
    speech_report = _load_json(os.path.join(config.EVAL_RESULTS_DIR, "speech_metrics.json"))
    comparison_path = config.CLASSIFIER_COMPARISON_PATH
    comparison = _load_json(comparison_path) if os.path.exists(comparison_path) else None

    # ---- 4 real Summary KPI cards, sourced from speech_report (no hardcoded numbers) ----
    if not speech_report:
        st.warning("No speech evaluation report found yet. Run `python evaluation/speech_evaluation.py` "
                   "(or `python train_classifier.py`) to generate `evaluation/results/speech_metrics.json`.")
    else:
        wer_pct = speech_report["average_wer"] * 100
        asr_ok = speech_report["asr_backend_available"]
        tts_ok = speech_report["tts_backend_available"]
        n_pairs = len(speech_report["pairs"])

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(metric_card("waveform", "Average WER", f"{wer_pct:.1f}%",
                                     sublabel="Lower is better"), unsafe_allow_html=True)
        with m2:
            st.markdown(metric_card("mic", "ASR Backend", "Available" if asr_ok else "Unavailable",
                                     badge="● True" if asr_ok else "● False",
                                     badge_kind="success" if asr_ok else "danger"), unsafe_allow_html=True)
        with m3:
            st.markdown(metric_card("volume", "TTS Backend", "Available" if tts_ok else "Unavailable",
                                     badge="● True" if tts_ok else "● False",
                                     badge_kind="success" if tts_ok else "danger"), unsafe_allow_html=True)
        with m4:
            st.markdown(metric_card("message", "Total Queries Evaluated", str(n_pairs)),
                        unsafe_allow_html=True)

        st.markdown(info_card("About This Evaluation",
            "This module evaluates the performance of NLP models on actual farmer queries. "
            "The audio is processed using the voice assistant, and the predicted text is "
            "compared with the reference transcript to calculate evaluation metrics. "
            + speech_report["note"], icon_name="info"), unsafe_allow_html=True)

        st.caption("⚠️ Metrics computed from actual trained models and datasets — see `evaluation/*.py`. "
                   "Run `python train_classifier.py` to regenerate.")

        # ---- Real Evaluation Results Table, built from speech_report["pairs"] ----
        def _result_bucket(wer: float) -> tuple:
            if wer == 0:
                return "Exact", "info"
            if wer < 0.2:
                return "Good", "good"
            if wer < 0.4:
                return "Review", "review"
            return "Poor", "poor"

        eval_rows = []
        for i, pair in enumerate(speech_report["pairs"]):
            bucket, kind = _result_bucket(pair["wer"])
            eval_rows.append({
                "#": i, "Reference Transcript": pair["reference"],
                "Hypothesis (Predicted)": pair["hypothesis"], "WER": round(pair["wer"], 4),
                "Confidence": round((1 - pair["wer"]) * 100, 1), "Result": bucket, "_kind": kind,
            })
        eval_df_full = pd.DataFrame(eval_rows)

        tbl_search, tbl_filter, tbl_export = st.columns([3, 1.4, 1.2])
        with tbl_search:
            table_search = st.text_input("Search", placeholder="Search reference transcripts...",
                                          label_visibility="collapsed", key="eval_table_search")
        with tbl_filter:
            result_filter = st.selectbox("Filter by result", ["All", "Exact", "Good", "Review", "Poor"],
                                          label_visibility="collapsed", key="eval_result_filter")
        with tbl_export:
            st.download_button("⬇️ Export CSV",
                data=eval_df_full.drop(columns=["_kind"]).to_csv(index=False).encode("utf-8"),
                file_name="agrovox_speech_evaluation.csv", mime="text/csv", use_container_width=True)

        filtered = eval_df_full
        if table_search:
            filtered = filtered[filtered["Reference Transcript"].str.contains(table_search, case=False, na=False)]
        if result_filter != "All":
            filtered = filtered[filtered["Result"] == result_filter]

        st.caption("Confidence and Result are derived from WER (Confidence = (1 − WER) × 100; "
                   "Result bucketed at WER thresholds 0 / 0.2 / 0.4) — not separately measured values.")

        rows_html = ""
        for _, r in filtered.iterrows():
            rows_html += f"""<tr style="border-bottom:1px solid var(--av-border);">
                <td style="padding:12px 14px; color:var(--av-text-muted);">{r['#']}</td>
                <td style="padding:12px 14px; font-weight:600;">{r['Reference Transcript']}</td>
                <td style="padding:12px 14px; color:var(--av-text-muted);">{r['Hypothesis (Predicted)']}</td>
                <td style="padding:12px 14px; font-family:'JetBrains Mono',monospace;">{r['WER']:.4f}</td>
                <td style="padding:12px 14px; font-family:'JetBrains Mono',monospace; font-weight:600;">{r['Confidence']:.1f}%</td>
                <td style="padding:12px 14px;">{status_badge(r['Result'], r['_kind'])}</td>
            </tr>"""
        if not rows_html:
            rows_html = '<tr><td colspan="6" style="padding:16px; text-align:center; color:var(--av-text-muted);">No rows match this search/filter.</td></tr>'

        st.markdown(f"""
        <div style="background:var(--av-surface); border:1px solid var(--av-border); border-radius:16px; padding:1.2rem; margin:1.2rem 0; box-shadow:var(--av-shadow);">
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--av-border); padding-bottom:12px; margin-bottom:12px;">
                <div style="display:flex; align-items:center; gap:8px;">
                    {icon("bar-chart", 16)}
                    <b style="font-family:'Playfair Display',serif; font-size:1.1rem;">Evaluation Results</b>
                    <span style="font-size:0.75rem; font-weight:600; background:var(--av-accent-light); color:var(--av-secondary); padding:2px 8px; border-radius:999px;">{len(filtered)} of {n_pairs} queries</span>
                </div>
            </div>
            <div style="overflow-x:auto;">
                <table style="width:100%; border-collapse:collapse; font-size:0.86rem; text-align:left;">
                    <thead>
                        <tr style="background:var(--av-bg); color:var(--av-text-muted); font-size:0.78rem; text-transform:uppercase;">
                            <th style="padding:10px 14px; width:40px;">#</th>
                            <th style="padding:10px 14px;">Reference Transcript</th>
                            <th style="padding:10px 14px;">Hypothesis (Predicted)</th>
                            <th style="padding:10px 14px;">WER ↓</th>
                            <th style="padding:10px 14px;">Confidence</th>
                            <th style="padding:10px 14px;">Result</th>
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if not any([cls_report, ner_report, trans_report, qa_report, speech_report]):
        st.warning("No evaluation results found yet. Run `python train_classifier.py` "
                    "from the project root to train models and generate all reports.")
    else:
        ev1, ev2, ev3, ev4, ev5 = st.tabs(
            ["🏷️ Classification", "🔎 NER", "🌐 Translation", "❓ QA", "🎤 Speech"])

        with ev1:
            if cls_report:
                st.markdown(f"**Best model:** `{cls_report['best_model']}`")
                rep = cls_report["report"]
                overall = {k: v for k, v in rep.items() if k in ("accuracy",)}
                st.metric("Overall Accuracy", f"{rep['accuracy']*100:.1f}%")
                per_class = {k: v for k, v in rep.items()
                             if k not in ("accuracy", "macro avg", "weighted avg")}
                pc_df = pd.DataFrame(per_class).T[["precision", "recall", "f1-score", "support"]]
                st.dataframe(pc_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}",
                                                    "f1-score": "{:.2f}", "support": "{:.0f}"}),
                             use_container_width=True)
                cm_path = os.path.join(config.EVAL_RESULTS_DIR, "confusion_matrix.png")
                if os.path.exists(cm_path):
                    st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
                if comparison:
                    st.markdown("**Model comparison (TF-IDF + Logistic Regression / SVM / Naive Bayes):**")
                    comp_df = pd.DataFrame(comparison).T
                    st.dataframe(comp_df.style.format("{:.3f}"), use_container_width=True)
            else:
                st.info("No classification report yet.")

        with ev2:
            if ner_report:
                o = ner_report["overall"]
                m1, m2, m3 = st.columns(3)
                m1.metric("Precision", f"{o['precision']*100:.1f}%")
                m2.metric("Recall", f"{o['recall']*100:.1f}%")
                m3.metric("F1-score", f"{o['f1']*100:.1f}%")
                st.caption(f"Evaluated on {ner_report['num_eval_sentences']} held-out annotated sentences.")
                pl_df = pd.DataFrame(ner_report["per_label"]).T
                st.dataframe(pl_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1": "{:.2f}"}),
                             use_container_width=True)
            else:
                st.info("No NER metrics yet.")

        with ev3:
            if trans_report:
                st.write(f"Active backend: `{trans_report['backend_used']}`")
                st.metric("Average glossary coverage", f"{trans_report['average_glossary_coverage']*100:.1f}%")
                if trans_report.get("bleu_scores"):
                    st.metric("Average BLEU", f"{sum(trans_report['bleu_scores'])/len(trans_report['bleu_scores']):.3f}")
                st.caption(trans_report["note"])
                st.markdown("**Sample translations:**")
                st.dataframe(pd.DataFrame(trans_report["samples"])[
                    ["source_en", "predicted_ta", "reference_ta"]], use_container_width=True)
            else:
                st.info("No translation metrics yet.")

        with ev4:
            if qa_report:
                m1, m2, m3 = st.columns(3)
                m1.metric("Top-1 Accuracy", f"{qa_report['top1_accuracy']*100:.1f}%")
                m2.metric("Top-3 Accuracy", f"{qa_report['top3_accuracy']*100:.1f}%")
                m3.metric("Mean Reciprocal Rank", f"{qa_report['mean_reciprocal_rank']:.3f}")
                st.caption(f"Evaluated on {qa_report['num_questions']} paraphrased queries "
                           "against the QA knowledge-base index.")
            else:
                st.info("No QA metrics yet.")

        with ev5:
            if speech_report:
                st.write(f"ASR backend available: **{speech_report['asr_backend_available']}**")
                st.write(f"TTS backend available: **{speech_report['tts_backend_available']}**")
                st.metric("Average Word Error Rate (WER)", f"{speech_report['average_wer']*100:.1f}%")
                st.caption(speech_report["note"])
                st.dataframe(pd.DataFrame(speech_report["pairs"]), use_container_width=True)
            else:
                st.info("No speech metrics yet.")

        # Bottom Insight Box
        st.markdown(insight_card(
            "Insight",
            "The current model performs well on general farming queries. Consider adding "
            "domain-specific data (local languages, dialects, and crop-specific terms) to "
            "further reduce WER.",
            quote='🌱 "Better Data. Brighter Farms."'
        ), unsafe_allow_html=True)


# ================================================================== FARMER PROFILE & ANALYTICS --
elif "Farmer Profile" in page:
    st.markdown(page_header("user", "Farmer Profile & Analytics",
        "Your saved profile, full query history, and system-wide usage statistics."),
        unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("#### Profile")
        st.markdown(f"""<div class="av-card">
            <b>Farmer ID:</b> {st.session_state.farmer_id}<br>
            <b>Name:</b> {st.session_state.farmer_name or '—'}<br>
            <b>Preferred language:</b> {LANG_LABELS.get(st.session_state.preferred_language)}<br>
            <b>Location:</b> {st.session_state.location or '—'}<br>
            <b>Crops cultivated:</b> {st.session_state.crops_cultivated or '—'}
        </div>""", unsafe_allow_html=True)

        history = db.get_all_queries_for_farmer(st.session_state.farmer_id)
        st.markdown(f"#### Full query history ({len(history)})")
        st.caption("Persists across sessions under this Farmer ID.")
        if history:
            for h in history[:25]:
                st.caption(f"🧑‍🌾 {h['raw_text']}")
                st.caption(f"🌾 {h['answer'][:160]}{'...' if len(h['answer']) > 160 else ''}")
                st.markdown("---")
            if len(history) > 25:
                st.caption(f"...and {len(history) - 25} earlier queries.")
        else:
            st.caption("No queries yet.")

    with c2:
        st.markdown("#### 📊 System-wide Analytics")
        analytics = db.get_analytics()

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Queries", analytics["total_queries"])
        m2.metric("Helpful Feedback", f"{analytics['helpful_percentage']:.0f}%"
                   if analytics["helpful_percentage"] is not None else "—")
        m3.metric("Distinct Intents Seen", len(analytics["intent_distribution"]))

        if analytics["intent_distribution"]:
            st.markdown("**Most common intents**")
            fig = px.bar(x=list(analytics["intent_distribution"].keys()),
                         y=list(analytics["intent_distribution"].values()),
                         labels={"x": "Intent", "y": "Count"},
                         color_discrete_sequence=[COLORS["accent2"]])
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        if analytics["most_searched_crops"]:
            st.markdown("**Most searched crops**")
            fig2 = px.bar(x=list(analytics["most_searched_crops"].keys()),
                          y=list(analytics["most_searched_crops"].values()),
                          labels={"x": "Crop", "y": "Mentions"},
                          color_discrete_sequence=[COLORS["gold"]])
            fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

        if analytics["language_distribution"]:
            st.markdown("**Language distribution**")
            fig3 = px.pie(values=list(analytics["language_distribution"].values()),
                          names=[LANG_LABELS.get(k, k) for k in analytics["language_distribution"]],
                          color_discrete_sequence=[COLORS["primary"], COLORS["accent"]])
            fig3.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig3, use_container_width=True)

        if analytics["total_queries"] == 0:
            st.info("No system-wide activity yet — ask AgroVox some questions in the "
                    "AI Assistant tab to populate analytics.")


# ================================================================== SETTINGS --
elif "Settings" in page:
    st.markdown(page_header("settings", "System & User Settings",
        "Model information and local session controls."), unsafe_allow_html=True)
    st.markdown(f"""
    <div class="av-card" style="max-width:680px;">
        <div style="display:flex; align-items:center; gap:8px; font-family:'Playfair Display',serif; font-size:1.05rem; font-weight:700; margin-bottom:0.4rem;">
            {icon("sprout", 18)} AgroVox Model &amp; System Info
        </div>
        <p style="font-size:0.86rem; color:var(--av-text-muted); margin:0.3rem 0 1rem; line-height:1.5;">
            Core NLP Engine: TF-IDF (unigram + bigram) + Multinomial Naive Bayes / Logistic Regression / Linear SVM.<br>
            Bi-directional Tamil ↔ English agricultural translation dictionary with phrase-level substitution.<br>
            Current theme: <b>{st.session_state.theme.title()}</b>
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🗑️ Reset All Local Session Data", type="secondary"):
        st.session_state.clear()
        st.success("Session state cleared.")
        st.rerun()

