"""
Custom CSS for the Streamlit app — dark theme.
"""

import streamlit as st

CUSTOM_CSS = """
<style>

/* ---------- Base app ---------- */

.stApp {
    background: linear-gradient(
        160deg,
        #0b1120 0%,
        #0f172a 45%,
        #111827 100%
    );
    color: #e2e8f0;
}

.stApp, .stApp p, .stApp span, .stApp label,
.stApp li, .stApp .stMarkdown, h1, h2, h3, h4, h5, h6 {
    color: #e2e8f0;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ---------- Sidebar ---------- */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1120 0%, #131c31 100%);
    border-right: 1px solid #1e293b;
}

section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

section[data-testid="stSidebar"] .stRadio > label {
    font-weight: 600;
    color: #93a3c1 !important;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
}

/* Sidebar nav radio styled like pill tabs */
section[data-testid="stSidebar"] [role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 6px;
}

section[data-testid="stSidebar"] [role="radiogroup"] label {
    background: #16213b;
    border: 1px solid #263255;
    border-radius: 12px;
    padding: 10px 14px !important;
    transition: 0.15s ease-in-out;
}

section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    border-color: #6366f1;
    background: #1a2444;
}

/* ---------- Header / hero ---------- */

.app-hero {
    background: linear-gradient(135deg, #1e1b4b, #0f172a);
    border: 1px solid #312e81;
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 10px 40px rgba(79, 70, 229, 0.15);
}

.app-hero h1 {
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #a5b4fc, #f0abfc);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.app-hero p {
    color: #94a3b8;
    margin-top: 6px;
    margin-bottom: 0;
    font-size: 0.95rem;
}

/* ---------- Panels / file cards ---------- */

.panel {
    background: #131c31;
    border: 1px solid #1e293b;
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

.file-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
    color: #f1f5f9;
}

/* ---------- File uploader ---------- */

[data-testid="stFileUploader"] section {
    background: #131c31 !important;
    border: 1.5px dashed #334155 !important;
    border-radius: 16px !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] * {
    color: #cbd5e1 !important;
}

/* Browse-files button — testid suffix stays "Button-secondary" across
   Streamlit versions even though the prefix casing has changed
   ("stBaseButton-secondary" vs "stbaseButton-secondary"). */
[data-testid$="Button-secondary"] {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}

[data-testid$="Button-secondary"]:hover {
    background: #263255 !important;
    border-color: #6366f1 !important;
}

[data-testid="stFileUploaderFileName"] {
    color: #e2e8f0 !important;
    font-weight: 500;
}

/* ---------- Tabs (legacy, kept in case re-used) ---------- */

.stTabs [data-baseweb="tab"] {
    color: #94a3b8;
}

.stTabs [aria-selected="true"] {
    color: #f1f5f9 !important;
}

/* ---------- Chat input ---------- */

[data-testid="stChatInput"] {
    border-radius: 14px !important;
}

[data-testid="stChatInput"] textarea {
    color: #f1f5f9 !important;
}

.stTextInput input {
    border-radius: 14px !important;
    border: 1px solid #334155 !important;
    background: #0f172a !important;
    color: #f1f5f9 !important;
    padding: 12px !important;
    font-size: 16px !important;
}

/* ---------- Chat messages ---------- */

[data-testid="stChatMessage"] {
    border-radius: 16px !important;
    padding: 14px !important;
    margin-top: 12px !important;
    border: 1px solid #1e293b !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #1e2a5e !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: #131c31 !important;
}

[data-testid="stChatMessageContent"], [data-testid="stChatMessageContent"] p {
    color: #e2e8f0 !important;
    font-size: 16px !important;
    line-height: 1.7 !important;
}

/* ---------- Buttons ---------- */

.stButton button,
.stDownloadButton button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 600 !important;
    transition: 0.2s ease-in-out !important;
}

.stButton button:hover,
.stDownloadButton button:hover {
    filter: brightness(1.1);
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(99, 102, 241, 0.35);
}

/* ---------- Expander ---------- */

.streamlit-expanderHeader, [data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: #cbd5e1 !important;
    background: #131c31 !important;
    border-radius: 10px !important;
}

/* ---------- Metrics ---------- */

[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}

/* ---------- Alerts (success/info/warning/error) ---------- */

[data-testid="stAlert"] {
    border-radius: 14px !important;
    border: 1px solid #1e293b !important;
}

/* ---------- Progress bar ---------- */

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #a855f7) !important;
}

/* ---------- Divider ---------- */

hr {
    border-color: #1e293b !important;
}

/* ---------- Cards / bordered containers ---------- */

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px !important;
    border: 1px solid #1e293b !important;
    background: #131c31 !important;
}

/* ---------- Code blocks (retrieved source snippets) ---------- */

[data-testid="stCodeBlock"] pre {
    background: #0b1120 !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
}
</style>
"""


def apply_custom_styles() -> None:
    """Inject the app's custom CSS into the page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
