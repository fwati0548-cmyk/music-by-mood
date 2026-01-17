"""
🎵 MELORA - Music Recommendation System
Landing Page - Music Only
"""

import streamlit as st
import os
import base64
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="MELORA - Music Recommendation System",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =======================
# Custom CSS
# =======================
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        display: none;
    }

    .main {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    }

    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }

    .hero-subtitle {
        font-size: 1.5rem;
        color: #CBD5E1;
        margin-bottom: 3rem;
        text-align: center;
    }

    .feature-card {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        border-color: #6366F1;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.25);
    }

    .feature-icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
    }

    .feature-title {
        font-size: 2rem;
        font-weight: 700;
        color: #000000;
        margin-bottom: 1rem;
    }

    .feature-description {
        font-size: 1rem;
        color: #CBD5E1;
        line-height: 1.7;
    }

    .stButton>button {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: white;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.9rem 2rem;
        border-radius: 999px;
        border: none;
        width: 100%;
        margin-top: 1.5rem;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%);
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(99, 102, 241, 0.4);
    }

    .footer {
        text-align: center;
        color: #64748B;
        font-size: 0.85rem;
        padding-top: 3rem;
        margin-top: 4rem;
    }
</style>
""", unsafe_allow_html=True)

# =======================
# Logo
# =======================
logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.jpeg")

if os.path.exists(logo_path):
    with open(logo_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(f"""
    <div style="display:flex; justify-content:center; margin-bottom:1.5rem;">
        <img src="data:image/jpeg;base64,{encoded}"
             style="width:200px; height:200px; border-radius:50%;
                    object-fit:cover;
                    box-shadow:0 10px 40px rgba(99,102,241,0.3);">
    </div>
    """, unsafe_allow_html=True)

# =======================
# Subtitle
# =======================
st.markdown("""
<div class="hero-subtitle">
    AI-Powered Music Recommendation System
</div>
""", unsafe_allow_html=True)

# =======================
# Music Card (CENTER)
# =======================
st.markdown("""
<div class="feature-card">
    <div class="feature-icon">🎵</div>
    <div class="feature-title">Music Recommendation</div>
    <div class="feature-description">
        Discover songs based on your mood and preferences<br><br>
        • Mood-based classification<br>
        • 100,000+ music tracks<br>
        • Audio feature analysis<br>
        • Interactive visual insights
    </div>
</div>
""", unsafe_allow_html=True)

if st.button("Explore Music 🎧", use_container_width=True):
    st.switch_page("pages/1_Music.py")

# =======================
# Footer
# =======================
st.markdown("""
<div class="footer">
    MELORA • Final Project • AI Music Recommendation System
</div>
""", unsafe_allow_html=True)
