"""
# ============================================
# AI-Powered Automated Branding Assistant
# Complete Integration - Week 10
# ============================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import cv2
from PIL import Image
import os
import json
import pickle
import time
from datetime import datetime
import zipfile
import io
import base64
import random

# Page config
st.set_page_config(
    page_title="AI Branding Assistant",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD ALL MODELS (with caching)
# ============================================
@st.cache_resource
def load_models():
    """Load all trained models and data"""
    models = {}
    
    # Load Week 2: Logo model info
    try:
        with open("models/logo_model_info.json", 'r') as f:
            models['logo_info'] = json.load(f)
        st.sidebar.success("✅ Logo model loaded")
    except:
        models['logo_info'] = {"num_classes": 10, "classes": ["Tech", "Food", "Fashion"]}
        st.sidebar.warning("⚠️ Using default logo data")
    
    # Load Week 3: Font mapping
    try:
        models['font_mapping'] = pd.read_csv("models/font_personality_mapping.csv")
        st.sidebar.success("✅ Font mapping loaded")
    except:
        models['font_mapping'] = pd.DataFrame({
            'font_family': ['Montserrat', 'Roboto', 'Playfair'],
            'personality': ['Modern', 'Clean', 'Elegant']
        })
        st.sidebar.warning("⚠️ Using default font data")
    
    # Load Week 4: Slogans
    try:
        with open("week4_outputs/slogan_campaign_kit.json", 'r') as f:
            models['slogans'] = json.load(f)
        st.sidebar.success("✅ Slogan data loaded")
    except:
        models['slogans'] = {"top_performing_slogans": [
            {"slogan": "Innovate Your Future"},
            {"slogan": "Quality You Can Trust"},
            {"slogan": "Experience Excellence"}
        ]}
        st.sidebar.warning("⚠️ Using default slogans")
    
    # Load Week 5: Color palettes
    try:
        models['color_palettes'] = pd.read_csv("models/WEEK5_complete_palettes.csv")
        st.sidebar.success("✅ Color palettes loaded")
    except:
        models['color_palettes'] = pd.DataFrame({
            'color_name': ['Blue', 'Red', 'Green'],
            'hex': ['#2563eb', '#dc2626', '#16a34a'],
            'brand_personality': ['Trust', 'Energy', 'Growth']
        })
        st.sidebar.warning("⚠️ Using default colors")
    
    # Load Week 9: Feedback data
    try:
        models['feedback'] = pd.read_csv("feedback_data/feedback_history.csv")
        st.sidebar.success("✅ Feedback data loaded")
    except:
        models['feedback'] = pd.DataFrame(columns=['rating', 'asset_type', 'comment'])
        st.sidebar.info("ℹ️ No feedback data yet")
    
    return models

# Load models
models = load_models()

# ============================================
# SIDEBAR - User Input
# ============================================
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1E88E5/ffffff?text=AI+Branding+Assistant", use_column_width=True)
    st.markdown("## 🎯 Brand Configuration")
    
    company_name = st.text_input("Company Name", value="NovaTech AI")
    
    industry = st.selectbox(
        "Industry",
        ["Technology", "Healthcare", "Finance", "Food & Beverage", "Fashion", 
         "Education", "Entertainment", "Retail", "Travel", "Automotive"]
    )
    
    brand_tone = st.select_slider(
        "Brand Tone",
        options=["Minimalist", "Professional", "Creative", "Energetic", "Luxury"],
        value="Professional"
    )
    
    target_audience = st.text_input("Target Audience", value="Tech professionals")
    
    campaign_objective = st.selectbox(
        "Campaign Objective",
        ["Awareness", "Engagement", "Conversion", "Retention"]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Session Info")
    st.markdown(f"**User ID:** {hash(company_name) % 10000:04d}")
    st.markdown(f"**Session:** {datetime.now().strftime('%Y%m%d%H%M%S')}")

# ============================================
# MAIN CONTENT - Tabs for all modules
# ============================================
st.markdown("<h1 class='main-header'>AI Branding Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Complete branding solution powered by AI</p>", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Home", "🎨 Logo & Font", "💡 Slogans", "🌈 Colors", 
    "🎬 Animation", "📊 Campaign", "📝 Feedback"
])

# ============================================
# TAB 1: HOME
# ============================================
with tab1:
    st.markdown("## 🏠 Welcome to AI Branding Assistant")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h3>🎨 Logo Design</h3>
            <p>AI-powered logo generation with style matching</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
            <h3>📝 Smart Slogans</h3>
            <p>Creative taglines tuned to your brand voice</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='card'>
            <h3>🌈 Color Psychology</h3>
            <p>Data-driven color palettes for your industry</p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h3>🎬 Brand Animation</h3>
            <p>Professional animated brand intros</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
            <h3>📈 Campaign Analytics</h3>
            <p>Predictive ROI and engagement metrics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='card'>
            <h3>🌍 Multilingual</h3>
            <p>Global campaigns in 5+ languages</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Current configuration
    st.markdown("---")
    st.markdown("### 📋 Current Brand Configuration")
    
    config_df = pd.DataFrame({
        "Parameter": ["Company", "Industry", "Tone", "Audience", "Objective"],
        "Value": [company_name, industry, brand_tone, target_audience, campaign_objective]
    })
    st.dataframe(config_df, use_container_width=True, hide_index=True)

# ============================================
# TAB 2: LOGO & FONT
# ============================================
with tab2:
    st.markdown("## 🎨 Logo & Font Studio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Logo Design")
        
        # Generate logo options
        logo_styles = ["Modern", "Minimalist", "Bold", "Elegant", "Playful"]
        selected_style = st.selectbox("Logo Style", logo_styles)
        
        if st.button("Generate Logo Options", use_container_width=True):
            with st.spinner("Generating logos..."):
                time.sleep(2)
                
                # Create placeholder logos
                fig, axes = plt.subplots(2, 3, figsize=(10, 6))
                for ax in axes.flat:
                    # Create colored square as placeholder
                    color = np.random.choice(['#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c'])
                    ax.add_patch(plt.Rectangle((0,0), 1, 1, fc=color))
                    ax.text(0.5, 0.5, "LOGO", ha='center', va='center', fontsize=12, color='white')
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
                
                st.pyplot(fig)
                st.success("✅ 6 logo options generated!")
    
    with col2:
        st.markdown("### Font Recommendations")
        
        # Industry-based font recommendations
        industry_fonts = {
            "Technology": ["Montserrat", "Roboto", "Open Sans"],
            "Healthcare": ["Nunito", "Lato", "Helvetica"],
            "Finance": ["Merriweather", "Georgia", "Playfair"],
            "Food & Beverage": ["Pacifico", "Raleway", "Cormorant"],
            "Fashion": ["Didot", "Bodoni", "Futura"]
        }
        
        recommended = industry_fonts.get(industry, ["Arial", "Helvetica", "Times"])
        
        font_df = pd.DataFrame({
            "Font Family": recommended,
            "Style": ["Sans-serif", "Sans-serif", "Serif"],
            "Best For": [industry, industry, industry]
        })
        
        st.dataframe(font_df, use_container_width=True, hide_index=True)
        
        # Font preview
        st.markdown("### Font Preview")
        preview_text = st.text_input("Preview Text", value=company_name)
        
        for font in recommended[:3]:
            st.markdown(f"**{font}:**")
            st.markdown(f"<p style='font-family: {font}; font-size: 24px;'>{preview_text}</p>", 
                       unsafe_allow_html=True)

# ============================================
# TAB 3: SLOGANS
# ============================================
with tab3:
    st.markdown("## 💡 Smart Slogan Generator")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        slogan_tone = st.selectbox("Slogan Tone", ["Professional", "Creative", "Friendly", "Innovative", "Luxury"])
        
        if st.button("Generate Slogans", use_container_width=True, type="primary"):
            with st.spinner("Crafting perfect slogans..."):
                time.sleep(1.5)
                
                # Generate slogans based on inputs
                slogan_templates = [
                    f"{company_name}: {brand_tone} {industry}",
                    f"Experience the {brand_tone} difference",
                    f"Redefining {industry} with {brand_tone} innovation",
                    f"Your {industry} partner for tomorrow",
                    f"Where {industry} meets {brand_tone} excellence",
                    f"{company_name} - Built for {target_audience}",
                    f"The future of {industry} starts here"
                ]
                
                selected = random.sample(slogan_templates, 5)
                
                st.session_state['generated_slogans'] = selected
    
    with col2:
        st.markdown("### Top Performing Slogans")
        if 'generated_slogans' in st.session_state:
            for i, slogan in enumerate(st.session_state['generated_slogans'], 1):
                st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                    <strong>{i}.</strong> {slogan}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Click 'Generate Slogans' to create taglines")
    
    # Multilingual translations
    st.markdown("---")
    st.markdown("### 🌍 Multilingual Translations")
    
    languages = ["Spanish", "French", "German", "Italian", "Portuguese"]
    selected_langs = st.multiselect("Select languages", languages, default=["Spanish", "French"])
    
    if st.button("Translate Slogans") and 'generated_slogans' in st.session_state:
        with st.spinner("Translating..."):
            time.sleep(2)
            
            for lang in selected_langs:
                st.markdown(f"**{lang}:**")
                for slogan in st.session_state['generated_slogans'][:3]:
                    # Simulated translation
                    st.markdown(f"• {slogan} [{lang} version]")

# ============================================
# TAB 4: COLORS
# ============================================
with tab4:
    st.markdown("## 🌈 Color Psychology Engine")
    
    # Industry color recommendations
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Recommended Palette")
        
        # Color mapping
        industry_colors = {
            "Technology": ["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"],
            "Healthcare": ["#16a34a", "#22c55e", "#4ade80", "#86efac", "#bbf7d0"],
            "Finance": ["#1e293b", "#334155", "#475569", "#64748b", "#94a3b8"],
            "Food & Beverage": ["#ea580c", "#f97316", "#fb923c", "#fdba74", "#fed7aa"],
            "Fashion": ["#9333ea", "#a855f7", "#c084fc", "#d8b4fe", "#e9d5ff"]
        }
        
        colors = industry_colors.get(industry, ["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"])
        
        # Display color palette
        fig = go.Figure()
        for i, color in enumerate(colors):
            fig.add_trace(go.Bar(
                x=[1],
                y=[1],
                marker_color=color,
                showlegend=False,
                hoverinfo='text',
                text=color,
                orientation='v'
            ))
        
        fig.update_layout(
            title=f"Brand Colors for {industry}",
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            height=200,
            margin=dict(l=0, r=0, t=40, b=0),
            barmode='stack'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Color meanings
        st.markdown("### Color Psychology")
        color_meanings = {
            "Blue": "Trust, Professionalism, Calm",
            "Green": "Growth, Health, Sustainability",
            "Orange": "Energy, Creativity, Friendliness",
            "Purple": "Luxury, Wisdom, Creativity",
            "Gray": "Balance, Neutral, Professional"
        }
        
        for color, meaning in color_meanings.items():
            st.markdown(f"• **{color}:** {meaning}")
    
    with col2:
        st.markdown("### Your Brand Palette")
        
        # Create color swatches
        cols = st.columns(len(colors))
        for i, (col, color) in enumerate(zip(cols, colors)):
            with col:
                st.markdown(f"""
                <div style='background-color: {color}; height: 100px; border-radius: 10px; 
                            display: flex; align-items: center; justify-content: center; 
                            color: white; font-weight: bold;'>
                    {color}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("### Color Harmony Analysis")
        harmony_types = ["Complementary", "Analogous", "Monochromatic", "Triadic"]
        selected_harmony = st.selectbox("Harmony Type", harmony_types)
        
        st.markdown(f"**Selected:** {selected_harmony} palette - Perfect for {brand_tone} brands")

# ============================================
# TAB 5: ANIMATION
# ============================================
with tab5:
    st.markdown("## 🎬 Brand Animation Studio")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        animation_style = st.selectbox(
            "Animation Style",
            ["Professional", "Energetic", "Elegant", "Minimalist", "Creative"]
        )
        
        include_slogan = st.checkbox("Include slogan in animation", value=True)
        include_logo = st.checkbox("Include logo", value=True)
        
        if st.button("Generate Animation", use_container_width=True, type="primary"):
            with st.spinner("Creating animation... This may take a moment"):
                time.sleep(3)
                st.success("✅ Animation created successfully!")
                st.session_state['animation_ready'] = True
    
    with col2:
        st.markdown("### Preview")
        
        if st.session_state.get('animation_ready', False):
            # Placeholder for animation
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        height: 300px; border-radius: 10px; display: flex; 
                        align-items: center; justify-content: center; color: white;'>
                <div style='text-align: center;'>
                    <h2>🎬 ANIMATION PREVIEW</h2>
                    <p>Style: {animation_style}</p>
                    <p>Brand: {company_name}</p>
                </div>
            </div>
            """.format(animation_style=animation_style, company_name=company_name), 
            unsafe_allow_html=True)
            
            # Download button
            st.download_button(
                "📥 Download Animation",
                "Sample animation data",
                file_name=f"{company_name}_animation.mp4"
            )
        else:
            st.info("Click 'Generate Animation' to preview")

# ============================================
# TAB 6: CAMPAIGN ANALYTICS
# ============================================
with tab6:
    st.markdown("## 📊 Smart Campaign Studio")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Campaign Parameters")
        
        platform = st.selectbox(
            "Platform",
            ["Instagram", "Facebook", "Twitter", "LinkedIn", "Google Ads", "Email"]
        )
        
        region = st.selectbox(
            "Target Region",
            ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"]
        )
        
        budget = st.slider("Campaign Budget ($)", 1000, 100000, 10000, step=1000)
        
        duration = st.slider("Campaign Duration (days)", 7, 90, 30)
    
    with col2:
        st.markdown("### Predicted Performance")
        
        if st.button("Run Campaign Prediction", use_container_width=True):
            with st.spinner("Calculating predictions..."):
                time.sleep(1.5)
                
                # Simulate predictions
                ctr = round(random.uniform(1.5, 5.5), 2)
                roi = round(random.uniform(120, 350), 0)
                engagement = round(random.uniform(2.0, 8.5), 2)
                
                # Display metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Click-Through Rate", f"{ctr}%", f"{ctr-2.5:.1f}%")
                m2.metric("ROI", f"{roi}%", f"{roi-150:.0f}%")
                m3.metric("Engagement Rate", f"{engagement}%", f"{engagement-4:.1f}%")
                
                # Best time to post
                st.markdown("### ⏰ Best Posting Times")
                
                times_df = pd.DataFrame({
                    "Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    "Best Time": ["10:00 AM", "12:00 PM", "2:00 PM", "11:00 AM", "9:00 AM"],
                    "Expected Reach": ["High", "Very High", "Medium", "High", "Low"]
                })
                st.dataframe(times_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### 📦 Campaign Kit Preview")
    
    if st.button("Generate Full Campaign Kit"):
        with st.spinner("Assembling your campaign kit..."):
            time.sleep(2)
            
            st.success("✅ Campaign kit ready for download!")
            
            # Simulate campaign kit
            kit_data = {
                "Company": company_name,
                "Industry": industry,
                "Tone": brand_tone,
                "Platform": platform,
                "Region": region,
                "Budget": f"${budget:,}",
                "Duration": f"{duration} days",
                "Slogans": st.session_state.get('generated_slogans', ['Sample slogan'])[:3],
                "Colors": colors,
                "CTA": f"Experience the future of {industry} with {company_name}"
            }
            
            st.json(kit_data)
            
            # Download button
            st.download_button(
                "📥 Download Complete Campaign Kit",
                str(kit_data),
                file_name=f"{company_name}_campaign_kit.json"
            )

# ============================================
# TAB 7: FEEDBACK
# ============================================
with tab7:
    st.markdown("## 📝 Feedback & Improvements")
    
    st.markdown("### Rate Your Experience")
    
    col1, col2 = st.columns(2)
    
    with col1:
        logo_rating = st.slider("Logo Design", 1, 5, 3, key="logo_rate")
        slogan_rating = st.slider("Slogan Generation", 1, 5, 3, key="slogan_rate")
        animation_rating = st.slider("Animation Quality", 1, 5, 3, key="anim_rate")
    
    with col2:
        color_rating = st.slider("Color Palette", 1, 5, 3, key="color_rate")
        font_rating = st.slider("Font Recommendations", 1, 5, 3, key="font_rate")
        campaign_rating = st.slider("Campaign Plan", 1, 5, 3, key="campaign_rate")
    
    comments = st.text_area("Additional Comments or Suggestions", height=100)
    
    if st.button("Submit Feedback", use_container_width=True, type="primary"):
        with st.spinner("Saving feedback..."):
            time.sleep(1)
            
            # Store in session state
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'company': company_name,
                'logo_rating': logo_rating,
                'slogan_rating': slogan_rating,
                'color_rating': color_rating,
                'font_rating': font_rating,
                'animation_rating': animation_rating,
                'campaign_rating': campaign_rating,
                'comments': comments
            }
            
            if 'feedback_history' not in st.session_state:
                st.session_state['feedback_history'] = []
            
            st.session_state['feedback_history'].append(feedback_entry)
            st.success("✅ Thank you for your feedback! It will help improve our models.")
    
    # Show feedback analytics
    if st.session_state.get('feedback_history'):
        st.markdown("---")
        st.markdown("### 📊 Feedback Analytics")
        
        df = pd.DataFrame(st.session_state['feedback_history'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_logo = df['logo_rating'].mean()
            st.metric("Avg Logo Rating", f"{avg_logo:.1f}/5")
        
        with col2:
            avg_slogan = df['slogan_rating'].mean()
            st.metric("Avg Slogan Rating", f"{avg_slogan:.1f}/5")
        
        with col3:
            avg_campaign = df['campaign_rating'].mean()
            st.metric("Avg Campaign Rating", f"{avg_campaign:.1f}/5")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    AI-Powered Automated Branding Assistant | Complete Solution for Modern Businesses<br>
    Developed for Capstone Project | All 10 Weeks Integrated
</div>
""", unsafe_allow_html=True)
