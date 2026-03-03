
# ============================================
# STREAMLIT FEEDBACK INTERFACE
# Copy this code to your Week 10 Streamlit app
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

# Page config
st.set_page_config(page_title="AI Branding Assistant - Feedback", layout="wide")

# Title
st.title("📊 Branding Asset Feedback")
st.markdown("Help us improve by rating your generated assets")

# Load feedback data
@st.cache_data
def load_feedback_data():
    return pd.read_csv("feedback_data/feedback_history.csv")

df = load_feedback_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Rate Assets", "View Analytics", "Model Performance"])

if page == "Rate Assets":
    st.header("⭐ Rate Your Branding Assets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Logo Design")
        logo_rating = st.slider("Rate the generated logo (1-5)", 1, 5, 3)
        logo_comment = st.text_area("Comments about the logo", key="logo")
        if st.button("Submit Logo Feedback"):
            st.success("✅ Logo feedback submitted!")
    
    with col2:
        st.subheader("Font Recommendation")
        font_rating = st.slider("Rate the font recommendation (1-5)", 1, 5, 3)
        font_comment = st.text_area("Comments about the font", key="font")
        if st.button("Submit Font Feedback"):
            st.success("✅ Font feedback submitted!")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Slogan Generation")
        slogan_rating = st.slider("Rate the generated slogan (1-5)", 1, 5, 3)
        slogan_comment = st.text_area("Comments about the slogan", key="slogan")
        if st.button("Submit Slogan Feedback"):
            st.success("✅ Slogan feedback submitted!")
    
    with col4:
        st.subheader("Campaign Plan")
        campaign_rating = st.slider("Rate the campaign plan (1-5)", 1, 5, 3)
        campaign_comment = st.text_area("Comments about the campaign", key="campaign")
        if st.button("Submit Campaign Feedback"):
            st.success("✅ Campaign feedback submitted!")

elif page == "View Analytics":
    st.header("📈 Feedback Analytics")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Ratings", len(df))
    with col2:
        st.metric("Average Rating", f"{df['rating'].mean():.2f}/5")
    with col3:
        st.metric("Comments", len(df[df['comment'].str.len() > 0]))
    with col4:
        st.metric("Avg Sentiment", f"{df['sentiment_score'].mean():.2f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(df.groupby('asset_type')['rating'].mean().reset_index(),
                     x='asset_type', y='rating', title='Average Rating by Asset Type')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x='rating', nbins=5, title='Rating Distribution')
        st.plotly_chart(fig, use_container_width=True)

else:
    st.header("🤖 Model Performance")
    st.info("Model refinement based on feedback will be applied here")
