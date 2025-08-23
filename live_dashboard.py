import streamlit as st
import pandas as pd
import json
import os
import glob
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from thread_conversation_extractor import ThreadConversationExtractor
from config import REDDIT_CONFIG

# --- Page Configuration ---
st.set_page_config(page_title="Mental Health Analyzer", layout="wide")

# --- Model & Extractor Loading (with caching for performance) ---
@st.cache_resource
def load_sentiment_model():
    model_path = "./my_custom_mental_health_model"
    if not os.path.exists(model_path): return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def get_extractor():
    return ThreadConversationExtractor(REDDIT_CONFIG)

sentiment_pipeline = load_sentiment_model()
extractor = get_extractor()

# --- Helper Function for Conclusive Sentiment ---
def get_conclusive_sentiment(df):
    if 'predicted_label' not in df.columns:
        return "Not Analyzed"
    
    label_counts = df['predicted_label'].value_counts(normalize=True)
    pain_percentage = label_counts.get('Expressing-Pain', 0)
    hope_percentage = label_counts.get('Expressing-Hope', 0)
    support_percentage = label_counts.get('Offering-Support', 0)

    if pain_percentage >= 0.4:
        return "Highly Concerning"
    elif (hope_percentage + support_percentage) >= 0.5:
        return "Supportive & Improving"
    elif pain_percentage > (hope_percentage + support_percentage):
        return "Leaning Negative"
    else:
        return "Mixed or Neutral Discussion"

# --- Main App Structure ---
st.title("🧠 Mental Health Reddit Analyzer")
st.sidebar.title("Controls")

selected_tab = st.radio("Select View", ["🔴 Live Monitor", "📖 On-Demand Explorer"], horizontal=True, label_visibility="collapsed")

# ==============================================================================
#                             TAB 1: LIVE MONITOR
# ==============================================================================
if selected_tab == "🔴 Live Monitor":
    # (Live Monitor code remains the same as the last working version)
    st.sidebar.header("Live Monitor")
    st.sidebar.info("Run `python3 live_analyzer.py` in a separate terminal to start the feed.")
    st.header("Live Analysis of New Reddit Comments")

    if 'comment_history' not in st.session_state:
        st.session_state.comment_history = []
    if 'last_timestamp' not in st.session_state:
        st.session_state.last_timestamp = 0
    if 'label_counts' not in st.session_state:
        st.session_state.label_counts = {}

    try:
        with open("live_data.json", "r") as f:
            new_comment = json.load(f)
        if new_comment['timestamp'] > st.session_state.last_timestamp:
            st.session_state.last_timestamp = new_comment['timestamp']
            st.session_state.comment_history.append(new_comment)
            if len(st.session_state.comment_history) > 50:
                st.session_state.comment_history.pop(0)
            label = new_comment['label']
            st.session_state.label_counts[label] = st.session_state.label_counts.get(label, 0) + 1
    except (FileNotFoundError, json.JSONDecodeError):
        st.info("Waiting for data from the live analyzer...")

    if st.session_state.comment_history:
        latest = st.session_state.comment_history[-1]
        st.subheader("Latest Comment Analyzed")
        st.metric("Predicted Sentiment", latest['label'], f"Confidence: {latest['confidence']}")
        st.markdown(f"**r/{latest['subreddit']}** by u/{latest['author']}")
        st.markdown(f"> {latest['text']}")

        st.subheader("Recent Comments")
        df_history = pd.DataFrame(st.session_state.comment_history).iloc[::-1]
        st.dataframe(df_history[['subreddit', 'author', 'label', 'text']], use_container_width=True)

        st.subheader("Live Sentiment Distribution")
        df_chart = pd.DataFrame(list(st.session_state.label_counts.items()), columns=['Sentiment', 'Count'])
        st.bar_chart(df_chart.set_index('Sentiment'))
    
    time.sleep(2)
    st.rerun()

# ==============================================================================
#                         TAB 2: ON-DEMAND EXPLORER
# ==============================================================================
elif selected_tab == "📖 On-Demand Explorer":
    st.sidebar.header("On-Demand Analysis Controls")
    st.header("Fetch and Analyze Historical Data")
    st.markdown("Select a subreddit and parameters, then fetch data directly from Reddit for analysis.")

    # --- NEW: On-Demand Data Extraction ---
    sub_list = extractor.target_subreddits
    selected_subreddit = st.sidebar.selectbox("1. Select a Subreddit", sub_list)
    
    num_threads = st.sidebar.slider("2. Number of Threads to Fetch", 1, 20, 3)
    min_comments = st.sidebar.slider("3. Minimum Comments per Thread", 5, 50, 10)
    
    if st.sidebar.button("🚀 Fetch Data & Analyze"):
        # Instantiate a new extractor for a fresh run
        on_demand_extractor = ThreadConversationExtractor(REDDIT_CONFIG)
        
        with st.spinner(f"Fetching {num_threads} threads from r/{selected_subreddit}..."):
            on_demand_extractor.target_subreddits = [selected_subreddit] # Focus on the selected subreddit
            total_extracted = on_demand_extractor.extract_all_subreddits(
                threads_per_subreddit=num_threads,
                min_comments=min_comments
            )
        
        if total_extracted > 0 and sentiment_pipeline:
            # Convert the extracted data to a DataFrame for analysis
            csv_filename = on_demand_extractor.save_to_csv_format()
            df = pd.read_csv(csv_filename)
            
            st.success(f"✅ Successfully fetched and loaded {len(df)} posts from {total_extracted} threads.")
            st.subheader("Full Conversation Data")
            st.dataframe(df)

            # Run analysis on the entire fetched dataset
            with st.spinner("Analyzing all fetched comments..."):
                texts = df['text'].dropna().tolist()
                results = sentiment_pipeline(texts, truncation=True, max_length=256)
                df.loc[df['text'].notna(), 'predicted_label'] = [res['label'] for res in results]
            
            # --- NEW: Display Summaries ---
            st.subheader("Analysis Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Overall Thread Sentiment")
                conclusive_sentiment = get_conclusive_sentiment(df)
                st.metric("Conclusive Sentiment", conclusive_sentiment)

            with col2:
                st.markdown("#### Sentiment Breakdown")
                label_counts = df['predicted_label'].value_counts()
                st.dataframe(label_counts)

            st.bar_chart(label_counts)

            # --- NEW: Single User Analysis Section ---
            st.subheader("Drill Down: Single User Analysis")
            author_list = sorted(df['author'].dropna().unique().tolist())
            selected_author = st.selectbox("Select a User to Analyze", author_list)
            
            if selected_author:
                user_df = df[df['author'] == selected_author].copy()
                st.markdown(f"#### Analysis for **{selected_author}**")
                
                user_label_counts = user_df['predicted_label'].value_counts()
                st.metric(f"Total Comments by User", len(user_df))
                
                # Display breakdown of what this user expressed
                st.dataframe(user_label_counts)
                st.bar_chart(user_label_counts)

        elif not sentiment_pipeline:
            st.error("Cannot perform analysis because the custom model is not loaded.")
        else:
            st.error("Failed to fetch any threads. Please try different parameters or check the subreddit.")

# --- NEW: Footer and Disclaimer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-style: italic; color: grey;">
<p><b>Disclaimer:</b> All data and analysis presented here are for educational and research purposes only. The sentiment labels are subjective and generated by a machine learning model. This tool is not a substitute for professional mental health advice.</p>
<p>This project's novelty lies in analyzing the sentiment trajectory of entire conversation threads, rather than isolated, random comments.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
footer {visibility: hidden;}
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: grey;
    text-align: center;
    padding: 10px;
}
</style>
<div class="footer">
    <p>© Shaina 2025</p>
</div>
""", unsafe_allow_html=True)