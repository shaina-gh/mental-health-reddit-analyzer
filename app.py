import streamlit as st
import pandas as pd
import praw
import time
import os
import glob
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
def get_reddit_instance():
    try:
        return praw.Reddit(**REDDIT_CONFIG)
    except Exception as e:
        st.error(f"Failed to connect to Reddit: {e}")
        return None

sentiment_pipeline = load_sentiment_model()
reddit = get_reddit_instance()

# --- Main App Structure ---
st.title("🧠 Mental Health Reddit Analyzer")
st.sidebar.title("Controls")

selected_tab = st.radio("Select View", ["📈 Recent Activity Monitor", "📖 On-Demand Explorer"], horizontal=True, label_visibility="collapsed")

# ==============================================================================
#                             TAB 1: RECENT ACTIVITY MONITOR
# ==============================================================================
if selected_tab == "📈 Recent Activity Monitor":
    st.sidebar.header("Recent Activity")
    st.sidebar.info("This dashboard automatically refreshes with the latest comments from Reddit every 60 seconds.")
    st.header("Analysis of Recent Reddit Comments")

    if 'recent_comments' not in st.session_state:
        st.session_state.recent_comments = []

    # Fetch new data only if the session state is empty
    if not st.session_state.recent_comments:
        if reddit and sentiment_pipeline:
            with st.spinner("Fetching and analyzing latest comments..."):
                subreddits = "depression+anxiety+mentalhealth+bipolar+BipolarReddit+ptsd+socialanxiety+OCD"
                latest_comments_generator = reddit.subreddit(subreddits).comments(limit=25)
                
                comments_to_analyze = []
                for comment in latest_comments_generator:
                    comments_to_analyze.append({
                        "author": comment.author.name if comment.author else "[deleted]",
                        "subreddit": comment.subreddit.display_name,
                        "text": comment.body
                    })

                texts = [c['text'] for c in comments_to_analyze]
                results = sentiment_pipeline(texts, truncation=True, max_length=256)

                for i, comment in enumerate(comments_to_analyze):
                    comment['label'] = results[i]['label']
                    comment['confidence'] = f"{results[i]['score']:.2f}"
                
                st.session_state.recent_comments = comments_to_analyze

    if st.session_state.recent_comments:
        st.subheader("Latest Comments Analyzed")
        df_recent = pd.DataFrame(st.session_state.recent_comments).iloc[::-1]
        st.dataframe(df_recent, use_container_width=True)

        st.subheader("Sentiment Distribution")
        st.bar_chart(df_recent['label'].value_counts())
    else:
        st.warning("Could not fetch any recent comments. This may be due to a Reddit API issue.")

    # Clear the state to force a refresh on the next run
    st.session_state.recent_comments = []
    time.sleep(60) # Wait 60 seconds before the next refresh
    st.rerun()

# ==============================================================================
#                         TAB 2: ON-DEMAND EXPLORER
# ==============================================================================
elif selected_tab == "📖 On-Demand Explorer":
    st.sidebar.header("On-Demand Analysis Controls")
    st.header("Fetch and Analyze Historical Data")
    
    if 'on_demand_df' not in st.session_state:
        st.session_state.on_demand_df = None

    extractor = ThreadConversationExtractor(REDDIT_CONFIG)
    sub_list = extractor.target_subreddits
    selected_subreddit = st.sidebar.selectbox("1. Select a Subreddit", sub_list)
    num_threads = st.sidebar.slider("2. Number of Threads to Fetch", 1, 10, 2)
    min_comments = st.sidebar.slider("3. Minimum Comments per Thread", 5, 30, 5)
    
    if st.sidebar.button("🚀 Fetch Data"):
        with st.spinner(f"Fetching {num_threads} threads from r/{selected_subreddit}..."):
            extractor.target_subreddits = [selected_subreddit]
            total_extracted = extractor.extract_all_subreddits(threads_per_subreddit=num_threads, min_comments=min_comments)
            if total_extracted > 0:
                csv_filename = extractor.save_to_csv_format()
                st.session_state.on_demand_df = pd.read_csv(csv_filename)
            else:
                st.session_state.on_demand_df = None
                st.error("Failed to fetch any threads.")

    if st.session_state.on_demand_df is not None:
        df = st.session_state.on_demand_df
        st.subheader("Fetched Conversation Data")
        st.dataframe(df)

        if st.button("Analyze Displayed Data"):
            with st.spinner("Analyzing sentiments..."):
                texts = df['text'].dropna().tolist()
                results = sentiment_pipeline(texts, truncation=True, max_length=256)
                df.loc[df['text'].notna(), 'predicted_label'] = [res['label'] for res in results]
                st.subheader("Analysis Results")
                st.dataframe(df[['author', 'position', 'predicted_label', 'text']])
                st.subheader("Sentiment Distribution")
                st.bar_chart(df['predicted_label'].value_counts())