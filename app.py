# import streamlit as st
# import pandas as pd
# import praw
# import time
# import os
# import glob
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# from thread_conversation_extractor import ThreadConversationExtractor

# # --- Page Configuration ---
# st.set_page_config(page_title="Mental Health Analyzer", layout="wide")

# # --- Model Loading (with caching for performance) ---
# @st.cache_resource
# def load_sentiment_model():
#     # --- FINAL FIX: Use the correct model ID from your Hugging Face Hub ---
#     model_id = "shaina05/mental-health-sentiment-analyzer"
    
#     try:
#         # Load the model directly from the Hub
#         tokenizer = AutoTokenizer.from_pretrained(model_id)
#         model = AutoModelForSequenceClassification.from_pretrained(model_id)
#         return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
#     except Exception as e:
#         # Provide a very clear error message if loading fails
#         st.sidebar.error(f"Fatal Error: Could not load model '{model_id}' from the Hub.")
#         st.sidebar.error("Please ensure the model repository is set to PUBLIC on your Hugging Face profile.")
#         st.sidebar.error(f"Details: {e}")
#         return None

# sentiment_pipeline = load_sentiment_model()

# # --- Main App Structure ---
# st.title("🧠 Mental Health Reddit Analyzer")
# st.sidebar.title("Controls")

# if 'active_tab' not in st.session_state:
#     st.session_state.active_tab = "📈 Recent Activity Monitor"

# cols = st.columns(2)
# if cols[0].button("📈 Recent Activity Monitor"):
#     st.session_state.active_tab = "📈 Recent Activity Monitor"
# if cols[1].button("📖 On-Demand Explorer"):
#     st.session_state.active_tab = "📖 On-Demand Explorer"

# # (The rest of the app.py code is the same as the last working version)
# # ...
# if st.session_state.active_tab == "📈 Recent Activity Monitor":
#     st.sidebar.header("Recent Activity")
#     st.sidebar.info("This dashboard automatically refreshes with the latest comments.")
#     st.header("Analysis of Recent Reddit Comments")

#     if 'recent_comments' not in st.session_state:
#         st.session_state.recent_comments = []

#     if not st.session_state.recent_comments:
#         try:
#             reddit_instance = praw.Reddit(**st.secrets["REDDIT_CONFIG"])
#             if reddit_instance and sentiment_pipeline:
#                 with st.spinner("Fetching and analyzing latest comments..."):
#                     subreddits = "depression+anxiety+mentalhealth+bipolar"
#                     comments = reddit_instance.subreddit(subreddits).comments(limit=25)
                    
#                     comments_data = [{"text": c.body} for c in comments]
#                     texts = [c['text'] for c in comments_data]
#                     results = sentiment_pipeline(texts, truncation=True, max_length=256)
                    
#                     for i, res in enumerate(results):
#                         comments_data[i]['label'] = res['label']
#                     st.session_state.recent_comments = comments_data
#         except Exception as e:
#             st.error(f"Could not fetch live data. Error: {e}")

#     if st.session_state.recent_comments:
#         df_recent = pd.DataFrame(st.session_state.recent_comments)
#         st.dataframe(df_recent, use_container_width=True)
#         st.bar_chart(df_recent['label'].value_counts())

#     st.session_state.recent_comments = []
#     time.sleep(60)
#     st.rerun()

# elif st.session_state.active_tab == "📖 On-Demand Explorer":
#     st.sidebar.header("On-Demand Analysis")
#     st.header("Fetch and Analyze Historical Threads")

#     if 'on_demand_df' not in st.session_state:
#         st.session_state.on_demand_df = None

#     extractor = ThreadConversationExtractor(st.secrets["REDDIT_CONFIG"])
    
#     selected_subreddit = st.sidebar.selectbox("1. Select Subreddit", extractor.target_subreddits)
#     num_threads = st.sidebar.slider("2. Threads to Fetch", 1, 10, 2)
#     min_comments = st.sidebar.slider("3. Min Comments per Thread", 5, 30, 5)
    
#     if st.sidebar.button("🚀 Fetch & Analyze"):
#         if sentiment_pipeline:
#             with st.spinner(f"Fetching threads from r/{selected_subreddit}..."):
#                 extractor.target_subreddits = [selected_subreddit]
#                 total_extracted = extractor.extract_all_subreddits(threads_per_subreddit=num_threads, min_comments=min_comments)
#                 if total_extracted > 0:
#                     csv_file = extractor.save_to_csv_format()
#                     df = pd.read_csv(csv_file)
#                     texts = df['text'].dropna().tolist()
#                     results = sentiment_pipeline(texts, truncation=True, max_length=256)
#                     df.loc[df['text'].notna(), 'predicted_label'] = [res['label'] for res in results]
#                     st.session_state.on_demand_df = df
#                 else:
#                     st.error("Failed to fetch threads.")
#         else:
#             st.error("Cannot analyze because the custom model failed to load. Check sidebar for details.")

#     if st.session_state.on_demand_df is not None:
#         st.subheader("Analysis Results")
#         st.dataframe(st.session_state.on_demand_df)
#         st.bar_chart(st.session_state.on_demand_df['predicted_label'].value_counts())

import streamlit as st
import pandas as pd
import praw
import time
import os
import glob
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from thread_conversation_extractor import ThreadConversationExtractor

# --- Page Configuration ---
st.set_page_config(page_title="Mental Health Analyzer", layout="wide")

# --- Model & Extractor Loading (Cached for performance) ---
@st.cache_resource
def load_sentiment_model():
    # Load your model from the Hugging Face Hub
    model_id = "shaina05/mental-health-sentiment-analyzer" # Uses your correct username
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.sidebar.error(f"Error loading model from Hub: {e}")
        return None

@st.cache_resource
def get_extractor():
    # Load credentials from Streamlit's secrets manager
    return ThreadConversationExtractor(st.secrets["REDDIT_CONFIG"])

sentiment_pipeline = load_sentiment_model()
extractor = get_extractor()

# --- Main App UI ---
st.title("🧠 Mental Health Reddit Analyzer")
st.sidebar.title("Controls")

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "📈 Recent Activity Monitor"

cols = st.columns(2)
if cols[0].button("📈 Recent Activity Monitor"):
    st.session_state.active_tab = "📈 Recent Activity Monitor"
if cols[1].button("📖 On-Demand Explorer"):
    st.session_state.active_tab = "📖 On-Demand Explorer"

# ==============================================================================
#                             TAB 1: RECENT ACTIVITY MONITOR
# ==============================================================================
if st.session_state.active_tab == "📈 Recent Activity Monitor":
    # (This section remains the same)
    st.sidebar.header("Recent Activity")
    st.sidebar.info("This dashboard automatically refreshes with the latest comments.")
    st.header("Analysis of Recent Reddit Comments")

    if 'recent_comments' not in st.session_state:
        st.session_state.recent_comments = []

    if not st.session_state.recent_comments:
        try:
            reddit_instance = praw.Reddit(**st.secrets["REDDIT_CONFIG"])
            if reddit_instance and sentiment_pipeline:
                with st.spinner("Fetching and analyzing latest comments..."):
                    subreddits = "depression+anxiety+mentalhealth+bipolar"
                    comments = reddit_instance.subreddit(subreddits).comments(limit=25)
                    
                    comments_data = [{"text": c.body, "author": c.author.name if c.author else "[deleted]", "subreddit": c.subreddit.display_name} for c in comments]
                    texts = [c['text'] for c in comments_data]
                    results = sentiment_pipeline(texts, truncation=True, max_length=256)
                    
                    for i, res in enumerate(results):
                        comments_data[i]['label'] = res['label']
                    st.session_state.recent_comments = comments_data
        except Exception as e:
            st.error(f"Could not fetch live data. Error: {e}")

    if st.session_state.recent_comments:
        df_recent = pd.DataFrame(st.session_state.recent_comments)
        st.dataframe(df_recent, use_container_width=True)
        st.bar_chart(df_recent['label'].value_counts())

    st.session_state.recent_comments = []
    time.sleep(60)
    st.rerun()

# ==============================================================================
#                         TAB 2: ON-DEMAND EXPLORER
# ==============================================================================
elif st.session_state.active_tab == "📖 On-Demand Explorer":
    st.sidebar.header("On-Demand Analysis")
    st.header("Fetch and Analyze Historical Threads")

    if 'on_demand_df' not in st.session_state:
        st.session_state.on_demand_df = None

    selected_subreddit = st.sidebar.selectbox("1. Select Subreddit", extractor.target_subreddits)
    num_threads = st.sidebar.slider("2. Threads to Fetch", 1, 10, 2)
    min_comments = st.sidebar.slider("3. Min Comments per Thread", 5, 30, 5)
    
    if st.sidebar.button("🚀 Fetch Data"):
        st.session_state.on_demand_df = None # Clear previous results
        with st.spinner(f"Fetching threads from r/{selected_subreddit}..."):
            extractor.target_subreddits = [selected_subreddit]
            total_extracted = extractor.extract_all_subreddits(threads_per_subreddit=num_threads, min_comments=min_comments)
            if total_extracted > 0:
                csv_file = extractor.save_to_csv_format()
                st.session_state.on_demand_df = pd.read_csv(csv_file)
            else:
                st.error("Failed to fetch threads.")

    if st.session_state.on_demand_df is not None:
        df = st.session_state.on_demand_df
        st.subheader("Fetched Conversation Data")
        st.dataframe(df)

        if sentiment_pipeline:
            st.subheader("Perform Analysis")
            # --- MODIFICATION START: Re-introducing the analysis type choice ---
            analysis_type = st.radio("Choose Analysis Type:", ("Full Thread", "Single User"), horizontal=True)

            if analysis_type == "Full Thread":
                if st.button("Analyze Displayed Data"):
                    with st.spinner("Analyzing sentiments..."):
                        texts = df['text'].dropna().tolist()
                        results = sentiment_pipeline(texts, truncation=True, max_length=256)
                        df.loc[df['text'].notna(), 'predicted_label'] = [res['label'] for res in results]
                        st.subheader("Analysis Results")
                        st.dataframe(df[['author', 'position', 'predicted_label', 'text']])
                        st.subheader("Sentiment Distribution")
                        st.bar_chart(df['predicted_label'].value_counts())
            
            elif analysis_type == "Single User":
                author_list = sorted(df['author'].dropna().unique().tolist())
                selected_author = st.selectbox("Select a User to Analyze", author_list)
                if st.button("Analyze Selected User"):
                    user_df = df[df['author'] == selected_author].copy()
                    with st.spinner(f"Analyzing comments from {selected_author}..."):
                        user_texts = user_df['text'].dropna().tolist()
                        if user_texts:
                            user_results = sentiment_pipeline(user_texts, truncation=True, max_length=256)
                            user_df.loc[user_df['text'].notna(), 'predicted_label'] = [res['label'] for res in user_results]
                            st.subheader(f"Sentiment Trajectory for {selected_author}")
                            st.dataframe(user_df[['position', 'predicted_label', 'text']])
                            st.bar_chart(user_df['predicted_label'].value_counts())
                        else:
                            st.warning("Selected user has no comments with text to analyze.")
            # --- MODIFICATION END ---
        else:
            st.error("Cannot perform analysis because the custom model failed to load.")