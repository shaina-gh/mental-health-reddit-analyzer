#!/usr/bin/env python3
"""
Mental Health Discussion Scraper & Sentiment Analysis Test Program
Project: Sentiment Trajectory Modelling in Mental Health Discussions
Team: Joel Prince, Shaina, Akshaya
"""

import praw
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# For transformer-based analysis (optional - install if needed)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Using TextBlob for sentiment analysis.")

class MentalHealthScraper:
    def __init__(self, reddit_config=None):
        """
        Initialize the scraper with Reddit API credentials
        
        To get credentials:
        1. Go to https://www.reddit.com/prefs/apps
        2. Create a new app (script type)
        3. Get client_id, client_secret, and set user_agent
        """
        if reddit_config:
            self.reddit = praw.Reddit(
                client_id=reddit_config['client_id'],
                client_secret=reddit_config['client_secret'],
                user_agent=reddit_config['user_agent']
            )
        else:
            self.reddit = None
            print("No Reddit config provided. Using sample data mode.")
        
        # Initialize sentiment analyzer
        self.setup_sentiment_analyzer()
        
        # Mental health subreddits to scrape
        self.target_subreddits = [
            'depression', 'anxiety', 'mentalhealth', 'bipolar', 
            'BipolarReddit', 'ptsd', 'socialanxiety', 'OCD'
        ]
        
        self.collected_data = []
        self.processed_data = pd.DataFrame()
    
    def setup_sentiment_analyzer(self):
        """Setup sentiment analysis pipeline"""
        if TRANSFORMERS_AVAILABLE:
            try:
                # Try to load a mental health specific model or general sentiment model
                model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis", 
                    model=model_name,
                    tokenizer=model_name
                )
                self.sentiment_method = "transformer"
                print(f"Using transformer model: {model_name}")
            except Exception as e:
                print(f"Failed to load transformer model: {e}")
                self.sentiment_method = "textblob"
        else:
            self.sentiment_method = "textblob"
    
    def generate_sample_data(self, num_threads=20, posts_per_thread=15):
        """Generate sample mental health discussion data for testing"""
        print("Generating sample mental health discussion data...")
        
        sample_topics = [
            "struggling with depression", "anxiety attacks", "therapy sessions",
            "medication side effects", "support group experience", "coping strategies",
            "bad days", "progress update", "relapse story", "recovery journey"
        ]
        
        # Sample text patterns for different sentiment trajectories
        positive_phrases = [
            "feeling better today", "therapy helped", "making progress",
            "grateful for support", "small victory", "hopeful about future",
            "getting stronger", "learned coping skills"
        ]
        
        negative_phrases = [
            "feeling overwhelmed", "can't handle this", "getting worse",
            "lost hope", "everything is falling apart", "no one understands",
            "giving up", "tired of fighting"
        ]
        
        neutral_phrases = [
            "just wanted to share", "update on my situation", "looking for advice",
            "has anyone experienced", "wondering if this is normal",
            "scheduled appointment", "reading about this topic"
        ]
        
        for thread_id in range(num_threads):
            thread_data = {
                'thread_id': f'sample_thread_{thread_id}',
                'subreddit': np.random.choice(self.target_subreddits),
                'topic': np.random.choice(sample_topics),
                'posts': []
            }
            
            # Generate sentiment trajectory (recovery, decline, stable, mixed)
            trajectory_type = np.random.choice(['recovery', 'decline', 'stable', 'mixed'])
            
            for post_idx in range(posts_per_thread):
                # Create sentiment based on trajectory and position
                if trajectory_type == 'recovery':
                    sentiment_prob = min(0.8, 0.2 + (post_idx / posts_per_thread) * 0.6)
                elif trajectory_type == 'decline':
                    sentiment_prob = max(0.1, 0.7 - (post_idx / posts_per_thread) * 0.6)
                elif trajectory_type == 'stable':
                    sentiment_prob = 0.4 + np.random.normal(0, 0.1)
                else:  # mixed
                    sentiment_prob = 0.5 + np.random.normal(0, 0.3)
                
                # Generate post content based on sentiment
                if sentiment_prob > 0.6:
                    base_content = np.random.choice(positive_phrases)
                    true_sentiment = 'positive'
                elif sentiment_prob < 0.4:
                    base_content = np.random.choice(negative_phrases)
                    true_sentiment = 'negative'
                else:
                    base_content = np.random.choice(neutral_phrases)
                    true_sentiment = 'neutral'
                
                post_data = {
                    'post_id': f'post_{thread_id}_{post_idx}',
                    'user_id': f'user_{thread_id % 5}',  # 5 users across threads
                    'content': f"{base_content}. Post {post_idx + 1} in thread about {thread_data['topic']}.",
                    'timestamp': datetime.now() - timedelta(days=30-post_idx, hours=np.random.randint(0, 24)),
                    'thread_position': post_idx,
                    'true_sentiment': true_sentiment,
                    'trajectory_type': trajectory_type
                }
                
                thread_data['posts'].append(post_data)
            
            self.collected_data.append(thread_data)
        
        print(f"Generated {num_threads} sample threads with {num_threads * posts_per_thread} total posts")
        return self.collected_data
    
    def scrape_reddit_data(self, max_posts_per_subreddit=50):
        """Scrape real data from Reddit (requires API credentials)"""
        if not self.reddit:
            print("Reddit API not configured. Use generate_sample_data() instead.")
            return None
        
        print("Starting Reddit data collection...")
        
        for subreddit_name in self.target_subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                print(f"Scraping r/{subreddit_name}...")
                
                # Get hot posts from subreddit
                for submission in subreddit.hot(limit=max_posts_per_subreddit):
                    if submission.num_comments > 3:  # Only threads with discussion
                        thread_data = {
                            'thread_id': submission.id,
                            'subreddit': subreddit_name,
                            'topic': submission.title,
                            'posts': []
                        }
                        
                        # Get original post
                        thread_data['posts'].append({
                            'post_id': submission.id,
                            'user_id': str(submission.author) if submission.author else 'deleted',
                            'content': submission.selftext if submission.selftext else submission.title,
                            'timestamp': datetime.fromtimestamp(submission.created_utc),
                            'thread_position': 0,
                            'score': submission.score
                        })
                        
                        # Get comments
                        submission.comments.replace_more(limit=0)
                        for idx, comment in enumerate(submission.comments[:20]):  # Limit comments
                            thread_data['posts'].append({
                                'post_id': comment.id,
                                'user_id': str(comment.author) if comment.author else 'deleted',
                                'content': comment.body,
                                'timestamp': datetime.fromtimestamp(comment.created_utc),
                                'thread_position': idx + 1,
                                'score': comment.score
                            })
                        
                        self.collected_data.append(thread_data)
                        time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Error scraping r/{subreddit_name}: {e}")
                continue
        
        print(f"Collected {len(self.collected_data)} threads from Reddit")
        return self.collected_data
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and subreddit references
        text = re.sub(r'u/\S+|r/\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short texts
        if len(text) < 10:
            return ''
        
        return text
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text using available method"""
        if not text or len(text.strip()) < 3:
            return {'sentiment': 'neutral', 'confidence': 0.0}
        
        try:
            if self.sentiment_method == "transformer":
                result = self.sentiment_pipeline(text)[0]
                return {
                    'sentiment': result['label'].lower(),
                    'confidence': result['score']
                }
            else:
                # Using TextBlob
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    sentiment = 'positive'
                elif polarity < -0.1:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                return {
                    'sentiment': sentiment,
                    'confidence': abs(polarity)
                }
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.0}
    
    def process_data(self):
        """Process collected data into structured format"""
        print("Processing collected data...")
        
        processed_posts = []
        
        for thread in self.collected_data:
            for post in thread['posts']:
                # Preprocess text
                cleaned_content = self.preprocess_text(post['content'])
                
                if cleaned_content:  # Only process non-empty content
                    # Analyze sentiment
                    sentiment_result = self.analyze_sentiment(cleaned_content)
                    
                    processed_post = {
                        'thread_id': thread['thread_id'],
                        'subreddit': thread.get('subreddit', 'unknown'),
                        'topic': thread.get('topic', 'unknown'),
                        'post_id': post['post_id'],
                        'user_id': post['user_id'],
                        'content': cleaned_content,
                        'content_length': len(cleaned_content),
                        'timestamp': post['timestamp'],
                        'thread_position': post['thread_position'],
                        'predicted_sentiment': sentiment_result['sentiment'],
                        'sentiment_confidence': sentiment_result['confidence'],
                        'score': post.get('score', 0)
                    }
                    
                    # Add ground truth if available (from sample data)
                    if 'true_sentiment' in post:
                        processed_post['true_sentiment'] = post['true_sentiment']
                        processed_post['trajectory_type'] = post['trajectory_type']
                    
                    processed_posts.append(processed_post)
        
        self.processed_data = pd.DataFrame(processed_posts)
        print(f"Processed {len(self.processed_data)} posts")
        
        return self.processed_data
    
    def analyze_sentiment_trajectories(self):
        """Analyze sentiment trajectories within threads"""
        print("Analyzing sentiment trajectories...")
        
        trajectory_analysis = {}
        
        for thread_id in self.processed_data['thread_id'].unique():
            thread_data = self.processed_data[
                self.processed_data['thread_id'] == thread_id
            ].sort_values('thread_position')
            
            if len(thread_data) > 2:  # Only analyze threads with multiple posts
                # Convert sentiment to numeric for trajectory analysis
                sentiment_mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
                thread_data['sentiment_numeric'] = thread_data['predicted_sentiment'].map(sentiment_mapping)
                
                # Calculate trajectory metrics
                sentiment_trend = np.polyfit(thread_data['thread_position'], 
                                           thread_data['sentiment_numeric'], 1)[0]
                sentiment_variance = thread_data['sentiment_numeric'].var()
                
                trajectory_analysis[thread_id] = {
                    'thread_length': len(thread_data),
                    'sentiment_trend': sentiment_trend,  # Positive = improving, Negative = declining
                    'sentiment_variance': sentiment_variance,
                    'start_sentiment': thread_data.iloc[0]['predicted_sentiment'],
                    'end_sentiment': thread_data.iloc[-1]['predicted_sentiment'],
                    'sentiment_sequence': thread_data['predicted_sentiment'].tolist(),
                    'confidence_sequence': thread_data['sentiment_confidence'].tolist()
                }
        
        return trajectory_analysis
    
    def visualize_results(self, save_plots=True):
        """Create visualizations of the analysis results"""
        print("Creating visualizations...")
        
        if self.processed_data.empty:
            print("No processed data available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Mental Health Discussion Analysis Results', fontsize=16)
        
        # 1. Sentiment Distribution
        sentiment_counts = self.processed_data['predicted_sentiment'].value_counts()
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Overall Sentiment Distribution')
        
        # 2. Sentiment by Subreddit
        if len(self.processed_data['subreddit'].unique()) > 1:
            sentiment_by_sub = pd.crosstab(self.processed_data['subreddit'], 
                                         self.processed_data['predicted_sentiment'])
            sentiment_by_sub.plot(kind='bar', stacked=True, ax=axes[0, 1])
            axes[0, 1].set_title('Sentiment Distribution by Subreddit')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Thread Position vs Sentiment
        sentiment_mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
        self.processed_data['sentiment_numeric'] = self.processed_data['predicted_sentiment'].map(sentiment_mapping)
        
        axes[1, 0].scatter(self.processed_data['thread_position'], 
                          self.processed_data['sentiment_numeric'], alpha=0.6)
        axes[1, 0].set_xlabel('Thread Position')
        axes[1, 0].set_ylabel('Sentiment Score')
        axes[1, 0].set_title('Sentiment vs Thread Position')
        
        # 4. Sample Trajectory
        sample_threads = self.processed_data['thread_id'].unique()[:5]
        for thread_id in sample_threads:
            thread_data = self.processed_data[
                self.processed_data['thread_id'] == thread_id
            ].sort_values('thread_position')
            
            if len(thread_data) > 2:
                axes[1, 1].plot(thread_data['thread_position'], 
                               thread_data['sentiment_numeric'], 
                               marker='o', label=f'Thread {thread_id[:8]}')
        
        axes[1, 1].set_xlabel('Thread Position')
        axes[1, 1].set_ylabel('Sentiment Score')
        axes[1, 1].set_title('Sample Sentiment Trajectories')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('mental_health_analysis_results.png', dpi=300, bbox_inches='tight')
            print("Plots saved as 'mental_health_analysis_results.png'")
        
        plt.show()
    
    def generate_report(self):
        """Generate analysis report"""
        if self.processed_data.empty:
            print("No processed data available for report generation")
            return
        
        print("\n" + "="*60)
        print("MENTAL HEALTH DISCUSSION ANALYSIS REPORT")
        print("="*60)
        
        # Basic statistics
        total_posts = len(self.processed_data)
        total_threads = self.processed_data['thread_id'].nunique()
        total_users = self.processed_data['user_id'].nunique()
        
        print(f"\nDataset Overview:")
        print(f"  Total Posts: {total_posts}")
        print(f"  Total Threads: {total_threads}")
        print(f"  Total Users: {total_users}")
        print(f"  Average Posts per Thread: {total_posts/total_threads:.1f}")
        
        # Sentiment distribution
        sentiment_dist = self.processed_data['predicted_sentiment'].value_counts(normalize=True)
        print(f"\nSentiment Distribution:")
        for sentiment, pct in sentiment_dist.items():
            print(f"  {sentiment.capitalize()}: {pct:.1%}")
        
        # Trajectory analysis
        trajectory_data = self.analyze_sentiment_trajectories()
        improving_threads = sum(1 for t in trajectory_data.values() if t['sentiment_trend'] > 0.1)
        declining_threads = sum(1 for t in trajectory_data.values() if t['sentiment_trend'] < -0.1)
        stable_threads = len(trajectory_data) - improving_threads - declining_threads
        
        print(f"\nTrajectory Analysis:")
        print(f"  Improving Threads: {improving_threads}")
        print(f"  Declining Threads: {declining_threads}")
        print(f"  Stable Threads: {stable_threads}")
        
        # Sample trajectory examples
        print(f"\nSample Trajectory Examples:")
        for i, (thread_id, data) in enumerate(list(trajectory_data.items())[:3]):
            print(f"  Thread {i+1}: {' → '.join(data['sentiment_sequence'][:5])}")
            if data['sentiment_trend'] > 0.1:
                print(f"    Pattern: Improving (trend: +{data['sentiment_trend']:.3f})")
            elif data['sentiment_trend'] < -0.1:
                print(f"    Pattern: Declining (trend: {data['sentiment_trend']:.3f})")
            else:
                print(f"    Pattern: Stable (trend: {data['sentiment_trend']:.3f})")
        
        print("\n" + "="*60)


def main():
    """Main function to run the test program"""
    print("Mental Health Discussion Scraper & Sentiment Analysis Test")
    print("=========================================================")
    
    # Option 1: Using sample data (recommended for testing)
    print("\n1. Testing with generated sample data...")
    scraper = MentalHealthScraper()  # No Reddit config for sample data
    
    # Generate sample data
    scraper.generate_sample_data(num_threads=25, posts_per_thread=12)
    
    # Process the data
    processed_data = scraper.process_data()
    
    # Analyze trajectories
    trajectories = scraper.analyze_sentiment_trajectories()
    
    # Generate visualizations
    scraper.visualize_results()
    
    # Generate report
    scraper.generate_report()
    
    # Option 2: Real Reddit data (uncomment and configure to use)
    """
    print("\n2. Testing with real Reddit data...")
    reddit_config = {
        'client_id': 'your_client_id',
        'client_secret': 'your_client_secret',
        'user_agent': 'MentalHealthAnalyzer/1.0 by YourUsername'
    }
    
    real_scraper = MentalHealthScraper(reddit_config)
    real_scraper.scrape_reddit_data(max_posts_per_subreddit=10)
    real_processed_data = real_scraper.process_data()
    real_scraper.visualize_results()
    real_scraper.generate_report()
    """
    
    print("\nTest completed! Check the generated visualizations and report.")
    return processed_data, trajectories

if __name__ == "__main__":
    # Install required packages if not available
    required_packages = [
        'praw', 'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'textblob', 'transformers', 'torch'
    ]
    
    print("Required packages:", ", ".join(required_packages))
    print("Install with: pip install " + " ".join(required_packages))
    print("\nNote: transformers and torch are optional but recommended for better sentiment analysis")
    print("\n" + "="*60 + "\n")
    
    # Run the main program
    data, trajectories = main()