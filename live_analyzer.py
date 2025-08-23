import praw
import time
import json
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from config import REDDIT_CONFIG

def run_live_analysis():
    """
    Connects to Reddit's live stream, analyzes comments in real-time
    with a custom model, and writes results to a temporary file.
    """
    print("🚀 Starting Real-Time Mental Health Analyzer...")
    print("=" * 50)

    # --- 1. Load Your Custom Model ---
    model_path = "./my_custom_mental_health_model"
    try:
        # --- MODIFICATION START ---
        # Be explicit about loading the model and tokenizer from the local path first
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer
        )
        # --- MODIFICATION END ---
        print(f"✅ Custom sentiment model loaded from '{model_path}'.")
    except Exception as e:
        print(f"❌ Critical Error: Could not load the custom model.")
        print(f"Make sure you have run 'train_model.py' and the folder '{model_path}' exists.")
        print(f"Error details: {e}")
        return

    # --- 2. Connect to Reddit ---
    try:
        reddit = praw.Reddit(**REDDIT_CONFIG)
        subreddits = "depression+anxiety+mentalhealth+bipolar+BipolarReddit+ptsd+socialanxiety+OCD"
        subreddit = reddit.subreddit(subreddits)
        print(f"📡 Connected to Reddit. Listening for new comments...")
    except Exception as e:
        print(f"❌ Critical Error: Could not connect to Reddit.")
        print(f"Please check your 'config.py' file and internet connection.")
        print(f"Error details: {e}")
        return

    # --- 3. Start the Live Stream ---
    try:
        for comment in subreddit.stream.comments(skip_existing=True):
            try:
                analysis_result = sentiment_pipeline(comment.body)[0]
                label = analysis_result['label']
                score = analysis_result['score']
                
                live_data = {
                    "author": comment.author.name if comment.author else "[deleted]",
                    "subreddit": comment.subreddit.display_name,
                    "text": comment.body,
                    "label": label,
                    "confidence": f"{score:.2f}",
                    "timestamp": int(time.time())
                }

                print(f"[{live_data['subreddit']}] New comment by {live_data['author']} -> {live_data['label']}")

                with open("live_data.json", "w") as f:
                    json.dump(live_data, f)
            
            except Exception as e:
                print(f"⚠️ Warning: Could not process a comment. Error: {e}")
                continue

    except KeyboardInterrupt:
        print("\n🛑 Analyzer stopped by user. Goodbye!")
    except Exception as e:
        print(f"❌ An unexpected error occurred in the stream: {e}")


if __name__ == "__main__":
    run_live_analysis()