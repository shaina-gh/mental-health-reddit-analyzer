
import praw
import pandas as pd
import json
import time
from datetime import datetime
import os
# from config import REDDIT_CONFIG

class ThreadConversationExtractor:
    def __init__(self, reddit_config): # It now receives the config as an argument
        """Initialize Reddit thread extractor"""
        # It uses the passed-in config to connect to Reddit
        self.reddit = praw.Reddit(**reddit_config)
        self.target_subreddits = [
            'depression', 'anxiety', 'mentalhealth', 'bipolar', 
            'BipolarReddit', 'ptsd', 'socialanxiety', 'OCD'
        ]
        self.extracted_threads = []
        
    def extract_complete_thread(self, submission):
        """Extract a complete thread with all comments in chronological order"""
        try:
            # Get submission details
            thread_data = {
                'thread_id': submission.id,
                'subreddit': submission.subreddit.display_name,
                'title': submission.title,
                'original_post': {
                    'author': str(submission.author) if submission.author else '[deleted]',
                    'text': submission.selftext,
                    'score': submission.score,
                    'created_utc': submission.created_utc,
                    'timestamp': datetime.fromtimestamp(submission.created_utc).isoformat(),
                    'position': 0  # Original post is position 0
                },
                'url': f"https://reddit.com{submission.permalink}",
                'total_comments': submission.num_comments,
                'conversation_flow': []  # This will contain the chronological conversation
            }
            
            # Extract all comments
            submission.comments.replace_more(limit=None)  # Load all comments
            all_comments = submission.comments.list()
            
            # Sort comments by creation time to get chronological order
            all_comments.sort(key=lambda x: x.created_utc)
            
            position = 1  # Start from 1 (original post is 0)
            for comment in all_comments:
                if hasattr(comment, 'body') and comment.body != '[deleted]' and comment.body != '[removed]':
                    comment_data = {
                        'comment_id': comment.id,
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'text': comment.body,
                        'score': comment.score,
                        'created_utc': comment.created_utc,
                        'timestamp': datetime.fromtimestamp(comment.created_utc).isoformat(),
                        'position': position,  # Position in conversation
                        'parent_id': comment.parent_id,
                        'is_reply_to_op': comment.parent_id == f"t3_{submission.id}"  # Direct reply to original post
                    }
                    thread_data['conversation_flow'].append(comment_data)
                    position += 1
            
            # Calculate conversation statistics
            thread_data['conversation_stats'] = {
                'total_participants': len(set([thread_data['original_post']['author']] + 
                                               [c['author'] for c in thread_data['conversation_flow']])),
                'total_exchanges': len(thread_data['conversation_flow']) + 1,  # +1 for original post
                'conversation_duration_hours': (
                    (max([c['created_utc'] for c in thread_data['conversation_flow']] + [submission.created_utc]) - 
                     min([c['created_utc'] for c in thread_data['conversation_flow']] + [submission.created_utc])) / 3600
                ) if thread_data['conversation_flow'] else 0
            }
            
            return thread_data
            
        except Exception as e:
            print(f"⚠️ Error extracting thread {submission.id}: {str(e)}")
            return None
    
    def extract_threads_from_subreddit(self, subreddit_name, thread_limit=10, min_comments=5):
        """Extract threads from a specific subreddit with conversation flow"""
        print(f"🔍 Extracting threads from r/{subreddit_name}...")
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            threads_extracted = 0
            
            # Get recent posts with good discussion
            for submission in subreddit.hot(limit=50):  # Check more posts to find good threads
                # Only extract threads with meaningful conversations
                if submission.num_comments >= min_comments and not submission.stickied:
                    print(f"    📝 Extracting: '{submission.title[:60]}...' ({submission.num_comments} comments)")
                    
                    thread_data = self.extract_complete_thread(submission)
                    if thread_data and thread_data['conversation_flow']:
                        self.extracted_threads.append(thread_data)
                        threads_extracted += 1
                        
                        if threads_extracted >= thread_limit:
                            break
                    
                    # Rate limiting - be respectful to Reddit's servers
                    time.sleep(1)
            
            print(f"✅ Extracted {threads_extracted} threads from r/{subreddit_name}")
            return threads_extracted
            
        except Exception as e:
            print(f"❌ Error accessing r/{subreddit_name}: {str(e)}")
            return 0
    
    def extract_all_subreddits(self, threads_per_subreddit=5, min_comments=5):
        """Extract threads from all target subreddits"""
        print("🚀 STARTING THREAD EXTRACTION")
        print("=" * 60)
        print(f"📍 Target subreddits: {', '.join(self.target_subreddits)}")
        print(f"🎯 Threads per subreddit: {threads_per_subreddit}")
        print(f"💬 Minimum comments per thread: {min_comments}")
        print()
        
        total_extracted = 0
        for subreddit_name in self.target_subreddits:
            extracted = self.extract_threads_from_subreddit(
                subreddit_name, 
                thread_limit=threads_per_subreddit, 
                min_comments=min_comments
            )
            total_extracted += extracted
            time.sleep(2)  # Respectful delay between subreddits
        
        print(f"\n✅ EXTRACTION COMPLETE: {total_extracted} total threads extracted")
        return total_extracted
    
    def save_to_json(self, filename=None):
        """Save extracted threads to JSON format"""
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(output_dir, f'mental_health_threads_{timestamp}.json')
        
        export_data = {
            'extraction_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_threads': len(self.extracted_threads),
                'subreddits': self.target_subreddits,
                'total_conversations': sum(len(t['conversation_flow']) for t in self.extracted_threads),
                'extraction_purpose': 'Sequential conversation sentiment analysis'
            },
            'threads': self.extracted_threads
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 JSON data saved to: {filename}")
        return filename
    
    def save_to_csv_format(self, filename=None):
        """Save threads in CSV format optimized for conversation analysis"""
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)

        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(output_dir, f'mental_health_conversations_{timestamp}.csv')
        
        conversation_rows = []
        
        for thread in self.extracted_threads:
            thread_id = thread['thread_id']
            subreddit = thread['subreddit']
            
            conversation_rows.append({
                'thread_id': thread_id, 'subreddit': subreddit, 'thread_title': thread['title'], 'position': 0, 'message_type': 'original_post',
                'author': thread['original_post']['author'], 'text': thread['original_post']['text'], 'timestamp': thread['original_post']['timestamp'],
                'score': thread['original_post']['score'], 'is_reply_to_op': False, 'total_participants': thread['conversation_stats']['total_participants'],
                'total_exchanges': thread['conversation_stats']['total_exchanges'], 'conversation_duration_hours': thread['conversation_stats']['conversation_duration_hours']
            })
            
            for comment in thread['conversation_flow']:
                conversation_rows.append({
                    'thread_id': thread_id, 'subreddit': subreddit, 'thread_title': thread['title'], 'position': comment['position'], 'message_type': 'comment',
                    'author': comment['author'], 'text': comment['text'], 'timestamp': comment['timestamp'], 'score': comment['score'],
                    'is_reply_to_op': comment['is_reply_to_op'], 'total_participants': thread['conversation_stats']['total_participants'],
                    'total_exchanges': thread['conversation_stats']['total_exchanges'], 'conversation_duration_hours': thread['conversation_stats']['conversation_duration_hours']
                })
        
        df = pd.DataFrame(conversation_rows)
        df.to_csv(filename, index=False, encoding='utf-8')
        
        print(f"💾 CSV data saved to: {filename}")
        return filename
    
    def generate_summary_report(self):
        """Generate summary report of extracted data"""
        if not self.extracted_threads:
            print("❌ No threads extracted yet!")
            return
        
        print("\n" + "=" * 60)
        print("📊 THREAD EXTRACTION SUMMARY REPORT")
        print("=" * 60)
        
        total_conversations = sum(len(t['conversation_flow']) for t in self.extracted_threads)
        all_participants = set()
        for t in self.extracted_threads:
            all_participants.add(t['original_post']['author'])
            for c in t['conversation_flow']:
                all_participants.add(c['author'])

        print(f"📈 Dataset Overview:")
        print(f"    • Total Threads Extracted: {len(self.extracted_threads)}")
        print(f"    • Total Conversation Messages: {total_conversations + len(self.extracted_threads)}")
        print(f"    • Average Messages per Thread: {(total_conversations + len(self.extracted_threads))/len(self.extracted_threads):.1f}")
        print(f"    • Total Unique Participants: {len(all_participants)}")
        
        subreddit_counts = {}
        for thread in self.extracted_threads:
            subreddit = thread['subreddit']
            subreddit_counts[subreddit] = subreddit_counts.get(subreddit, 0) + 1
        
        print(f"\n📍 Threads by Subreddit:")
        for subreddit, count in sorted(subreddit_counts.items()):
            print(f"    • r/{subreddit}: {count} threads")
        
        conversation_lengths = [len(t['conversation_flow']) for t in self.extracted_threads]
        print(f"\n💬 Conversation Length Statistics:")
        print(f"    • Shortest conversation: {min(conversation_lengths)} messages")
        print(f"    • Longest conversation: {max(conversation_lengths)} messages")
        print(f"    • Average conversation: {sum(conversation_lengths)/len(conversation_lengths):.1f} messages")
        
        print(f"\n📝 Sample Thread Examples:")
        for i, thread in enumerate(self.extracted_threads[:3]):
            print(f"    Thread {i+1}: r/{thread['subreddit']}")
            print(f"      Title: {thread['title'][:60]}...")
            print(f"      Messages: {len(thread['conversation_flow'])} exchanges")
            print(f"      Participants: {thread['conversation_stats']['total_participants']}")
            print()

# --- MODIFICATION START ---
# The main function is now non-interactive and uses default values.
def main():
    """Main function to extract conversation threads with default settings."""
    print("🧵 REDDIT CONVERSATION THREAD EXTRACTOR")
    print("=" * 60)
    
    # Set default values directly instead of asking for user input
    threads_per_subreddit = 5
    min_comments = 10
    
    print("Running with default settings:")
    print(f"  - Threads per subreddit: {threads_per_subreddit}")
    print(f"  - Minimum comments per thread: {min_comments}")
    
    extractor = ThreadConversationExtractor(REDDIT_CONFIG)
    
    total_extracted = extractor.extract_all_subreddits(
        threads_per_subreddit=threads_per_subreddit,
        min_comments=min_comments
    )
    
    if total_extracted > 0:
        extractor.generate_summary_report()
        print(f"\n💾 SAVING EXTRACTED DATA")
        json_file = extractor.save_to_json()
        csv_file = extractor.save_to_csv_format()
        print(f"\n✅ DATA EXTRACTION COMPLETE!")
        print(f"📁 Files created: {json_file}, {csv_file}")
    else:
        print("❌ No threads were extracted. Check your Reddit API connection.")
# --- MODIFICATION END ---

if __name__ == "__main__":
    main()