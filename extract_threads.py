#!/usr/bin/env python3
"""
Reddit Thread Extractor Runner
Extract complete conversation threads for sequential sentiment analysis
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Simple runner for thread extraction"""
    print("🧵 REDDIT CONVERSATION THREAD EXTRACTOR")
    print("=" * 60)
    
    try:
        # Import the main extractor
        from thread_conversation_extractor import ThreadConversationExtractor
        from config import REDDIT_CONFIG
        
        # Initialize extractor
        extractor = ThreadConversationExtractor(REDDIT_CONFIG)
        
        print("📋 EXTRACTION OPTIONS:")
        print("1. Quick extraction (3 threads per subreddit, 5+ comments)")
        print("2. Standard extraction (5 threads per subreddit, 7+ comments)")  
        print("3. Deep extraction (10 threads per subreddit, 10+ comments)")
        print("4. Custom extraction")
        print("5. Exit")
        
        while True:
            try:
                choice = input("\nSelect option (1-5): ").strip()
                
                if choice == "1":
                    threads_per_subreddit = 3
                    min_comments = 5
                    break
                elif choice == "2":
                    threads_per_subreddit = 5
                    min_comments = 7
                    break
                elif choice == "3":
                    threads_per_subreddit = 10
                    min_comments = 10
                    break
                elif choice == "4":
                    try:
                        threads_per_subreddit = int(input("Threads per subreddit: "))
                        min_comments = int(input("Minimum comments per thread: "))
                        break
                    except ValueError:
                        print("❌ Please enter valid numbers")
                        continue
                elif choice == "5":
                    print("👋 Goodbye!")
                    return
                else:
                    print("❌ Invalid choice. Please enter 1-5.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return
        
        # Start extraction
        print(f"\n🚀 Starting extraction: {threads_per_subreddit} threads per subreddit (min {min_comments} comments)")
        print("This will take a few minutes depending on the amount of data...")
        print("Press Ctrl+C to stop\n")
        
        # Extract threads
        total_extracted = extractor.extract_all_subreddits(
            threads_per_subreddit=threads_per_subreddit,
            min_comments=min_comments
        )
        
        if total_extracted > 0:
            # Generate report and save files
            extractor.generate_summary_report()
            
            print(f"\n💾 SAVING DATA...")
            json_file = extractor.save_to_json()
            csv_file = extractor.save_to_csv_format()
            
            print(f"\n🎉 SUCCESS! Extracted {total_extracted} conversation threads")
            print(f"📁 Files created:")
            print(f"   • {json_file}")
            print(f"   • {csv_file}")
            print(f"\n🔬 Data is ready for your sequential sentiment analysis model!")
            
        else:
            print("❌ No threads extracted. Please check:")
            print("   • Your Reddit API credentials in config.py")
            print("   • Your internet connection")
            print("   • Try lowering the minimum comments requirement")
    
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("Make sure you have created thread_conversation_extractor.py")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    except KeyboardInterrupt:
        print("\n🛑 Extraction stopped by user")

if __name__ == "__main__":
    main()