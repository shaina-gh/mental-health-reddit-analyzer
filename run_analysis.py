#!/usr/bin/env python3
"""
VS Code Runner for Mental Health Sentiment Analysis
Run this file to execute the complete analysis pipeline
"""

import os
import sys
from config import REDDIT_CONFIG, PROJECT_SETTINGS, OUTPUT_SETTINGS
from mental_health_scraper import MentalHealthScraper

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = ['praw', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'textblob']
    optional_packages = ['transformers', 'torch']
    
    print("🔍 Checking dependencies...")
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"❌ {package}")
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package} (optional)")
        except ImportError:
            missing_optional.append(package)
            print(f"⚠️ {package} (optional - will use fallback)")
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("Install with: python3 -m pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\n⚠️ Missing optional packages: {', '.join(missing_optional)}")
        print("For better performance, install with: python3 -m pip install " + " ".join(missing_optional))
    
    print("✅ All required dependencies are available!")
    return True

def test_sample_data_mode():
    """Test with generated sample data (no API required)"""
    print("\n" + "="*60)
    print("🧪 TESTING WITH SAMPLE DATA (NO API REQUIRED)")
    print("="*60)
    
    try:
        # Create scraper without Reddit config (sample mode)
        scraper = MentalHealthScraper()
        
        # Generate sample data
        print("🔄 Generating sample mental health discussions...")
        scraper.generate_sample_data(
            num_threads=PROJECT_SETTINGS['sample_threads'],
            posts_per_thread=PROJECT_SETTINGS['posts_per_thread']
        )
        
        # Process data
        print("🔄 Processing and analyzing sentiment...")
        processed_data = scraper.process_data()
        
        # Analyze trajectories
        print("🔄 Analyzing sentiment trajectories...")
        trajectories = scraper.analyze_sentiment_trajectories()
        
        # Create visualizations
        print("🔄 Creating visualizations...")
        scraper.visualize_results(save_plots=OUTPUT_SETTINGS['save_plots'])
        
        # Generate report
        print("🔄 Generating analysis report...")
        scraper.generate_report()
        
        # Export data if requested
        if OUTPUT_SETTINGS['data_export']:
            processed_data.to_csv(OUTPUT_SETTINGS['export_filename'], index=False)
            print(f"📊 Data exported to {OUTPUT_SETTINGS['export_filename']}")
        
        print("\n✅ Sample data analysis completed successfully!")
        return scraper, processed_data, trajectories
        
    except Exception as e:
        print(f"❌ Error in sample data analysis: {e}")
        return None, None, None

def test_reddit_api_connection():
    """Test Reddit API connection"""
    print("\n" + "="*60)
    print("🔌 TESTING REDDIT API CONNECTION")
    print("="*60)
    
    # Check if credentials are configured
    if (REDDIT_CONFIG['client_id'] == 'YOUR_CLIENT_ID_HERE' or 
        REDDIT_CONFIG['client_secret'] == 'YOUR_CLIENT_SECRET_HERE'):
        print("⚠️ Reddit API credentials not configured in config.py")
        print("📝 To use Reddit API:")
        print("1. Open config.py")
        print("2. Replace YOUR_CLIENT_ID_HERE with your actual client ID")
        print("3. Replace YOUR_CLIENT_SECRET_HERE with your actual secret")
        print("4. Replace YOUR_REDDIT_USERNAME with your Reddit username")
        return None
    
    try:
        # Create scraper with Reddit config
        scraper = MentalHealthScraper(REDDIT_CONFIG)
        
        # Test connection
        test_subreddit = scraper.reddit.subreddit('test')
        subscriber_count = test_subreddit.subscribers
        
        print(f"✅ Successfully connected to Reddit API!")
        print(f"✅ Test subreddit has {subscriber_count:,} subscribers")
        return scraper
        
    except Exception as e:
        print(f"❌ Failed to connect to Reddit API: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Double-check your client_id and client_secret in config.py")
        print("2. Make sure you selected 'script' as app type when creating the Reddit app")
        print("3. Verify your user_agent format")
        print("4. Check your internet connection")
        return None

def run_reddit_data_analysis(scraper):
    """Run analysis with real Reddit data"""
    print("\n" + "="*60)
    print("📡 COLLECTING & ANALYZING REAL REDDIT DATA")
    print("="*60)
    
    try:
        # Collect data from Reddit
        print("🔄 Collecting data from Reddit...")
        print(f"📍 Target subreddits: {', '.join(PROJECT_SETTINGS['target_subreddits'])}")
        
        scraper.scrape_reddit_data(
            max_posts_per_subreddit=PROJECT_SETTINGS['max_posts_per_subreddit']
        )
        
        if not scraper.collected_data:
            print("⚠️ No data was collected from Reddit")
            return None, None
        
        # Process data
        print("🔄 Processing real Reddit data...")
        processed_data = scraper.process_data()
        
        # Analyze trajectories
        print("🔄 Analyzing real sentiment trajectories...")
        trajectories = scraper.analyze_sentiment_trajectories()
        
        # Create visualizations
        print("🔄 Creating visualizations for real data...")
        scraper.visualize_results(save_plots=OUTPUT_SETTINGS['save_plots'])
        
        # Generate report
        print("🔄 Generating real data analysis report...")
        scraper.generate_report()
        
        # Export data
        if OUTPUT_SETTINGS['data_export']:
            real_data_filename = f"real_{OUTPUT_SETTINGS['export_filename']}"
            processed_data.to_csv(real_data_filename, index=False)
            print(f"📊 Real data exported to {real_data_filename}")
        
        print("\n✅ Real Reddit data analysis completed successfully!")
        return processed_data, trajectories
        
    except Exception as e:
        print(f"❌ Error in Reddit data analysis: {e}")
        return None, None

def main():
    """Main execution function"""
    print("🧠 MENTAL HEALTH SENTIMENT ANALYSIS - VS CODE EDITION")
    print("=" * 80)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again")
        return
    
    # Step 2: Test sample data mode (always works)
    sample_scraper, sample_data, sample_trajectories = test_sample_data_mode()
    
    if sample_scraper is None:
        print("❌ Sample data test failed. Check your installation.")
        return
    
    # Step 3: Test Reddit API (optional)
    reddit_scraper = test_reddit_api_connection()
    
    if reddit_scraper:
        # Step 4: Run Reddit data analysis
        real_data, real_trajectories = run_reddit_data_analysis(reddit_scraper)
        
        if real_data is not None:
            print("\n🎉 BOTH SAMPLE AND REAL DATA ANALYSES COMPLETED!")
        else:
            print("\n✅ SAMPLE DATA ANALYSIS COMPLETED (Reddit data failed)")
    else:
        print("\n✅ SAMPLE DATA ANALYSIS COMPLETED (Reddit API not configured)")
    
    # Final summary
    print("\n" + "="*80)
    print("📋 ANALYSIS SUMMARY")
    print("="*80)
    
    if sample_data is not None:
        print(f"✅ Sample Data: {len(sample_data)} posts analyzed")
        print(f"✅ Sample Trajectories: {len(sample_trajectories)} threads analyzed")
    
    if 'real_data' in locals() and real_data is not None:
        print(f"✅ Real Reddit Data: {len(real_data)} posts analyzed")
        print(f"✅ Real Trajectories: {len(real_trajectories)} threads analyzed")
    
    print(f"\n📁 Output files in: {os.getcwd()}")
    if OUTPUT_SETTINGS['save_plots']:
        print(f"📊 Visualizations: {OUTPUT_SETTINGS['plot_filename']}")
    if OUTPUT_SETTINGS['data_export']:
        print(f"📊 Data exports: {OUTPUT_SETTINGS['export_filename']}")
    
    print("\n🎯 Next steps for your project:")
    print("1. Review the generated visualizations")
    print("2. Examine the sentiment trajectories")
    print("3. Scale up data collection for larger dataset")
    print("4. Fine-tune transformer models with your data")
    print("5. Implement advanced trajectory modeling features")

if __name__ == "__main__":
    # Check if we're using the right Python command
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. You have:", sys.version)
        print("Please use: python3 run_analysis.py")
        sys.exit(1)
    
    main()