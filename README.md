# Mental Health Reddit Conversation Analyzer

A comprehensive tool for extracting and analyzing mental health discussions from Reddit, focusing on sentiment trajectory analysis through conversation threads.

## 🎯 Project Overview

This project aims to analyze mental health discussions on Reddit by:
- Extracting complete conversation threads (not just individual posts)
- Preserving chronological order of comments
- Tracking sentiment changes throughout conversations
- Identifying mental health trajectory patterns in discussions

## 🌟 Key Features

- **Thread-Focused Analysis**: Extracts complete conversation threads with chronological ordering
- **Multi-Subreddit Support**: Monitors 8 mental health subreddits
- **Sentiment Analysis**: Uses transformer models (RoBERTa-based) for accurate sentiment detection
- **Real-Time Monitoring**: Optional real-time analysis of new posts
- **Data Export**: Multiple formats (JSON, CSV) for further analysis
- **Conversation Trajectory**: Tracks sentiment changes through discussion flow

## 📊 Target Subreddits

- r/depression
- r/anxiety  
- r/mentalhealth
- r/bipolar
- r/BipolarReddit
- r/ptsd
- r/socialanxiety
- r/OCD

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Reddit API credentials

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/mental-health-reddit-analyzer.git
   cd mental-health-reddit-analyzer
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv mental_health_env
   source mental_health_env/bin/activate  # Mac/Linux
   # or
   mental_health_env\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Reddit API:**
   ```bash
   cp config_template.py config.py
   # Edit config.py with your Reddit API credentials
   ```

## 🔑 Reddit API Setup

1. Go to [Reddit Apps](https://www.reddit.com/prefs/apps)
2. Click "Create App" or "Create Another App"
3. Choose "script" as the app type
4. Fill in the details:
   - **Name**: MentalHealthAnalyzer
   - **Description**: Academic project for mental health sentiment analysis
   - **Redirect URI**: http://localhost:8080
5. Get your `client_id` and `client_secret`
6. Update `config.py` with your credentials

## 🚀 Usage

### Basic Thread Extraction
```bash
python3 extract_threads.py
```

### Complete Analysis (Historical Data)
```bash
python3 run_analysis.py
```

### Real-Time Monitoring
```bash
python3 start_realtime_monitor.py
```

## 📁 Output Files

The program generates several types of output:

- **`mental_health_threads_*.json`**: Complete thread data with conversation flow
- **`mental_health_conversations_*.csv`**: Flattened conversation data for analysis
- **`mental_health_analysis_results.png`**: Visualization charts
- **`real_processed_mental_health_data.csv`**: Processed sentiment data

## 📈 Data Structure

### Thread Data Format
```json
{
  "thread_id": "abc123",
  "subreddit": "depression",
  "title": "Thread title",
  "original_post": {
    "position": 0,
    "author": "user1",
    "text": "Original post content",
    "timestamp": "2024-08-15T10:00:00"
  },
  "conversation_flow": [
    {
      "position": 1,
      "author": "user2",
      "text": "Reply content",
      "timestamp": "2024-08-15T10:15:00"
    }
  ]
}
```

## 🧠 Sentiment Analysis

The project uses multiple sentiment analysis approaches:

1. **Primary**: `cardiffnlp/twitter-roberta-base-sentiment-latest` (Transformer model)
2. **Fallback**: TextBlob sentiment analysis
3. **Custom**: Ready for integration with specialized mental health models

## 📊 Key Metrics

- **Sentiment Distribution**: Positive, Negative, Neutral percentages
- **Trajectory Analysis**: Improving, Declining, Stable conversation patterns  
- **Thread Statistics**: Participants, exchanges, duration
- **Alert System**: Flags concerning content in real-time

## 🔬 Research Applications

This tool is designed for:
- Mental health research
- Conversation sentiment trajectory analysis
- Social media mental health monitoring
- Academic studies on online support communities

## ⚠️ Ethical Considerations

- **Privacy**: No personal information is stored
- **Research Ethics**: Intended for academic/research purposes
- **Content Guidelines**: Follows Reddit's API terms of service
- **Mental Health**: Tool is for research, not clinical diagnosis

## 📋 Requirements

See `requirements.txt` for full list of dependencies:
- praw (Reddit API)
- pandas (Data manipulation)
- numpy (Numerical operations)
- matplotlib (Visualization)
- seaborn (Statistical visualization)
- textblob (Natural language processing)
- transformers (Advanced sentiment analysis)
- torch (Machine learning backend)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Reddit API for data access
- Hugging Face for transformer models
- Mental health communities for their openness in sharing experiences
- Academic research community for mental health analytics

## 📞 Contact

For questions about this research project, please open an issue in this repository.

---

**Note**: This tool is for research purposes only and should not be used as a substitute for professional mental health advice or diagnosis.