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
