
# Real-Time & Historical Mental Health Trajectory Analyzer

This is a comprehensive, web-based tool for analyzing mental health discussions on Reddit. It moves beyond simple sentiment analysis by focusing on the **trajectory of conversations**, tracking how sentiment evolves through entire threads. The application features a live monitor for real-time insights and a powerful on-demand explorer for deep-dive historical analysis, all powered by a custom-trained sentiment model.

## 🎯 Project Overview

This project aims to analyze mental health discussions on Reddit by:

  - Extracting complete conversation threads (not just individual posts)
  - Preserving chronological order of comments
  - Tracking sentiment changes throughout conversations
  - Identifying mental health trajectory patterns in discussions

## 🌟 Key Features

  - **🧠 Custom Sentiment Model:** Utilizes a fine-tuned version of `mental-bert`, a language model specifically pre-trained on Reddit mental health data. Our custom model is trained to recognize nuanced categories beyond positive/negative, including `Expressing-Pain`, `Seeking-Support`, `Offering-Support`, and `Expressing-Hope`.
  - **🔴 Live Monitoring Dashboard:** A real-time web interface that fetches and analyzes the latest comments from mental health subreddits as they happen, providing an up-to-the-second view of community sentiment.
  - **📖 On-Demand Historical Explorer:** A powerful interactive tool that allows you to:
      - Select a specific subreddit.
      - Fetch a user-defined number of threads and comments directly from the UI.
      - Instantly analyze the sentiment of the fetched data.
      - Generate conclusive sentiment summaries for entire threads.
      - Perform drill-down analysis on a single user's comments within a conversation.
  - **🧵 Thread-Focused Analysis:** The core of this project is its ability to extract and analyze complete, chronologically-ordered conversation threads, revealing the flow and evolution of sentiment.

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

The project uses a custom-trained sentiment analysis model.

1.  **Primary**: A fine-tuned `mental/mental-bert-base-uncased` model trained on custom labels (`Expressing-Pain`, `Seeking-Support`, etc.) for nuanced understanding of mental health discussions.
2.  **Architecture**: The application is a unified Streamlit dashboard, providing both live and historical analysis in a single interface.

## 📊 Key Metrics

  - **Sentiment Distribution**: Breakdown by custom labels (`Expressing-Pain`, `Expressing-Hope`, etc.).
  - **Conclusive Thread Sentiment**: An overall summary of a thread's emotional tone (e.g., "Highly Concerning", "Supportive & Improving").
  - **Single-User Trajectory**: A summary of an individual user's sentiment distribution within a thread.

## 🔬 Research Applications

This tool is designed for:

  - Mental health research
  - Conversation sentiment trajectory analysis
  - Social media mental health monitoring
  - Academic studies on online support communities

## ⚠️ Ethical Considerations

  - **Privacy**: No personal information is stored; data is fetched live or from local files for analysis.
  - **Research Ethics**: Intended for academic/research purposes.
  - **Content Guidelines**: Follows Reddit's API terms of service.
  - **Mental Health**: Tool is for research, not clinical diagnosis.

## 📋 Requirements

See `requirements.txt` for a full list of dependencies, including:

  - praw (Reddit API)
  - pandas (Data manipulation)
  - streamlit (Web dashboard)
  - transformers (Advanced sentiment analysis)
  - torch (Machine learning backend)
  - scikit-learn (Model training utilities)
  - accelerate (Training performance)

## 🤝 Contributing

1.  Fork the repository
2.  Create a feature branch (`git checkout -b feature/amazing-feature`)
3.  Commit your changes (`git commit -m 'Add amazing feature'`)
4.  Push to the branch (`git push origin feature/amazing-feature`)
5.  Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## 🙏 Acknowledgments

  - Reddit API for data access
  - Hugging Face for the `mental-bert` model and the `transformers` library
  - The Streamlit team for their excellent dashboarding tool

## 📞 Contact

For questions about this research project, please open an issue in this repository.

-----

**Note**: This tool is for research purposes only and should not be used as a substitute for professional mental health advice or diagnosis.
