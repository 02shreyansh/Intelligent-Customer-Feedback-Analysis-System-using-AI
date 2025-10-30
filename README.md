# 🤖 Intelligent Customer Feedback Analysis System using AI

A comprehensive AI-based system for analyzing, summarizing, and predicting customer sentiment from feedback data using state-of-the-art NLP models.

## 📋 Project Overview

This project implements an end-to-end customer feedback analysis pipeline that includes:
- Data preprocessing and cleaning
- Sentiment classification using DistilBERT
- Text summarization using BART and extractive methods
- Predictive analytics and trend forecasting
- Interactive web application
- AI chatbot for insights exploration

## 🗂️ Project Structure

```
feedback-analysis-system/
│
├── data_preprocessing.py          # Part 1: Data handling
├── sentiment_model.py              # Part 2: Sentiment classification
├── text_summarization.py           # Part 3: Summarization
├── predictive_insights.py          # Part 4: Analytics & forecasting
├── app.py                          # Part 5: Streamlit web app
├── chatbot.py                      # Bonus: AI chatbot
│
├── requirements.txt                # Dependencies
├── README.md                       # Documentation
│
├── cleaned_feedback_data.csv       # Output: Cleaned data
├── processed_data.csv              # Output: Processed data
├── evaluation_metrics.csv          # Output: Model metrics
├── recurring_issues.csv            # Output: Issues analysis
├── satisfaction_forecast.csv       # Output: Predictions
├── AI_insights_report.txt          # Output: Final report
│
└── sentiment_model/                # Trained model directory
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files
```

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/feedback-analysis-system.git
cd feedback-analysis-system
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## 📊 Dataset

**Source**: Amazon Polarity Dataset from Hugging Face
- **Size**: 3.6M+ customer reviews
- **Labels**: Positive (1), Negative (0)
- **Usage**: 10,000 samples for faster processing (configurable)

## 🎯 Usage Guide

### Part 1: Data Preprocessing
```bash
python data_preprocessing.py
```
**Outputs**: 
- `cleaned_feedback_data.csv`
- `processed_data.csv`

**Features**:
- Removes duplicates and special characters
- Tokenization and lemmatization
- Stopword removal
- Handles missing/noisy data

### Part 2: Sentiment Classification
```bash
python sentiment_model.py
```
**Outputs**:
- `sentiment_model.pkl`
- `evaluation_metrics.csv`
- `./sentiment_model/` directory

**Model**: DistilBERT fine-tuned for 3-class sentiment classification
**Metrics**: Accuracy, Precision, Recall, F1 Score

### Part 3: Text Summarization
```bash
python text_summarization.py
```
**Outputs**:
- `bart_summaries.csv`
- `extractive_summaries.csv`
- `summary_comparison.csv`

**Methods**:
1. **BART (Transformer)**: Abstractive summarization
2. **TF-IDF + Cosine Similarity**: Extractive summarization

### Part 4: Predictive Insights
```bash
python predictive_insights.py
```
**Outputs**:
- `recurring_issues.csv`
- `satisfaction_forecast.csv`
- `insights_visualization.png`
- `AI_insights_report.txt`
- `insights_data.json`

**Features**:
- Identifies top recurring issues using N-gram analysis
- Forecasts satisfaction trends for next 30 days using Prophet
- Generates comprehensive insights report with visualizations

### Part 5: Deployment
```bash
streamlit run app.py
```
**Features**:
- 📊 Interactive dashboard
- 📤 Upload and analyze custom data
- 😊 Real-time sentiment analysis
- 📝 Text summarization interface
- 📈 Predictive insights visualization
- 📊 Reports and downloads

**Access**: Open browser at `http://localhost:8501`

### AI Chatbot
```bash
streamlit run chatbot.py
```
**Features**:
- Natural language query interface
- Answers questions about feedback analysis
- Provides actionable recommendations
- Quick action buttons for common queries
