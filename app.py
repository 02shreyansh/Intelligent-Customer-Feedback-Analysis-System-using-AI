import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from datetime import datetime
import json

st.set_page_config(
    page_title="AI Customer Feedback Analysis",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)
st.markdown('<p class="main-header">🤖 AI Customer Feedback Analysis System</p>', unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Module", [
    "Dashboard",
    "Upload & Analyze",
    "Sentiment Analysis",
    "Text Summarization",
    "Predictive Insights",
    "Reports"
])

@st.cache_resource
def load_sentiment_model():
    try:
        tokenizer = DistilBertTokenizer.from_pretrained('sentiment_model')
        model = DistilBertForSequenceClassification.from_pretrained('sentiment_model')
        return tokenizer, model
    except:
        st.warning("Sentiment model not found. Please train the model first.")
        return None, None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('dataset/csv/processed_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except:
        return None

tokenizer, model = load_sentiment_model()
df = load_data()
if page == "Dashboard":
    st.header("Dashboard Overview")
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Feedback", f"{len(df):,}")
        with col2:
            positive_pct = (df['original_label'] == 1).sum() / len(df) * 100
            st.metric("Positive Feedback", f"{positive_pct:.1f}%")
        with col3:
            negative_pct = (df['original_label'] == 0).sum() / len(df) * 100
            st.metric("Negative Feedback", f"{negative_pct:.1f}%")
        with col4:
            avg_length = df['processed_text'].str.split().str.len().mean()
            st.metric("Avg Words/Feedback", f"{avg_length:.0f}")
        st.markdown("---")
    
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment Distribution")
            sentiment_counts = df['original_label'].value_counts()
            sentiment_map = {0: 'Negative', 1: 'Positive'}
            labels = [sentiment_map[i] for i in sentiment_counts.index]
            
            fig = px.pie(
                values=sentiment_counts.values,
                names=labels,
                color_discrete_sequence=['#ff6b6b', '#51cf66']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Feedback by Source")
            source_counts = df['source'].value_counts()
            fig = px.bar(
                x=source_counts.index,
                y=source_counts.values,
                labels={'x': 'Source', 'y': 'Count'},
                color=source_counts.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Feedback Timeline")
        daily_counts = df.groupby(df['timestamp'].dt.date).size().reset_index()
        daily_counts.columns = ['Date', 'Count']
        
        fig = px.line(
            daily_counts,
            x='Date',
            y='Count',
            title='Daily Feedback Volume'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("No data loaded. Please run the preprocessing script first.")
elif page == "Upload & Analyze":
    st.header("Upload Feedback Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        upload_df = pd.read_csv(uploaded_file)
        st.success(f"File uploaded successfully! {len(upload_df)} records found.")
        st.subheader("Preview Data")
        st.dataframe(upload_df.head(10))
        st.subheader("Data Summary")
        st.write(upload_df.describe())
        if st.button("Analyze Uploaded Data"):
            with st.spinner("Analyzing feedback..."):
                if 'text' in upload_df.columns or 'feedback' in upload_df.columns:
                    text_col = 'text' if 'text' in upload_df.columns else 'feedback'
                    sentiments = []
                    progress_bar = st.progress(0)
                    for idx, text in enumerate(upload_df[text_col].head(100)):
                        if tokenizer and model:
                            inputs = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=128)
                            outputs = model(**inputs)
                            pred = torch.argmax(outputs.logits, dim=1).item()
                            sentiments.append(pred)
                        progress_bar.progress((idx + 1) / min(100, len(upload_df)))
                    
                    upload_df['sentiment'] = sentiments + [None] * (len(upload_df) - len(sentiments))
                    
                    st.success("Analysis complete!")
                    st.dataframe(upload_df.head(20))
                    csv = upload_df.to_csv(index=False)
                    st.download_button(
                        label="Download Analyzed Data",
                        data=csv,
                        file_name="analyzed_feedback.csv",
                        mime="text/csv"
                    )

elif page == "Sentiment Analysis":
    st.header("Real-time Sentiment Analysis")
    st.subheader("Analyze Single Feedback")
    user_input = st.text_area("Enter customer feedback:", height=150)
    if st.button("Analyze Sentiment"):
        if user_input and tokenizer and model:
            with st.spinner("Analyzing..."):
                inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=128)
                outputs = model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=1).item()
                confidence = torch.softmax(outputs.logits, dim=1).max().item()
                sentiment_map = {0: 'Negative', 1: 'Positive'}
                color_map = {0: 'red', 1: 'green'}
                st.markdown(f"### Sentiment: :{color_map[prediction]}[{sentiment_map[prediction]}]")
                st.metric("Confidence", f"{confidence*100:.2f}%")
                probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
                prob_df = pd.DataFrame({
                    'Sentiment': ['Negative', 'Positive'],
                    'Probability': probs
                })
                fig = px.bar(
                    prob_df,
                    x='Sentiment',
                    y='Probability',
                    color='Sentiment',
                    color_discrete_map={'Negative': '#ff6b6b', 'Positive': '#51cf66'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please enter feedback text.")
    
    st.markdown("---")
    st.subheader("Batch Sentiment Analysis")
    if df is not None:
        st.write(f"Analyzing {len(df)} feedback entries from database...")
        sentiment_dist = df['original_label'].value_counts()
        sentiment_map = {0: 'Negative', 1: 'Positive'}
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Positive Feedbacks", sentiment_dist.get(1, 0))
        with col2:
            st.metric("Negative Feedbacks", sentiment_dist.get(0, 0))

elif page == "Text Summarization":
    st.header("Text Summarization")
    st.subheader("Summarize Long Feedback")
    feedback_text = st.text_area("Enter long feedback to summarize:", height=200)
    summary_length = st.select_slider(
        "Summary Length",
        options=['Short', 'Medium', 'Detailed'],
        value='Medium'
    )
    
    if st.button("Generate Summary"):
        if feedback_text:
            with st.spinner("Generating summary..."):
                sentences = feedback_text.split('.')
                sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
                if summary_length == 'Short':
                    summary = '. '.join(sentences[:2]) + '.'
                elif summary_length == 'Medium':
                    summary = '. '.join(sentences[:3]) + '.'
                else:
                    summary = '. '.join(sentences[:5]) + '.'
                
                st.success("Summary Generated!")
                st.info(summary)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Length", f"{len(feedback_text.split())} words")
                with col2:
                    st.metric("Summary Length", f"{len(summary.split())} words")
        else:
            st.warning("Please enter feedback text.")
    
    st.markdown("---")
    st.subheader("Pre-generated Summaries")
    try:
        summaries_df = pd.read_csv('dataset/csv/bart_summaries.csv')
        st.dataframe(summaries_df)
    except:
        st.info("No pre-generated summaries available. Run the summarization script first.")

elif page == "Predictive Insights":
    st.header("Predictive Analytics & Insights")
    try:
        with open('dataset/json/insights_data.json', 'r') as f:
            insights = json.load(f)
        st.subheader("Key Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Current Satisfaction",
                insights['satisfaction_trends']['current_avg_score']
            )
        
        with col2:
            st.metric(
                "Predicted Satisfaction",
                insights['satisfaction_trends']['predicted_avg_score'],
                delta=insights['satisfaction_trends']['change_percentage']
            )
        
        with col3:
            st.metric(
                "Trend",
                insights['satisfaction_trends']['trend']
            )
        
        st.markdown("---")
        st.subheader("Top Recurring Issues")
        issues_df = pd.read_csv('dataset/csv/recurring_issues.csv')
        fig = px.bar(
            issues_df.head(10),
            x='Frequency',
            y='Issue/Complaint',
            orientation='h',
            title='Top 10 Customer Complaints'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Satisfaction Forecast (Next 30 Days)")
        forecast_df = pd.read_csv('dataset/csv/satisfaction_forecast.csv')
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['predicted_satisfaction'],
            mode='lines',
            name='Predicted Satisfaction',
            line=dict(color='#1f77b4', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['upper_bound'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['lower_bound'],
            mode='lines',
            name='Lower Bound',
            fill='tonexty',
            line=dict(width=0),
            showlegend=False
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Recommendations")
        for rec in insights['recommendations']:
            st.markdown(f"{rec}")
    except Exception as e:
        st.error("Insights data not found. Please run the predictive insights script first.")
        st.info(str(e))

elif page == "Reports":
    st.header("Reports & Downloads")
    st.subheader("📄 Available Reports")
    col1, col2 = st.columns(2)
    with col1:
        files = [
            ('dataset/csv/cleaned_feedback_data.csv', 'Cleaned Dataset'),
            ('dataset/csv/processed_data.csv', 'Processed Data'),
            ('dataset/csv/evaluation_metrics.csv', 'Model Metrics'),
            ('dataset/csv/recurring_issues.csv', 'Recurring Issues'),
            ('dataset/csv/satisfaction_forecast.csv', 'Satisfaction Forecast'),
            ('dataset/text/AI_insights_report.txt', 'Insights Report')
        ]
        for filename, description in files:
            try:
                with open(filename, 'rb') as f:
                    st.download_button(
                        label=f"Download {description}",
                        data=f,
                        file_name=filename,
                        mime='text/csv' if filename.endswith('.csv') else 'text/plain'
                    )
            except:
                st.info(f"{description} not available yet")
    with col2:
        try:
            st.image('dataset/image/insights_visualization.png', caption='Insights Dashboard')
        except:
            st.info("Visualization not generated yet")
    
    st.markdown("---")

    st.subheader("Full Insights Report")
    try:
        with open('dataset/text/AI_insights_report.txt', 'r') as f:
            report = f.read()
        st.text_area("Report Content", report, height=400)
    except:
        st.info("Report not generated yet. Run the predictive insights script.")
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>AI Customer Feedback Analysis System | Powered by DistilBERT & Prophet</p>
    </div>
""", unsafe_allow_html=True)