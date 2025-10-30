import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

dataset = load_dataset("mteb/amazon_polarity",split="train[:1500]")
df = dataset.to_pandas()
print(df.head())
print(df.shape)

df['feedback_id'] = range(1, len(df) + 1)
df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
df['source'] = np.random.choice(['email', 'chat', 'social_media'], len(df))
df.rename(columns={'text': 'feedback_text', 'label': 'original_label'}, inplace=True)
df.drop_duplicates(subset=['feedback_text'], inplace=True)

print(df.isnull().sum())

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_text'] = df['feedback_text'].apply(clean_text)
print("Sample cleaned text:")
print(df[['feedback_text', 'cleaned_text']].head(2))

def tokenize_text(text):
    return word_tokenize(text)

df['tokens'] = df['cleaned_text'].apply(tokenize_text)
stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words and len(word) > 2]

df['tokens_no_stopwords'] = df['tokens'].apply(remove_stopwords)

lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

df['lemmatized_tokens'] = df['tokens_no_stopwords'].apply(lemmatize_tokens)
df['processed_text'] = df['lemmatized_tokens'].apply(lambda x: ' '.join(x))


df['word_count'] = df['processed_text'].apply(lambda x: len(x.split()))
df = df[df['word_count'] >= 3]
df = df[df['word_count'] <= 500]

df.to_csv('cleaned_feedback_data.csv', index=False)

df[['feedback_id', 'processed_text', 'original_label', 'timestamp', 'source']].to_csv('processed_data.csv', index=False)