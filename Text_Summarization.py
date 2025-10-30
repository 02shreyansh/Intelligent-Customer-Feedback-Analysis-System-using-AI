import pandas as pd
import numpy as np
from transformers import BartForConditionalGeneration, BartTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/data/csv/processed_data.csv')
long_feedback = df[df['processed_text'].str.split().str.len() > 50].copy()

bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

def bart_summarize(text, max_length=150, min_length=50):
    inputs = bart_tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = bart_model.generate(
        inputs['input_ids'],
        num_beams=4,
        max_length=max_length,
        min_length=min_length,
        early_stopping=True
    )
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

sample_texts = long_feedback['processed_text'].head(5).tolist()
bart_summaries = []
for idx, text in enumerate(sample_texts, 1):
    print(f"\nProcessing feedback {idx}/5...")
    short_summary = bart_summarize(text, max_length=50, min_length=20)
    detailed_summary = bart_summarize(text, max_length=150, min_length=50)
    bart_summaries.append({
        'feedback_id': idx,
        'original_text': text[:200] + '...',
        'short_summary': short_summary,
        'detailed_summary': detailed_summary
    })
    print(f"Short Summary: {short_summary}")
    print(f"Detailed Summary: {detailed_summary}")

bart_df = pd.DataFrame(bart_summaries)
bart_df.to_csv('bart_summaries.csv', index=False)

def extractive_summarize(text, num_sentences=3):
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(sentences) <= num_sentences:
        return '. '.join(sentences) + '.'
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    top_sentences = ranked_sentences[:num_sentences]
    top_sentences = sorted(top_sentences, key=lambda x: sentences.index(x[1]))
    
    summary = '. '.join([s[1] for s in top_sentences]) + '.'
    return summary

extractive_summaries = []
for idx, text in enumerate(sample_texts, 1):
    print(f"\nProcessing feedback {idx}/5...")
    short_summary = extractive_summarize(text, num_sentences=2)
    detailed_summary = extractive_summarize(text, num_sentences=4)
    
    extractive_summaries.append({
        'feedback_id': idx,
        'original_text': text[:200] + '...',
        'short_summary': short_summary,
        'detailed_summary': detailed_summary
    })
    
    print(f"Short Summary: {short_summary}")
    print(f"Detailed Summary: {detailed_summary}")

extractive_df = pd.DataFrame(extractive_summaries)
extractive_df.to_csv('/data/csv/extractive_summaries.csv', index=False)
comparison_data = []
for i in range(len(sample_texts)):
    comparison_data.append({
        'Feedback_ID': i+1,
        'Original_Length': len(sample_texts[i].split()),
        'BART_Short': len(bart_summaries[i]['short_summary'].split()),
        'BART_Detailed': len(bart_summaries[i]['detailed_summary'].split()),
        'Extractive_Short': len(extractive_summaries[i]['short_summary'].split()),
        'Extractive_Detailed': len(extractive_summaries[i]['detailed_summary'].split())
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df)
comparison_df.to_csv('/data/csv/summary_comparison.csv', index=False)

example_idx = 0
summary_output = {
    'total_feedback_analyzed': len(long_feedback),
    'summaries_generated': len(sample_texts),
    'methods_used': ['BART (Transformer)', 'TF-IDF + Cosine Similarity (Extractive)'],
    'output_files': ['/data/csv/bart_summaries.csv', '/data/csv/extractive_summaries.csv', '/data/csv/summary_comparison.csv']
}
for key, value in summary_output.items():
    print(f"{key}: {value}")