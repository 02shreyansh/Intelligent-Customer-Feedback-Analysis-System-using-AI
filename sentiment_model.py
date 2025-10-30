import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/data/csv/processed_data.csv')
print(df['sentiment'].value_counts())

np.random.seed(42)
df['sentiment'] = df['original_label'].copy()
sentiment_map = {0: 'Negative', 1: 'Positive'}
df['sentiment_label'] = df['sentiment'].map(sentiment_map)

# Split data into train and val
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentiment'])
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['sentiment'])
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

train_dataset = Dataset.from_dict({
    'text': train_df['processed_text'].tolist(),
    'label': train_df['sentiment'].tolist()
})
val_dataset = Dataset.from_dict({
    'text': val_df['processed_text'].tolist(),
    'label': val_df['sentiment'].tolist()
})
test_dataset = Dataset.from_dict({
    'text': test_df['processed_text'].tolist(),
    'label': test_df['sentiment'].tolist()
})
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)

def compute_metric(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    save_strategy="epoch",
    eval_strategy="epoch",  
    load_best_model_at_end=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metric,
)
trainer.train()
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)
labels = predictions.label_ids
accuracy = accuracy_score(labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Value': [accuracy, precision, recall, f1]
})
metrics_df.to_csv('/data/csv/evaluation_metrics.csv', index=False)
model.save_pretrained('./sentiment_model')
tokenizer.save_pretrained('./sentiment_model')
model_info = {
    'model_path': './sentiment_model',
    'tokenizer': tokenizer,
    'label_map': {0: 'Negative', 1: 'Positive'},
    'metrics': {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
}

with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model_info, f)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    label_map = {0: 'Negative', 1: 'Positive'}
    return label_map[prediction]

test_texts = [
    "This product is amazing! I love it!",
    "Terrible quality, waste of money.",
    "It's okay, nothing special."
]
for text in test_texts:
    sentiment = predict_sentiment(text)
    print(f"Predicted Sentiment: {sentiment}")