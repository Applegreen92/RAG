import re

import pandas as pd
from transformers import AutoTokenizer

# Load dataset
df = pd.read_csv('faq_dataset.csv')


# Function to clean text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove non-alphanumeric characters
    text = text.lower()  # Convert to lowercase
    return text


# Clean questions and answers
df['question'] = df['question'].apply(clean_text)
df['answer'] = df['answer'].apply(clean_text)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize questions and answers
df['question_tokens'] = df['question'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
df['answer_tokens'] = df['answer'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

# Save cleaned and tokenized data
df.to_csv('tokenized_faq_dataset.csv', index=False)

print("Data preprocessing and tokenization completed. Output saved to tokenized_faq_dataset.csv.")
