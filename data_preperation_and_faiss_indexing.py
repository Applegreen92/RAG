import os
import fitz  # PyMuPDF
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
import torch
import pickle

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Directory containing the PDF files
pdf_folder = 'paper'

# List all PDF files in the directory
pdf_paths = [os.path.join(pdf_folder, file) for file in os.listdir(pdf_folder) if file.endswith('.pdf')]

# Extract text from all PDFs
all_text = ""
for pdf_path in pdf_paths:
    all_text += extract_text_from_pdf(pdf_path)

# Function to clean text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove non-alphanumeric characters
    text = text.lower()  # Convert to lowercase
    return text

# Clean the extracted text
cleaned_text = clean_text(all_text)

# Function to split text into chunks
def split_text_into_chunks(text, max_length):
    words = text.split()
    chunks = [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
    return chunks

# Split the cleaned text into chunks
max_chunk_length = 512
chunks = split_text_into_chunks(cleaned_text, max_chunk_length)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize each chunk
chunk_tokens = [tokenizer.encode(chunk, add_special_tokens=True, padding='max_length', truncation=True, max_length=max_chunk_length) for chunk in chunks]

# Convert tokens to a DataFrame for further processing
df = pd.DataFrame({'text': chunks, 'tokens': chunk_tokens})

# Save cleaned and tokenized data
df.to_csv('tokenized_bike_sharing_dataset.csv', index=False)

print("Data extraction, preprocessing, and tokenization completed. Output saved to tokenized_bike_sharing_dataset.csv.")

# Load pre-trained model
model = AutoModel.from_pretrained('bert-base-uncased')

# Function to convert tokens to embeddings
def tokens_to_embeddings(tokens):
    with torch.no_grad():
        outputs = model(torch.tensor([tokens]))
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

# Convert tokens to embeddings
df['embeddings'] = df['tokens'].apply(tokens_to_embeddings)

# Create a FAISS index
d = df['embeddings'][0].shape[0]  # Dimension of embeddings
index = faiss.IndexFlatL2(d)

# Add embeddings to the index
index.add(np.stack(df['embeddings'].values))

print("FAISS index created and embeddings added.")

# Save FAISS index and embeddings
faiss.write_index(index, 'faiss_index.bin')
df[['text', 'embeddings']].to_pickle('embeddings.pkl')

print("FAISS index and embeddings saved.")
