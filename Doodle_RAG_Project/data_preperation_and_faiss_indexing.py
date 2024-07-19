import os
import faiss
import fitz
import re
import unicodedata
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel



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


# Define the function to clean text
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # Remove non-alphanumeric characters (excluding spaces)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace, tabs, and newlines
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Define the function to split text into token-based chunks
def split_text_into_chunks_token_based(text, tokenizer, max_length=300, overlap=50):
    # Warning about chunksize can be ignored, due to overlap it is handled in this function
    tokens = tokenizer.encode(text)

    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + max_length]
        chunks.append(chunk_tokens)
        i += max_length - overlap

    return chunks


# Define the function to convert tokens to embeddings
def tokens_to_embeddings(tokens):
    with torch.no_grad():
        input_tensor = torch.tensor([tokens]).to(model.device)
        outputs = model(input_tensor)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embeddings


# Clean the text
cleaned_text = clean_text(all_text)
print(cleaned_text)
# Load a tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Define chunking parameters
max_chunk_length = 300
overlap = 50

# Split the cleaned text into token-based chunks
chunks = split_text_into_chunks_token_based(cleaned_text, tokenizer, max_length=max_chunk_length, overlap=overlap)

# Convert tokens to embeddings for each chunk
embeddings = [tokens_to_embeddings(chunk) for chunk in chunks]

# Convert chunks and embeddings to a DataFrame for further processing
df = pd.DataFrame({'text': [tokenizer.decode(chunk) for chunk in chunks], 'tokens': chunks, 'embeddings': embeddings})

# Save cleaned and tokenized data with embeddings
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


# Convert tokens to embeddings, apply uses the embedding function on each token within the dataframe
df['embeddings'] = df['tokens'].apply(tokens_to_embeddings)

# Create a FAISS index
d = df['embeddings'][0].shape[0]  # Dimension of embeddings
index = faiss.IndexFlatL2(d)

# Add embeddings to the index
index.add(np.stack(df['embeddings'].values))

faiss.write_index(index, 'faiss_index.bin')

# Save FAISS index and embeddings
faiss.write_index(index, 'faiss_index.bin')
df[['text', 'embeddings']].to_pickle('embeddings.pkl')

print("FAISS index created and embeddings added and saved.")


