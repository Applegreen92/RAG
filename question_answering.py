import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

# Load the FAISS index
index = faiss.read_index('faiss_index.bin')

# Load the embeddings and text data
df = pd.read_pickle('embeddings.pkl')

# Load the tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')


def query_to_embedding(query):
    tokens = tokenizer.encode(query, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(tokens)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding


def search(query, top_k=5):
    query_embedding = query_to_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return df.iloc[indices[0]]


# Example query
query = "what are benefits of bike sharing stations ?"
results = search(query)
print(results)
