import os
import faiss
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

# Ensure environment is set up to avoid any library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the FAISS index
index = faiss.read_index('faiss_index.bin')

# Load the embeddings and text data
df = pd.read_pickle('embeddings.pkl')

# Load the tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Example query
query = "what are benefits of bike sharing stations ?"

# Tokenize the query
tokens = tokenizer.encode(query, return_tensors='pt').to(model.device)

# Generate the query embedding
with torch.no_grad():
    outputs = model(tokens)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# Perform the search on the FAISS index
query_embedding = embedding.reshape(1, -1)
distances, indices = index.search(query_embedding, 5)

# Retrieve and print the results
results = df.iloc[indices[0]]
print(results)
