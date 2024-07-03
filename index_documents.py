import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load cleaned and tokenized dataset
df = pd.read_csv('tokenized_faq_dataset.csv')

# Load pre-trained model for generating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the questions
question_embeddings = model.encode(df['question'].tolist())

# Convert embeddings to a format suitable for FAISS
question_embeddings = np.array(question_embeddings).astype('float32')

# Create a FAISS index
index = faiss.IndexFlatL2(question_embeddings.shape[1])
index.add(question_embeddings)

# Save the FAISS index and the dataframe with answers
faiss.write_index(index, 'faq_index.faiss')
df.to_csv('faq_with_embeddings.csv', index=False)

print("Document indexing completed. FAISS index and data saved.")
