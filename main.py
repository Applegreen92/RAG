import os
import pandas as pd
import numpy as np
import unicodedata
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModel
import torch
import re


data_path = "paper/[15] Proactive vehicle routing with inferred demand to solve the bikesharing rebalancing problem.pdf"

def load_documents():
    loader = PyPDFLoader(data_path)
    documents = loader.load()
    return documents

print(load_documents())