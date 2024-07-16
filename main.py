from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from get_embedding_function import get_embedding_function

from langchain_core.documents import Document



data_path = "paper/[1] Probabilistic Forecasts of Bike-Sharing Systems for Journey Planning.pdf"

def load_documents(FILE_PATH):
    loader = PyPDFLoader(data_path)
    documents = loader.load()
    return documents


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)



document = load_documents(data_path)

splitted_documents = split_documents(document)
print(splitted_documents[1])