from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List

# Extract Data From the PDF File
def load_pdf_file(data: str) -> List[Document]:
    """
    Summary: Loads the pdf file from its path to the list of docs

    Args:
        data (str): the path of the pdf file
    
    Returns:
        docs (List[Document]): list of documents loaded

    NextStep: The docs pass into the text splitter func to prepare the data to chunks    
    """
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)
    docs = loader.load()

    return docs


# Split the Data into Text Chunks   
def text_split(extracted_data: List[Document]) -> List[Document]:
    """
    Summary: The function does document splitting, which converts the long pages' content into chunks based on
    chunk size using recursive character text splitter, which care about the content

    Args:
        extracted_data (List[Document]): The loaded pdf file
    
    Returns: 
        text_chunks (List[Document]): List of docs after splitting into chunks
    
    NextStep: The chunks pass to the type of vector database to store them
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks 


# Download the Embeddings from Hugging Face
def download_hugging_face_embedding() -> HuggingFaceEmbeddings:
    """
    Summary: Download the embedding from hugging face lib

    Returns:
        embeddings (HuggingFaceEmbeddings): The embedding object that holds the weights from the sentence transformers model chosen
    
    NextStep: Using the embeddings into the vector database to store the semantic of the sentence
    """
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings


