from src.helpers.helper import load_pdf_file, text_split, download_hugging_face_embedding
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# Load the api keys
load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Execute the loader, splitter and embeddings
extracted_data = load_pdf_file(data='data/')
text_chunks = text_split(extracted_data=extracted_data)
embeddings = download_hugging_face_embedding()

# Setup the Pinecone vector database
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = 'medicalbot'

if index_name not in pc.list_indexes().names():

    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    
else:
    print(f"Index {index_name} already exists. Skipping creation.")


# Embed each chunk upsert the embeddings into your Pinecone index
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)



