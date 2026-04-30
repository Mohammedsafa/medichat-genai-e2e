from flask import Flask, render_template, jsonify, request
from src.helpers.helper import download_hugging_face_embedding
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)

# Load api ley
load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Download the embeddings
embeddings = download_hugging_face_embedding()


index_name = "medicalbot"


# Embed each chunk and upsert the embeddings into your Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
    
)

# Create a retriever from the vector db
retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={"k":3})

# Setup the config of the model
llm = GoogleGenerativeAI(
    model="models/gemini-2.5-flash", 
    temperature=0.4,
    max_tokens=500
)

# Setup the chat prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ("human", "{input}"),
    ]
)

# Create STUFF chain 
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Routes the chat UI file
@app.route("/")
def index():
    return render_template('chat.html')

# Route the responses during the chat bot
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return jsonify({"answer": response["answer"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)


