# medichat-genai-e2e

[![Python 3.8+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://python.langchain.com/)
[![VectorDB: Pinecone](https://img.shields.io/badge/VectorDB-Pinecone-blueviolet.svg)](https://www.pinecone.io/)
[![LLM: Gemini](https://img.shields.io/badge/LLM-Gemini--2.5--Flash-orange.svg)](https://ai.google.dev/)

An intelligent medical assistance system built using **Retrieval-Augmented Generation (RAG)**. This project leverages the power of Gemini 2.5 Flash and Pinecone to provide accurate answers from medical PDF documents.

---

## 📂 1. Project Deliverables

This repository provides a complete end-to-end RAG pipeline:

1. **Orchestration Server (`app.py`)**: A Flask-based production script managing the communication between the UI and the LangChain retrieval logic.
2. **Data Ingestion Pipeline (`store_index.py`)**: A modular script for automated document loading, splitting, and vector synchronization.
3. **Prompt Engineering Engine (`src/prompt.py`)**: Specialized system prompts designed to enforce medical accuracy and conciseness.
4. **Environment Configuration**: A structured .env.example for secure API credential management.

---

## 📊 2. Data Transformation Pipeline

The system utilizes a multi-stage pipeline to convert unstructured PDF data into a searchable knowledge base:

*   **Extraction:** Implementation of `PyPDFLoader` to parse and extract text from complex medical layouts.
*   **Recursive Splitting:** Utilization of `RecursiveCharacterTextSplitter` with a `chunk_size` of **500** characters and a `chunk_overlap` of **20** to maintain semantic continuity across fragments.
*   **Vectorization:** Deployment of the `all-MiniLM-L6-v2` Sentence-Transformer model to generate high-dimensional (**384-D**) embeddings.

---

## 🧠 3. RAG Architecture & Logic

To ensure clinical-grade precision, **MediChat-GenAI** follows a rigorous retrieval and generation flow:

*   **Semantic Retrieval:** Queries are processed using **Cosine Similarity** against the **Pinecone** index to fetch the **Top-K (k=3)** most relevant document chunks.
*   **Grounded Generation:** The retrieved context is injected into the **Gemini 2.5 Flash** model. The model is strictly instructed to respond *"I don't know"* if the answer is not present in the context, preventing unauthorized AI generation.
*   **Latency Optimization:** Powered by Gemini's **Flash** architecture, providing near-instantaneous response times suitable for real-time chat applications.

---

## 🛠️ 4. Tech Stack & Architecture

*   **Engine:** Python 3.12+
*   **RAG Framework:** LangChain (Classic Chains & Core Components)
*   **LLM:** Google Gemini 2.5 Flash API
*   **Vector Store:** Pinecone (Serverless Architecture on AWS)
*   **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
*   **Frontend:** Flask with modular HTML/CSS templates

---

## 📁 5. Repository Structure

```text
.
├── src/                 # Core logic and source code
│   ├── helpers/
│   │   ├── helper.py    # Document loading, splitting, and embedding logic
│   │   └── __init__.py  # Makes helpers a sub-package
│   ├── prompt.py        # System prompt templates for the RAG chain
│   └── __init__.py      # Makes src a Python package
├── research/            # RAG experimentation and prototyping
│   └── trials.ipynb     # Jupyter Notebook for initial testing
├── data/                # Medical source documents (Knowledge Base)
│   └── The Gale Encyclopedia of Medicine.pdf
├── static/              # Frontend styling assets
│   └── style.css
├── templates/           # HTML user interface
│   └── chat.html
├── app.py               # Main Flask Orchestrator (Entry Point)
├── store_index.py       # Vector ingestion and Pinecone synchronization
├── requirements.txt     # Python dependencies
├── .env.example         # Template for environment variables
├── .gitignore           # Git exclusion rules
├── LICENSE              # MIT License
└── README.md            # Project documentation
```

---

## ⚙️ 6. Setup & Installation

1. Environment Configuration
Clone the repository and prepare your environment variables:

```bash
cp .env.example .env
```

Update the `.env` file with your credentials:

*   **PINECONE_API_KEY**: Your Pinecone cloud API key.
*   **GOOGLE_API_KEY**: Your Google AI Studio (Gemini) API key.

2. Dependency Management
Install the required packages using the generated requirements file:

```bash
pip install -r requirements.txt
```

3. Data Ingestion
Place your medical PDF files in the `data/` directory and execute the ingestion script:

```bash
python store_index.py
```

---

##  7. Execution

To launch the medical chatbot:
```bash
python app.py
```
Access the interface at `http://localhost:8080`.

--- 


## Authors

Course project implementation by:

 - Mohammed Sherif Safa



