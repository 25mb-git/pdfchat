# Proper RAG architecture
This application improves on the problems of the forked version.
(reported in https://github.com/SonicWarrior1/pdfchat/issues/7#issuecomment-2547580815)
- **Vector DB**: Uses FAISS DB properly to implement the RAG architecture 
- **Multiple Users**: Each user session has its own workspace
- **Multiple Files**: Multiple files can be uploaded

# Multiple Files, Multiple Users - PDF Chat Application with Mistral 7B LLM, LangChain, Ollama, FAISS DB, and Streamlit
Enables users to interact with **Multiple** PDF documents using a locally deployed **Mistral 7B** language model. By leveraging **LangChain**, **FAISS** vector database, and **Retrieval-Augmented Generation (RAG)**, it efficiently retrieves relevant document content and provides contextually accurate responses grounded strictly within the uploaded PDFs.

The system supports the ingestion of multiple documents, and each browser reload initiates a **private session**, ensuring that users can interact exclusively with their specific uploaded documents.

## Key Features
- **FAISS Vector Database**: Enables fast and efficient semantic search for document content.  
- **Retrieval-Augmented Generation (RAG)**: Combines the LLM’s generative capabilities with relevant information retrieval to deliver precise, document-grounded answers.  
- **Mistral 7B via Ollama**: Runs the lightweight, high-performance Mistral 7B model locally for inference.  
- **Streamlit Interface**: Provides an intuitive, interactive frontend for seamless user interaction.  

## How It Works
1. **Document Ingestion**: Users upload one or more PDF files.  
2. **Vectorization**: Document content is embedded and stored in a FAISS vector database.  
3. **Semantic Search**: User queries trigger a semantic search within the vector database to locate the most relevant document passages.  
4. **Contextual Response Generation**: The system integrates retrieved information with the Mistral 7B model to generate highly accurate responses.  

---

## Running Mistral 7B Locally with Ollama

### For Mac Users
Follow the instructions to install Ollama here: [Ollama GitHub Repository](https://github.com/ollama/ollama).

---

## Usage Instructions

1. **Clone this repository**:  
   ```bash
   git clone https://github.com/25mb-git/pdfchat.git
    ```

2. **Install dependencies**:
   ```bash
    pip install -r requirements.txt
    ```

3. **Run the application:**:
   ```bash
    streamlit run app.py
    ```

## Demo:

The application has been successfully deployed on an 8GB Mac Mini. Access the deployed version here:
https://mac-mini.boga-vector.ts.net/mistral

Read more about how to deploy in this Medium Post: https://medium.com/@25mb.git/deploying-mistral-7b-on-a-budget-friendly-mac-mini-with-reverse-proxy-using-tailserver-6bae3cb69365


## Technologies Used

Mistral 7B: Lightweight, open-weight LLM optimized for local deployment.
Ollama: Simplifies LLM model deployment and inference.
LangChain: Facilitates seamless integration of LLMs and external tools.
FAISS: High-performance vector database for semantic search.
Streamlit: User-friendly framework for creating interactive web applications.
