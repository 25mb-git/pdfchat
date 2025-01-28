# Contributions
This application improves on the problems of the forked version.
(reported in https://github.com/SonicWarrior1/pdfchat/issues/7#issuecomment-2547580815)

 ```markdown
    Significantly reworked the forked version to properly support Retrieval-Augmented Generation (RAG).
 ```

Successfully deployed on a 8GB Mac Mini. Access the deployed version here: https://mac-mini.boga-vector.ts.net/mistral

Read more about how to **deploy and host** this on a budget friendly Mac [Medium Post](https://medium.com/@25mb.git/deploying-mistral-7b-on-a-budget-friendly-mac-mini-with-reverse-proxy-using-tailserver-6bae3cb69365).


# What has changed
- **Retrieval-Augmented Generation (RAG) architecture**: to select documents
- **Vector Store**: Uses FAISS DB properly to implement the RAG architecture 
- **Multiple Users**: Each user session has its own workspace
- **Multiple Files**: Multiple files can be uploaded

The system supports the ingestion of multiple documents, and each browser reload initiates a **private session**, ensuring that users can interact exclusively with their specific uploaded documents.

# Use Case: Retrieve Important Information from Email PDFs
This app enables you to upload email PDFs and interactively extract key details such as sender information, dates, attachments, and discussion points. Whether you're summarizing meeting notes, tracking follow-ups, or searching for approvals, the app leverages Retrieval-Augmented Generation (RAG) to provide precise, context-aware answers from one or multiple email threads. 

❤️ I am using this to find the discounts promotions in my inbox ❤️ 

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

# How to Deploy

### Before starting
Follow the instructions to install Ollama here: [Ollama GitHub Repository](https://github.com/ollama/ollama).
1. **Install Mistral**:  

   ```bash
    ollama run mistral
    ```
7B Mistral runs on my Mac mini with 8GB RAM


## Instructions

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

# Tests
## Test cases

1. setup_session: Ensures the files folder exists.
2. pdf_bytes: Simulates a PDF file in memory using BytesIO.
3. test_pdf_upload_and_storage: Verifies PDF uploads and file path creation.
4. test_vector_store_creation: Tests the creation of the FAISS vector store with dummy PDFs.
5. test_streamlit_ui_elements: Ensures session state is initialized.
6. test_user_input_and_chat_flow: Simulates the full chat flow with user input and assistant responses.

## Test execution
1. **Run test code:**:
Make sure pytest is installed:
   ```bash
   pip install pytest
    ```

2. **Run test code:**:
   ```bash
    pytest test/test_app.py
    ```

# Technologies Used

- Mistral 7B: Lightweight, open-weight LLM optimized for local deployment.
- Ollama: Simplifies LLM model deployment and inference.
- LangChain: Facilitates seamless integration of LLMs and external tools.
- FAISS: High-performance vector database for semantic search.
- Streamlit: User-friendly framework for creating interactive web applications.
