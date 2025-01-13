import os
import uuid
import base64
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores.faiss import FAISS

FILE_FOLDER = './files'

def footer():
    footer_html = """<div style='text-align: left;'>
      <p>Developed with ❤️ by Millie Bay at British School in Tokyo</p>
    </div>"""
    st.markdown(footer_html, unsafe_allow_html=True)
def _get_session():
    """Create a unique session folder to save uploaded files."""
    if 'session_id' not in st.session_state:
        # Generate a unique session ID using uuid
        st.session_state['session_id'] = str(uuid.uuid4())
    session_folder = os.path.join(FILE_FOLDER, st.session_state['session_id'])
    os.makedirs(session_folder, exist_ok=True)
    return session_folder

def create_vector_store(files):
    """Create a vector store from multiple PDF files."""
    documents = []
    
    # Load and process each PDF
    for file in files:
        loader = PyPDFLoader(file)
        documents.extend(loader.load())  # Add the documents from each PDF

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    texts = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

def show_banner(image_path, height="200px"):
    """Display a responsive banner image at the top of the app."""
    with open(image_path, "rb") as f:
        banner_base64 = base64.b64encode(f.read()).decode()

    st.markdown(f"""
        <style>
        .banner-container {{
            width: 100%;
            max-width: 900px;  /* Ensure banner doesn't extend beyond content */
            margin: 0 auto;  /* Center the banner */
            box-sizing: border-box;  /* Prevent overflow due to padding/margin */
            display: flex;
            justify-content: center;
            position: fixed;
            top: 20px;  /* Move the banner 20px lower */
            left: 0;
            right: 0;
            z-index: 1000;
        }}
        .banner {{
            height: {height};
            background-image: url("data:image/png;base64,{banner_base64}");
            background-repeat: no-repeat;
            background-size: auto 80%;  /* Scale width automatically to fit height */
            background-position: center 40px;
            width: 100%;
        }}
        .main-content {{
            margin-top: calc({height} + 30px);  /* Add extra space to avoid overlap with the moved banner */
            padding-top: 0px;
        }}
        .block-container {{
            padding-top: 0px;
        }}

        .github-link {{
            font-size: 14px;  /* Smaller font for GitHub link */
            text-align: left;
            margin-top: 10px;
            color: #666;  /* Gray color */
        }}
        /* Handle smaller screens (e.g., iPhone Safari) */
        @media (max-width: 600px) {{
            .banner-container {{
                max-width: 100%;  /* Full width on smaller screens */
                padding-left: 10px;
                padding-right: 10px;
            }}
            .banner {{
                background-size: contain;  /* Fit the image without overflow */
                height: 150px;  /* Adjust banner height for mobile */
            }}
            .main-content {{
                margin-top: 170px;  /* Adjust for smaller screens */
            }}
        }}
        </style>
        <div class="banner-container">
            <div class="banner"></div>
        </div>
        <div class="main-content">
    """, unsafe_allow_html=True)



if __name__ == "__main__":
    st.set_page_config(page_title="PDF Chatbot with Multiple PDF Support")
    
    # Show the banner at the top
    show_banner('./art/banner.png')  # Add the path to your banner image

    # Session folder for storing PDF uploads
    session_folder = _get_session()

    # Create file folder if it doesn't exist
    os.makedirs(FILE_FOLDER, exist_ok=True)

    # Initialize session state variables
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # st.title("📄 Chat with your PDF documents!")
    st.markdown(
        '<div class="github-link">'
        'My Source code: <a href="https://github.com/25mb-git/pdf-chat" target="_blank">GitHub</a>'
        '</div>',
        unsafe_allow_html=True,
    )

    unsafe_allow_html=True,
    # st.write("Upload multiple PDF / exported eMail files, and the assistant will answer your questions based on their content.")

    # Upload multiple PDF files
    uploaded_files = st.file_uploader("Upload multiple PDF files, email PDFs. The assistant will answer your questions based on their content", type='pdf', accept_multiple_files=True)

    if uploaded_files:
        st.write(f"✅ {len(uploaded_files)} file(s) uploaded. Processing ...")

        # Save uploaded PDFs to the session folder
        saved_file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(session_folder, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            saved_file_paths.append(file_path)

        # Create or update vectorstore
        st.session_state.vectorstore = create_vector_store(saved_file_paths)
        retriever = st.session_state.vectorstore.as_retriever()

        # Set up the QA chain
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=Ollama(base_url="http://localhost:11434", model="mistral", verbose=True,
                       callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])),
            chain_type="stuff",
            retriever=retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": PromptTemplate(
                    input_variables=["history", "context", "question"],
                    template="""You are a knowledgeable chatbot, here to help with questions of the user. Only questions specific to the documents. Your tone should be professional and informative. If you do not know, ask for more information.
                    Context: {context}
                    History: {history}
                    User: {question}
                    Chatbot:"""
                ),
                "memory": ConversationBufferMemory(memory_key="history", return_messages=True, input_key="question"),
            }
        )

        st.success("Documents loaded and vector database created!")
        st.write(f"✅ {len(uploaded_files)} file(s) Processed")

    # Display chat interface
    if st.session_state.qa_chain:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["message"])

        user_input = st.chat_input("Ask a question about the uploaded PDFs:")
        if user_input:
            user_message = {"role": "user", "message": user_input}
            st.session_state.chat_history.append(user_message)
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Assistant is typing..."):
                    response = st.session_state.qa_chain(user_input)
                st.session_state.chat_history.append({"role": "assistant", "message": response["result"]})
                st.markdown(response["result"])

    else:
        st.warning("Please upload PDF files to start chatting!")

    # Close the reserved space for the banner
    st.markdown("</div>", unsafe_allow_html=True)
