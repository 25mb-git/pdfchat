import os
import time
import base64

from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


FILE_FOLDER = './files'

def _get_session():
    ctx = get_script_run_ctx()
    session_id = ctx.session_id
    return os.path.join(FILE_FOLDER, session_id)

def create_vector_store(data_dir):
    '''Create a vector store from PDF files'''
    # define what documents to load
    loader = DirectoryLoader(path=data_dir, glob="*.pdf", loader_cls=PyPDFLoader)

    # interpret information in the documents
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500,
                                              chunk_overlap=200)
    texts = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # create the vector store database
    db = FAISS.from_documents(texts, embeddings)
    return db

def footer_image(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-repeat: no-repeat;
    background-size: 1000px 200px;
    background-position-x: center;
    background-position-y: 50px;
    }
    </style>
    <div>
    <br><br><br><br><br><br>
    </div>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

def footer():
    footer_html = """<div style='text-align: left;'>
      <p>Developed with ❤️ by Millie Bay at British School in Tokyo</p>
    </div>"""
    st.markdown(footer_html, unsafe_allow_html=True)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-repeat: no-repeat;
    background-position: right 40px;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

def create_vectorstore_if_unavailable():
    if 'vectorstore' not in st.session_state and len(os.listdir(st.session_state['session_id'])) > 0 :
        st.session_state.vectorstore = create_vector_store(st.session_state['session_id'])

def background_image():
    set_background('./bot.png')
    footer_image('./christmas.png')

if __name__ == "__main__":

    background_image()
    
    if not os.path.exists(FILE_FOLDER):
        os.mkdir(FILE_FOLDER)

    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = _get_session()
        if not os.path.exists(st.session_state['session_id']):
            os.mkdir(st.session_state['session_id'])

    create_vectorstore_if_unavailable()

    if 'template' not in st.session_state:
        st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative. If you do not know, ask for more information.

        Context: {context}
        History: {history}

        User: {question}
        Chatbot:"""
    if 'prompt' not in st.session_state:
        st.session_state.prompt = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=st.session_state.template,
        )
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True,
            input_key="question")

    if 'llm' not in st.session_state:
        st.session_state.llm = Ollama(base_url="http://localhost:11434",
                                      model="mistral",
                                      verbose=True,
                                      callback_manager=CallbackManager(
                                          [StreamingStdOutCallbackHandler()]),
                                      )

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


    if 'prev_doc' not in st.session_state:
        st.session_state['prev_doc'] = ""

    # Upload a PDF file
    uploaded_file = st.file_uploader("Upload your PDF", type='pdf')

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["message"])

    if uploaded_file is not None:
        # path = os.path.join("files", uploaded_file.name)
        path = os.path.join(st.session_state['session_id'], uploaded_file.name)
        if st.session_state['prev_doc'] != uploaded_file.name:
            st.session_state.chat_history = []
            st.session_state.memory.clear()

            print (st.session_state['prev_doc'], uploaded_file.name)
            with st.status("Analyzing your document..."):
                bytes_data = uploaded_file.read()
                f = open(path, "wb")
                f.write(bytes_data)
                f.close()
                loader = PyPDFLoader(path)
                data = loader.load()

                # Initialize text splitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=200,
                    length_function=len
                )

                create_vectorstore_if_unavailable()
                st.session_state.vectorstore.add_documents(data)
                st.session_state['prev_doc'] = uploaded_file.name

        st.session_state.retriever = st.session_state.vectorstore.as_retriever()
        # Initialize the QA chain
        if 'qa_chain' not in st.session_state:
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                chain_type='stuff',
                retriever=st.session_state.retriever,
                verbose=True,
                chain_type_kwargs={
                    "verbose": True,
                    "prompt": st.session_state.prompt,
                    "memory": st.session_state.memory,
                }
            )

        # Chat input
        if user_input := st.chat_input("You:", key="user_input"):
            user_message = {"role": "user", "message": user_input}
            st.session_state.chat_history.append(user_message)
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Assistant is typing..."):
                    response = st.session_state.qa_chain(user_input)
                message_placeholder = st.empty()
                full_response = ""
                for chunk in response['result'].split():
                    full_response += chunk + " "
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

            chatbot_message = {"role": "assistant", "message": response['result']}
            st.session_state.chat_history.append(chatbot_message)

    else:
        st.write("Please upload a PDF file.")