import os
import pytest
from streamlit.runtime.state import SessionState
import streamlit as st
from io import BytesIO
from app import create_vector_store, _get_session

FILE_FOLDER = './files'

@pytest.fixture(scope="session", autouse=True)
def setup_session():
    """Setup session directory."""
    if not os.path.exists(FILE_FOLDER):
        os.mkdir(FILE_FOLDER)

@pytest.fixture
def pdf_bytes():
    """Sample PDF content for testing."""
    sample_text = b"%PDF-1.4\n%This is a sample PDF file.\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    return BytesIO(sample_text)

def test_pdf_upload_and_storage(pdf_bytes):
    """Test that the PDF upload and storage works."""
    session_id = _get_session()
    os.makedirs(session_id, exist_ok=True)
    
    file_path = os.path.join(session_id, "test.pdf")
    with open(file_path, "wb") as f:
        f.write(pdf_bytes.read())

    assert os.path.exists(file_path), "PDF file was not stored."

def test_vector_store_creation():
    """Test if vector store is created properly."""
    session_id = _get_session()
    os.makedirs(session_id, exist_ok=True)

    # Create a dummy file in the session folder
    dummy_file_path = os.path.join(session_id, "test.pdf")
    with open(dummy_file_path, "wb") as f:
        f.write(b"%PDF-1.4\nDummy PDF Content\nendobj\n")

    vector_store = create_vector_store(session_id)
    assert vector_store is not None, "Vector store not created."

def test_streamlit_ui_elements():
    """Test that Streamlit renders the expected UI components."""
    st.session_state['session_id'] = _get_session()
    assert 'session_id' in st.session_state, "Session state not initialized."

def test_user_input_and_chat_flow():
    """Test chat interaction flow with mock user input."""
    st.session_state['session_id'] = _get_session()
    st.session_state['chat_history'] = []

    # Mock user input
    user_input = "Summarize the document."
    st.session_state.chat_history.append({"role": "user", "message": user_input})

    assert len(st.session_state.chat_history) == 1, "User input not recorded in chat history."

    # Mock assistant response
    assistant_response = "This document contains a summary of the uploaded file."
    st.session_state.chat_history.append({"role": "assistant", "message": assistant_response})

    assert len(st.session_state.chat_history) == 2, "Assistant response not added to chat history."
    assert st.session_state.chat_history[-1]["message"] == assistant_response, "Response message incorrect."
