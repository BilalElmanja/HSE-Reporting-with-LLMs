import os
import chainlit as cl
from chainlit.input_widget import Slider, Switch
from langchain.chains import LLMChain
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader


# setup langchain
import os

from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings


class DocumentProcessor:
    def __init__(self, directory, glob_pattern, loader_class):
        self.directory = directory
        self.glob_pattern = glob_pattern
        self.loader_class = loader_class
        self.documents = None
        self.texts = None

    def load_documents(self):
        loader = DirectoryLoader(self.directory, glob=self.glob_pattern, loader_cls=self.loader_class)
        self.documents = loader.load()
        return self.documents

    def split_text(self, chunk_size=200, chunk_overlap=50):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.texts = text_splitter.split_documents(self.documents)
        return self.texts


class EmbeddingManager:
    def __init__(self, model_name, normalize_embeddings=True, persist_directory=None):
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.persist_directory = persist_directory
        self.embedding = None
        self.vector_db = None

    def create_embedding(self):
        self.embedding = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': self.normalize_embeddings}
        )

    def embed_documents(self, documents):
        if not self.embedding:
            self.create_embedding()

        self.vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding,
            persist_directory=self.persist_directory
        )

    def get_retriever(self, k=1):
        return self.vector_db.as_retriever(search_kwargs={"k": k})



class QuestionAnsweringChain:
    def __init__(self, llm, chain_type, retriever):
        self.llm = llm
        self.chain_type = chain_type
        self.retriever = retriever
        self.qa_chain = None

    def create_qa_chain(self):
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=self.chain_type,
            retriever=self.retriever,
            return_source_documents=True
        )

    def answer_question(self, question):
        if not self.qa_chain:
            self.create_qa_chain()

        llm_response = self.qa_chain(question)
        return process_llm_response(llm_response)


# Example usage:
# response = qa_chain.answer_question("Your question here")


## Cite sources

import textwrap

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    text = str(wrap_text_preserve_newlines(llm_response['result']))

    return text


# Setup the model
import together
import os
import logging
from typing import Any, Dict, List, Mapping, Optional

from pydantic import Extra, Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env
import together

import os

os.environ["TOGETHER_API_KEY"] = "8b4f8ff4f0cee629b54893cec547dddbca975ad63eae06f1d1ca34082f275591"

# set your API key
together.api_key = os.environ["TOGETHER_API_KEY"]

# list available models and descriptons
models = together.Models.list()
together.Models.start("togethercomputer/llama-2-70b-chat")



class TogetherLLM(LLM):
    """Together large language models."""

    model: str = "togethercomputer/llama-2-70b-chat"
    """model endpoint to use"""

    together_api_key: str = os.environ["TOGETHER_API_KEY"]
    """Together API key"""

    temperature: float = 0.1
    """What sampling temperature to use."""

    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion."""

    class Config:
        extra = Extra.forbid

    

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together"

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt,
                                          model=self.model,
                                          max_tokens=self.max_tokens,
                                          temperature=self.temperature,
                                          )
        text = output['output']['choices'][0]['text']
        return text



class Model:
    def __init__(self):
        self.model = None
        self.doc_processor = None
        self.documents = None
        self.texts = None
        self.embedding_manager = None
        self.retriever = None
        self.llm = None
        self.qa_chain = None


    def initialize(self,):
        # Initialize the document processor
        self.doc_processor = DocumentProcessor('../new_papers/', '*.pdf', PyPDFLoader)
        self.documents =  self.doc_processor.load_documents()
        self.texts =  self.doc_processor.split_text()

    # Initialize and use the embedding manager
        self.embedding_manager = EmbeddingManager("BAAI/bge-base-en", normalize_embeddings=True, persist_directory='db')
        self.embedding_manager.embed_documents(self.texts)
        self.retriever =  self.embedding_manager.get_retriever(k=1)

        # initialize the model
        self.llm = TogetherLLM(
            model= "togethercomputer/llama-2-70b-chat",
            temperature = 0.1,
            max_tokens = 1024
        )

        # Initialize and use the QA chain
        qa_chain = QuestionAnsweringChain(self.llm, "stuff",  self.retriever)

        self.model = qa_chain
   


    def predict(self, input):
        llm_response = self.model.answer_question(input)
        if isinstance(llm_response, str):
            return wrap_text_preserve_newlines(llm_response)
        elif isinstance(llm_response, dict) and 'result' in llm_response:
            return wrap_text_preserve_newlines(llm_response['result'])
        else:
            # Handle other types or unexpected responses
            return "An unexpected response type was received."





model = Model()
# Initialize the model

model.initialize()

import streamlit as st
from PIL import Image
import pandas as pd
import pdfplumber
 # Replace with the actual module name where Model class is defined

st.set_page_config(layout="wide")

# Function to save the uploaded file
def save_uploaded_file(uploaded_file):
    try:
        os.makedirs('documents', exist_ok=True)
        with open(os.path.join('documents', uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        return False

# Function to display the document content
def display_document(file_path):
    if file_path.endswith('.pdf'):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                st.text(page.extract_text())
    elif file_path.endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(file_path)
        st.image(image, caption=file_path.split('/')[-1])
    elif file_path.endswith('.csv') or file_path.endswith('.xlsx'):
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        st.dataframe(df)
    else:
        st.error("Unsupported file type!")

# Streamlit app layout

st.title('Chat and Document Viewer Application')
# Set page configuration - use full page width


# Splitting the layout into two columns
# Create three columns. The middle column acts as a spacer.
col1, spacer, col2 = st.columns([3, 1, 2])

# Chat interface in the first column
with col1:
    st.header("Chat")
    user_input = st.text_input("Enter your question:")
    if st.button('Send'):
        if user_input:
            # Get the model's response
            response = model.predict(user_input)
            st.text_area("AI Response", value=response, height=100)
        else:
            st.warning("Please enter a question.")

    # File uploader
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "png", "jpg", "jpeg", "csv", "xlsx"])
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            st.success(f"File {uploaded_file.name} uploaded successfully.")

# Document display in the second column
with col2:
    st.header("Document Viewer")
    for file in os.listdir('../new_papers'):
        file_path = os.path.join('../new_papers', file)
        st.subheader(file)
        display_document(file_path)
