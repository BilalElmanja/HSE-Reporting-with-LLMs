import os
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from typing import Any, Dict, List, Mapping, Optional
from pydantic import Extra, Field, root_validator
from langchain.utils import get_from_dict_or_env
import PyPDF2
from io import BytesIO
import re

import replicate
import pdfplumber
import time
from datetime import datetime

os.environ["REPLICATE_API_KEY"] = "r8_ajmkmZq4JFMU31elTaIkBXcakllnvuA2463RL"



class LLM_model(LLM):
    """Together large language models."""

    model: str = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
    """model endpoint to use"""

    together_api_key: str = os.environ["REPLICATE_API_KEY"]
    """Together API key"""

    temperature: float = 0.01
    """What sampling temperature to use."""

    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion."""

    class Config:
        extra = Extra.forbid



    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "replicate"

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""

        output = replicate.run(
            self.model,
            input = {
                    "debug": False,
                    "top_k": 10,
                    "top_p": 1,
                    "temperature": 0.01,
                    "prompt": prompt,
                    "system_prompt": "",
                    "max_new_tokens": 4096,
                    "min_new_tokens": -1
            }
            
        )
        response = ""
        for text in output:
            response += text
        return response
    
    def predict(self, prompt):
        output = replicate.run(
            self.model,
            input = {
                    "debug": False,
                    "top_k": 10,
                    "top_p": 1,
                    "temperature": 0.01,
                    "prompt": prompt,
                    "system_prompt": "",
                    "max_new_tokens": 4096,
                    "min_new_tokens": -1
            }
            
        )
        
        return output

        
        
    

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "llm"

        

class Document_Chain:
    def __init__(self):
        self.docs = None
        self.model = None
        self.chain = None
        self.retriever = None
        self.model_config = None
        self.model_name = None
        self.bge_embeddings = None
        self.vectorstore = None
        self.store = None


    def embedding_model(self):
        model_name = "BAAI/bge-small-en-v1.5"
        encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

        bge_embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs=encode_kwargs
        )

        return bge_embeddings


    def load_documents(self, directory):
        loaders = []
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if filename.endswith('.txt'):
                loaders.append(TextLoader(filepath))
            elif filename.endswith('.pdf'):
                loaders.append(PyPDFLoader(filepath))
        docs = []
        for loader in loaders:
            docs.extend(loader.load())

        print(f"Loaded {len(docs)} documents")
        return docs

    def retriever_big_chunks(self, docs):
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=70)
        self.vectorstore = Chroma(collection_name="split_parents", embedding_function=self.bge_embeddings)
        self.store = InMemoryStore()

        big_chunks_retriever = ParentDocumentRetriever(
            vectorstore= self.vectorstore,
            docstore= self.store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,

        )

        big_chunks_retriever.add_documents(docs)
        return big_chunks_retriever
    
    
    def initilize_model(self, api_token):
        os.environ["REPLICATE_API_TOKEN"]=api_token
        self.model_config = {
                    "debug": False,
                    "top_k": 10,
                    "top_p": 1,
                    "temperature": 0.01,
                    "system_prompt": "",

                    "max_new_tokens": 512,
                    "min_new_tokens": -1
                },
        self.model_name = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
        self.model = LLM_model()

        
    
    def initilize_chain(self, document_directory, api_token):
        self.bge_embeddings = self.embedding_model()
        self.initilize_model(api_token)
        self.docs = self.load_documents(document_directory)
        self.retriever = self.retriever_big_chunks(self.docs)
        self.chain = RetrievalQA.from_chain_type(llm=self.model,
                                 chain_type="stuff",
                                 retriever=self.retriever)
        
    


class ConversationChain(Document_Chain):
    def __init__(self, base_directory):
        super().__init__()
        self.base_directory = base_directory
        self.current_conversation_id = None
        self.current_conversation_file = None

    def start_new_conversation(self):
        self.current_conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
        conversation_filename = f"conversation_{self.current_conversation_id}.txt"
        self.current_conversation_file = os.path.join(self.base_directory, conversation_filename)
        with open(self.current_conversation_file, 'w') as file:
            file.write(f"Conversation ID: {self.current_conversation_id}\n")

    def add_to_conversation(self, user_input, agent_response):
        with open(self.current_conversation_file, 'a') as file:
            file.write(f"Me: {user_input}\n")
            file.write(f"You: {agent_response}\n")



class PromptTemplate:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        # additional initialization

    def generate_response(self, user_input, retrieved_docs, model):
        
        return self.create_response(user_input, retrieved_docs, model)
        

    def is_relevant(self, user_input , docs, model):
        # Use LLM to generate a response based on the query and docs
        model_prompt = f"below is a provided content extracted from some document in the database, \n " + \
         f"content : \n {docs} \n \n now based on this content, is it sufficient to answer to this question even to get just a little close : {user_input} ? \n answer only by yes or no"
        response = model.predict(model_prompt)
        if "yes" in response:
            return True
        else:
            return False # placeholder

    def create_response(self, user_input, docs, conversation, model):
        # Use LLM to generate a response based on the query and docs
        model_prompt = f"below is a provided content extracted from some document in the database, and it is the most relevant content to this question : {user_input} \n " + \
         f"content : \n {docs} \n and here's the conversations history between you and me \n conversation : {conversation} \n now based on this content, and the conversation, can you please give a well detailed answer to the question : {user_input} " + \
             f" \n  if you find the content not sufficient to answer the question, answer based on the conversation only and in a friendly way "
        
        return model.predict(model_prompt)  # placeholder
        

    def ask_for_clarification(self):
        return "Can you please provide more specific details?"





def convert_pdf_to_text(pdf_file_path, text_file_path=None):
    # Determine the output text file path
    directory, pdf_filename = os.path.split(pdf_file_path)
    base_filename = os.path.splitext(pdf_filename)[0]
    text_file_path = os.path.join("./documents", f"{base_filename}.txt")

    # Open the PDF and extract text
    with pdfplumber.open(pdf_file_path) as pdf, open(text_file_path, 'w', encoding='utf-8') as text_file:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_file.write(text + "\n")

    return text_file_path



def clean_text_file(file_path):
    # Define a regular expression pattern for allowed characters (English and French)
    pattern = re.compile(r"[a-zA-Z0-9\séàèùâêîôûçëïüÉÀÈÙÂÊÎÔÛÇËÏÜ.,!?'\-]")

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Filter out unwanted characters
    filtered_content = ''.join(pattern.findall(content))

    # Write the cleaned content back to the file or to a new file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(filtered_content)

    return file_path



import chainlit as cl

chain = Document_Chain()
conversation_chain = ConversationChain(base_directory='./conversations')
prompt_template = PromptTemplate()

conversation_chain.initilize_chain("./conversations", "r8_ajmkmZq4JFMU31elTaIkBXcakllnvuA2463RL")

#pdf_directory = "./documents_pdf"
#txt_directory = "./documents"


#for filename in os.listdir(pdf_directory):
    #if filename.endswith(".pdf"):
        #pass
        #pdf_file_path = os.path.join(pdf_directory, filename)
        #text_file_path = convert_pdf_to_text(pdf_file_path, )
        #print(f"Converted {filename} to {text_file_path}")

#for filename in os.listdir(txt_directory):
    #if filename.endswith(".txt"):
        #pass
        #txt_file_path = os.path.join(txt_directory, filename)
        #clean_text_file(txt_file_path)
        #print(f"Cleaned {filename}")


#chain.initilize_chain("./documents", "r8_eehqKGwhuv0jF2KLISsOwIJo0eupYm60oj12m")
# Usage

conversation_chain.start_new_conversation()



@cl.on_chat_start
async def on_chat_start():
    # Example usage
    await cl.Message(content="Hello there, Welcome to HSE Agent").send()
    files = None
    

    # Wait for the user to upload a PDF file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload your PDF Files to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=500,
        ).send()

    
    for file in files:

        # Convert the PDF file to text
        msg = cl.Message(content=f"Processing `{file.name}`...")
        await msg.send()

        # Read the PDF file
        pdf_stream = BytesIO(file.content)
        pdf = PyPDF2.PdfReader(pdf_stream)
        pdf_text = ""
        for page in pdf.pages:

            pdf_text += page.extract_text()

        # Write the PDF text to a text file
        txt_filename = f"{file.name.split('.')[0]}.txt"
        txt_filepath = os.path.join("./documents", txt_filename)
        with open(txt_filepath, 'w', encoding='utf-8') as txt_file:
            txt_file.write(pdf_text)
        
        # Clean the text file
        clean_text_file(txt_filepath)

    
    # Let the user know that the model is being initialized
    msg.content = f"the chain is about to be initialized..."
    await msg.update()

    # Initialize the chain for documents and conversation
    
    await cl.make_async(chain.initilize_chain)("./documents", "r8_ajmkmZq4JFMU31elTaIkBXcakllnvuA2463RL")
    await cl.make_async(conversation_chain.initilize_chain)("./conversations", "r8_ajmkmZq4JFMU31elTaIkBXcakllnvuA2463RL")

    time.sleep(1)

    # Let the user know that the system is ready
    msg.content = f"chain initialized! Ask me anything!"
    await msg.update()

    cl.user_session.set("chain", chain)
    cl.user_session.set("conversation_chain", conversation_chain)

@cl.on_message
async def on_message( message: cl.Message):
    
   
    user_input = message.content
    chain = cl.user_session.get("chain")
    conversation_chain = cl.user_session.get("conversation_chain")
    msg = cl.Message(content="processing question... ")
    await msg.send()
    conversation_chain.initilize_chain("./conversations", "r8_ajmkmZq4JFMU31elTaIkBXcakllnvuA2463RL")

    # check if the user has uploaded a file
    files = message.elements
    if len(files) > 0:
        for file in files:
            # Convert the PDF file to text
            msg.content = f"Processing `{files[0].name}`..."
            await msg.update()

            # Read the PDF file
            pdf_stream = BytesIO(files[0].content)
            pdf = PyPDF2.PdfReader(pdf_stream)
            pdf_text = ""
            for page in pdf.pages:
                pdf_text += page.extract_text()

            # Write the PDF text to a text file
            txt_filename = f"{files[0].name.split('.')[0]}.txt"
            txt_filepath = os.path.join("./documents", txt_filename)
            with open(txt_filepath, 'w', encoding='utf-8') as txt_file:
                txt_file.write(pdf_text)
            
            # Clean the text file
            clean_text_file(txt_filepath)


        # Initialize the chain
        await cl.make_async(chain.initilize_chain)("./documents", "r8_ajmkmZq4JFMU31elTaIkBXcakllnvuA2463RL")
        time.sleep(4)
    
    #response_document = chain.chain.invoke(user_input)
    #response_conversation = conversation_chain.chain.invoke(user_input)
    response_doc =  chain.retriever.get_relevant_documents(user_input)
    response_conv =  conversation_chain.retriever.get_relevant_documents(user_input)
    model_response = await cl.make_async(prompt_template.create_response)(user_input, response_doc, response_conv, chain.model)

    msg.content = ""
    await msg.update()

    for text in model_response:
        await msg.stream_token(text)
        time.sleep(0.1)


    
    await msg.send()
    conversation_chain.add_to_conversation(user_input, msg.content)

    
