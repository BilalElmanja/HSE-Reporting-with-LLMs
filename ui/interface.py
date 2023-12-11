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
import replicate
from datetime import datetime

os.environ["REPLICATE_API_KEY"] = "r8_eehqKGwhuv0jF2KLISsOwIJo0eupYm60oj12m"


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
                    "max_new_tokens": 128,
                    "min_new_tokens": -1
            }
            
        )
        response = ""
        for text in output:
            response += text
        return response
        
        
    

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
        return docs

    def retriever_big_chunks(self, docs):
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        vectorstore = Chroma(collection_name="split_parents", embedding_function=self.bge_embeddings)
        store = InMemoryStore()

        big_chunks_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
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
                    "max_new_tokens": 128,
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


class ConversationMemory:
    def __init__(self, base_directory):
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
            file.write(f"User: {user_input}\n")
            file.write(f"Agent: {agent_response}\n")



        
    


import chainlit as cl

chain =  Document_Chain()
conversation_memory = ConversationMemory(base_directory='./documents')

@cl.on_chat_start
async def on_chat_start():
    
    chain.initilize_chain("./documents", "r8_eehqKGwhuv0jF2KLISsOwIJo0eupYm60oj12m")
    # Usage
    
    conversation_memory.start_new_conversation()




@cl.on_message
async def on_message( message: cl.Message):
   
    user_input = message.content
    msg = cl.Message(content="")
    response = chain.chain.invoke(user_input)
    for text in response['result']:
        await msg.stream_token(text)
        time.sleep(0.01)



    await msg.send()
