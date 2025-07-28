# Core Imports
from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, Document, StorageContext,
    load_index_from_storage, Settings
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine.types import ChatMode

# Ollama Specific Imports
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Project Module Imports
from src.config import settings
from src.response_structures import ResearchResponse
from src.structured_prompt import CUSTOM_PROMPT_TEMPLATE

# Miscellaneous Imports
from pathlib import Path
from typing import cast
import json


# Global Configuration of the Settings for Llama-Index
Settings.llm = Ollama(model=settings.llm_model_name, request_timeout=300.0, context_window=20000)
Settings.embed_model = OllamaEmbedding(settings.embedding_model_name)


class QueryEngine:
    """This class implements the RAG pipeline that forms the backbone of the Research Companion application.
    
    It creates a vector index for the input files and constructs a Query Engine (soon extended to Chat Engine).
    The Query Engine encapsulates the end - to - end workflow executing the RAG pipeline with Gemma3n:e4b model."""

    def __init__(self, filepath: str) -> None:
        """Class Constructor."""
        self.documents: list[Document] = []
        self.vector_store: VectorStoreIndex

        self.index_registry: Path = settings.vector_store_path
        self.file_path: Path = Path(filepath)

        self.structured_llm = Settings.llm.as_structured_llm(ResearchResponse)
        self.memory_buffer = ChatMemoryBuffer.from_defaults(token_limit=20000)
        self.query_engine = self.construct_query_engine()

    def check_index_exists(self) -> bool:
        """Checks if the index for a given file already exists."""
        return (self.index_registry / self.file_path.stem).exists()

    def construct_query_engine(self):
        """Loads, Transforms and Indexes the input file / reloads them if exits and provides a query engine object."""
        index_dir = self.index_registry / self.file_path.stem

        if not self.check_index_exists():
            # Creating the Document from the specific file path
            self.documents = SimpleDirectoryReader(input_files=[self.file_path]).load_data()
            # Calculating the Indexes
            self.vector_store = VectorStoreIndex.from_documents(self.documents, embed_model=Settings.embed_model, show_progress=True)
            # self.vector_store = MultiModalVectorStoreIndex.from_documents(
            #     documents=self.documents, show_progress=True
            # )
            # Storing the Indexes for reuse
            self.vector_store.storage_context.persist(persist_dir=index_dir)
        else:
            # Loading the Indexes
            storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
            intermediate_index = load_index_from_storage(
                storage_context=storage_context, embed_model=Settings.embed_model
            )
            self.vector_store = cast(VectorStoreIndex, intermediate_index)

        # Creating a contextual chat engine from the index
        custom_chat_eng = self.vector_store.as_chat_engine(
            chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT, memory=self.memory_buffer, 
            llm=self.structured_llm, context_prompt=CUSTOM_PROMPT_TEMPLATE
        )

        return custom_chat_eng
    
    def run_query(self, user_prompt: str) -> str:
        """Runs a user prompt for query on the Query Engine."""
        response_obj = self.query_engine.chat(user_prompt)
        response_json = json.loads(str(response_obj))
        research_response_output = ResearchResponse.model_validate(response_json)
        return research_response_output.model_dump_json(indent=4)


# ==== Unit Test ====
# if __name__ == "__main__":
#     gemma_engine = QueryEngine()
#     for i in range(3):
#         inp_prompt = input("Enter Query: ")
#         response = gemma_engine.run_query(inp_prompt)
#         print(response)