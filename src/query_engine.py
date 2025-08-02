# Core Imports
from llama_index.core import (
    VectorStoreIndex, Document, StorageContext, Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.vector_stores.types import VectorStoreQueryMode

# Ollama Specific Imports
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Project Module Imports
from src.config import settings
from src.response_structures import ResearchResponse
from src.structured_prompt import CUSTOM_PROMPT_TEMPLATE

# Miscellaneous Imports
from pathlib import Path
import json
import chromadb


# Global Configuration of the Settings for Llama-Index
Settings.llm = Ollama(
    model=settings.llm_model_name, request_timeout=300.0, 
    context_window=30000, additional_kwargs={"num_predict": 3072}
)
Settings.embed_model = OllamaEmbedding(settings.embedding_model_name)
Settings.transformations = [SentenceSplitter(chunk_size=1024, chunk_overlap=120)]


class QueryEngine:
    """This class implements the RAG pipeline that forms the backbone of the Research Companion application.
    
    It creates a vector index for the input files and constructs a Query Engine (soon extended to Chat Engine).
    The Query Engine encapsulates the end - to - end workflow executing the RAG pipeline with Gemma3n:e4b model."""

    def __init__(self, filepath: str) -> None:
        """Class Constructor."""
        # List of all the documents loaded from VectorStores
        self.documents: list[Document] = []
        # VectorStore Object
        self.vector_store: ChromaVectorStore
        # Persist Storage
        self.storage_context: StorageContext
        # Calculated Indexes
        self.index: VectorStoreIndex

        # VectorStore Path
        self.index_registry: Path = settings.vector_store_path
        # PDF File Path
        self.file_path: Path = Path(filepath)

        # ChromaDB Client
        self.chroma_client = chromadb.PersistentClient(path=self.index_registry)
        self.chroma_collection = self.chroma_client.get_or_create_collection("research_companion")

        # Chat Engine Parameters
        self.top_k = settings.top_k
        self.structured_llm = Settings.llm.as_structured_llm(ResearchResponse)
        self.memory_buffer = ChatMemoryBuffer.from_defaults(token_limit=30000)
        self.query_engine = self.construct_chat_engine()

    def check_index_exists(self) -> bool:
        """Checks if the index for a given file already exists."""
        result = self.chroma_collection.get(
            where={"file_path": str(self.file_path.resolve())},
            limit=1
        )
        return len(result["ids"]) > 0

    def construct_chat_engine(self):
        """Loads, Transforms and Indexes the input file / reloads them if exits and provides a query engine object."""

        # Accessing the ChromaVectorStore and setting the storage
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        if not self.check_index_exists():
            # Creating the Document from the specific file path
            self.documents = PyMuPDFReader().load_data(file_path=self.file_path)
            
            # Uniquely storing the documents in the DB using Filepath
            for doc in self.documents:
                doc.metadata["file_path"] = str(self.file_path.resolve())

            # Calculating the New Indexes for ChromaVectorStore
            self.index = VectorStoreIndex.from_documents(
                self.documents, embed_model=Settings.embed_model, 
                show_progress=True, storage_context=self.storage_context, 
                transformations=Settings.transformations
            )
        else:
            # Loading the Indexes
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store, storage_context=self.storage_context
            )

        # Loading the Calculated Indexes for the correct document
        filepath_filters = MetadataFilters(
            filters=[
                ExactMatchFilter(key="file_path", value=str(self.file_path.resolve()))
            ]
        )

        # Creating the Custom Retriever
        custom_retriever = VectorIndexRetriever(
            index=self.index, similarity_top_k=self.top_k, embed_model=Settings.embed_model,
            vector_store_query_mode=VectorStoreQueryMode.SEMANTIC_HYBRID, alpha=0.5, filters=filepath_filters
        )

        custom_chat_eng = ContextChatEngine.from_defaults(
            retriever=custom_retriever, llm=self.structured_llm, 
            context_prompt=CUSTOM_PROMPT_TEMPLATE, memory=self.memory_buffer
        )

        return custom_chat_eng
    
    def run_query(self, user_prompt: str) -> str:
        """Runs a user prompt for query on the Query Engine."""
        response_obj = self.query_engine.chat(user_prompt)
        response_json = json.loads(str(response_obj))
        research_response_output = ResearchResponse.model_validate(response_json)
        try:
            return research_response_output.model_dump_json(indent=4)
        except Exception as e:
            print(f"Error during query: {e}")
            error_response = ResearchResponse(
                answer="""Sorry, I have encountered an issue processing your request.
                This can happen sometimes when a document is loaded for the first time or 
                the first query on a new document.
                Could you please try asking the question again.""",
                citations=[]
            )
            return error_response.model_dump_json(indent=4)


# ==== Unit Test ====
# if __name__ == "__main__":
#     gemma_engine = QueryEngine()
#     for i in range(3):
#         inp_prompt = input("Enter Query: ")
#         response = gemma_engine.run_query(inp_prompt)
#         print(response)