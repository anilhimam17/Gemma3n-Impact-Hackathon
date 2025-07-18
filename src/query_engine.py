from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, Document, StorageContext, 
    load_index_from_storage, get_response_synthesizer
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.response_synthesizers import ResponseMode

from pathlib import Path
from typing import cast


# Root Path
ROOT: Path = Path.cwd()
# Default Embedding Model
EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
# Default LLM
GEMMA3N: str = "gemma3n:e4b"


class QueryEngine:
    """This class implements the RAG pipeline that forms the backbone of the Research Companion application.
    
    It creates a vector index for the input files and constructs a Query Engine (soon extended to Chat Engine).
    The Query Engine encapsulates the end - to - end workflow executing the RAG pipeline with Gemma3n:e4b model."""

    def __init__(self, filename: str = "YOLOv7.pdf", embed_model: str = EMBED_MODEL, llm: str = GEMMA3N) -> None:
        """Class Constructor."""
        self.llm = llm

        self.documents: list[Document] = []
        self.vector_store: VectorStoreIndex
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model) 

        self.file_name: Path = Path(filename)
        self.file_registry: Path = ROOT / "assets"
        self.index_registry: Path = ROOT / "vector_store"
        self.file_path: Path = self.file_registry / filename

    def check_index_exists(self) -> bool:
        """Checks if the index for a given file already exists."""
        return (self.index_registry / self.file_name.stem).exists()

    def construct_query_engine(self) -> BaseQueryEngine:
        """Loads, Transforms and Indexes the input file / reloads them if exits and provides a query engine object."""
        if not self.check_index_exists():
            # Creating the Document
            self.documents = SimpleDirectoryReader(self.file_registry).load_data()
            # Calculating the Indexes
            self.vector_store = VectorStoreIndex.from_documents(self.documents, embed_model=self.embed_model, show_progress=True)
            # Storing the Indexes for reuse
            self.vector_store.storage_context.persist(persist_dir=self.index_registry / self.file_name.stem)
        else:
            # Loading the Indexes
            storage_context = StorageContext.from_defaults(persist_dir=str(self.index_registry / self.file_name.stem))
            intermediate_index = load_index_from_storage(storage_context=storage_context, embed_model=self.embed_model)
            self.vector_store = cast(VectorStoreIndex, intermediate_index)

        # Creating the Query Engine
        custom_retriever = VectorIndexRetriever(self.vector_store, similarity_top_k=5)
        custom_response_synthesiser = get_response_synthesizer(
            llm=Ollama(self.llm, request_timeout=120.0, context_window=2000)
        )
        custom_query_eng = RetrieverQueryEngine.from_args(
            retriever=custom_retriever, response_synthesizer=custom_response_synthesiser, 
            response_mode=ResponseMode.TREE_SUMMARIZE, llm=Ollama(self.llm, request_timeout=120.0, context_window=2000)
        )
        return custom_query_eng
    
    def start_query_engine(self) -> None:
        """Starts the query engine."""
        self.query_engine = self.construct_query_engine()
    
    def run_query(self, user_prompt: str) -> RESPONSE_TYPE:
        """Runs a user prompt for query on the Query Engine."""
        self.start_query_engine()
        response = self.query_engine.query(user_prompt)
        return response


gemma_engine = QueryEngine()
print(gemma_engine.run_query("Generate a simplified abstract for this paper with the main focus of Object Detection and its contributions to real-time Object Detection."))
