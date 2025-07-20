from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, Document, StorageContext,
    load_index_from_storage, get_response_synthesizer
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.response_synthesizers import ResponseMode

from response_structures import ResearchResponse
from structured_prompt import CUSTOM_PROMPT_TEMPLATE

from pathlib import Path
from typing import cast
import json


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
        self.llm = Ollama(model=llm, request_timeout=300.0, context_window=4000)
        self.structured_llm = self.llm.as_structured_llm(ResearchResponse)

        self.documents: list[Document] = []
        self.vector_store: VectorStoreIndex  # Current VectorStore only create Indexes using Text, need to extend to MultiModal => Todo.
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model) 

        self.file_name: Path = Path(filename)
        self.file_registry: Path = ROOT / "assets"
        self.index_registry: Path = ROOT / "vector_store"
        self.file_path: Path = self.file_registry / filename

        self.query_engine = self.construct_query_engine()

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

        # Creating the Custom Index Retriever for the Store.
        custom_retriever = VectorIndexRetriever(self.vector_store, similarity_top_k=3)

        # Creating the Custom Response Synthesiser with the Prompt template SLLM.
        custom_response_synthesiser = get_response_synthesizer(
            self.structured_llm, response_mode=ResponseMode.COMPACT, text_qa_template=CUSTOM_PROMPT_TEMPLATE
        )

        # Compiling and Constructing the Custom QueryEngine.
        custom_query_eng = RetrieverQueryEngine.from_args(
            retriever=custom_retriever, response_synthesizer=custom_response_synthesiser, 
            response_mode=ResponseMode.COMPACT, llm=self.structured_llm
        )
        return custom_query_eng
    
    def run_query(self, user_prompt: str) -> str:
        """Runs a user prompt for query on the Query Engine."""
        response_obj = self.query_engine.query(user_prompt)
        response_json = json.loads(str(response_obj))
        research_response_output = ResearchResponse.model_validate(response_json)
        return research_response_output.model_dump_json(indent=4)


gemma_engine = QueryEngine()
for i in range(3):
    inp_prompt = input("Enter Query: ")
    response = gemma_engine.run_query(inp_prompt)
    print(response)