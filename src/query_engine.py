from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, Document, StorageContext,
    load_index_from_storage, get_response_synthesizer, Settings
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.response_synthesizers import ResponseMode

from src.config import settings
from src.response_structures import ResearchResponse
from src.structured_prompt import CUSTOM_PROMPT_TEMPLATE

from pathlib import Path
from typing import cast
import json


# Global Configuration of the Settings for Llama-Index
Settings.llm = Ollama(model=settings.llm_model_name, request_timeout=300.0, context_window=4000)
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
        self.query_engine = self.construct_query_engine()

    def check_index_exists(self) -> bool:
        """Checks if the index for a given file already exists."""
        return (self.index_registry / self.file_path.stem).exists()

    def construct_query_engine(self) -> BaseQueryEngine:
        """Loads, Transforms and Indexes the input file / reloads them if exits and provides a query engine object."""
        index_dir = self.index_registry / self.file_path.stem

        if not self.check_index_exists():
            # Creating the Document from the specific file path
            self.documents = SimpleDirectoryReader(input_files=[self.file_path]).load_data()
            # Calculating the Indexes
            self.vector_store = VectorStoreIndex.from_documents(self.documents, embed_model=Settings.embed_model, show_progress=True)
            # Storing the Indexes for reuse
            self.vector_store.storage_context.persist(persist_dir=index_dir)
        else:
            # Loading the Indexes
            storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
            intermediate_index = load_index_from_storage(storage_context=storage_context, embed_model=Settings.embed_model)
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


# ==== Unit Test ====
# if __name__ == "__main__":
#     gemma_engine = QueryEngine()
#     for i in range(3):
#         inp_prompt = input("Enter Query: ")
#         response = gemma_engine.run_query(inp_prompt)
#         print(response)