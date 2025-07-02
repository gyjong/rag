"""LangGraph implementation for the Naive RAG system."""
import logging
from typing import TypedDict, List
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END

from src.rag_systems import naive_rag as naive_rag_utils
from src.utils.llm_manager import LLMManager
from src.utils.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class NaiveRagState(TypedDict):
    """The state for the Naive RAG graph."""
    query: str
    k: int
    documents: List[Document]
    answer: str

def retrieve_node(state: NaiveRagState, vector_store_manager: VectorStoreManager):
    """Retrieves documents."""
    logger.info("--- Naive RAG: Retrieving documents ---")
    documents = naive_rag_utils.retrieve_naive(vector_store_manager, state["query"], state["k"])
    return {"documents": documents}

def generate_node(state: NaiveRagState, llm_manager: LLMManager):
    """Generates the answer."""
    logger.info("--- Naive RAG: Generating answer ---")
    answer_stream = naive_rag_utils.generate_naive_answer_stream(
        llm_manager, state["query"], state["documents"]
    )
    full_answer = "".join(list(answer_stream))
    return {"answer": full_answer}

def create_naive_rag_graph(llm_manager: LLMManager, vector_store_manager: VectorStoreManager):
    """Creates the Naive RAG graph."""
    workflow = StateGraph(NaiveRagState)
    
    workflow.add_node("retrieve", lambda state: retrieve_node(state, vector_store_manager))
    workflow.add_node("generate", lambda state: generate_node(state, llm_manager))
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    graph = workflow.compile()
    logger.info("Naive RAG graph compiled successfully.")
    return graph 