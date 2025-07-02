"""LangGraph implementation for the Advanced RAG system."""
import logging
from typing import TypedDict, List, Dict, Any, Tuple
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END

from src.rag_systems import advanced_rag as advanced_rag_utils
from src.utils.llm_manager import LLMManager
from src.utils.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class AdvancedRagState(TypedDict):
    """The state for the Advanced RAG graph."""
    query: str
    k: int
    rerank_top_k: int
    
    # Pre-processing results
    preprocessing_details: Dict[str, Any]
    optimized_query: str

    # Retrieval results
    docs_with_scores: List[Tuple[Document, float]]
    
    # Reranking results
    reranked_docs: List[Document]
    
    # Compression results
    compressed_context: str
    compression_ratio: float
    
    # Generation result
    answer: str


def preprocess_query_node(state: AdvancedRagState):
    """Preprocesses and optimizes the user query."""
    logger.info("--- Advanced RAG: Preprocessing query ---")
    query = state["query"]
    result = advanced_rag_utils.preprocess_query(query)
    logger.info(f"Optimized query: {result['enhanced_query']}")
    return {
        "preprocessing_details": result,
        "optimized_query": result["enhanced_query"]
    }

def retrieve_node(state: AdvancedRagState, vector_store_manager: VectorStoreManager):
    """Retrieves documents based on the optimized query."""
    logger.info("--- Advanced RAG: Retrieving documents ---")
    docs_with_scores = advanced_rag_utils.retrieve_with_scores(
        vector_store_manager, state["optimized_query"], state["k"]
    )
    logger.info(f"Retrieved {len(docs_with_scores)} documents initially.")
    return {"docs_with_scores": docs_with_scores}

def rerank_node(state: AdvancedRagState):
    """Reranks the retrieved documents."""
    logger.info("--- Advanced RAG: Reranking documents ---")
    reranked_docs = advanced_rag_utils.rerank_documents(
        state["optimized_query"], state["docs_with_scores"], state["rerank_top_k"]
    )
    logger.info(f"Reranked to top {len(reranked_docs)} documents.")
    return {"reranked_docs": reranked_docs}
    
def compress_context_node(state: AdvancedRagState):
    """Compresses the context from reranked documents."""
    logger.info("--- Advanced RAG: Compressing context ---")
    result = advanced_rag_utils.compress_context(state["reranked_docs"])
    logger.info(f"Context compressed with ratio: {result['compression_ratio']:.2f}")
    return {
        "compressed_context": result["compressed_context"],
        "compression_ratio": result["compression_ratio"]
    }

def generate_node(state: AdvancedRagState, llm_manager: LLMManager):
    """Generates the final answer."""
    logger.info("--- Advanced RAG: Generating answer ---")
    answer_stream = advanced_rag_utils.generate_answer_stream(
        llm_manager, state["query"], state["compressed_context"]
    )
    full_answer = "".join(list(answer_stream))
    return {"answer": full_answer}
    
def create_advanced_rag_graph(llm_manager: LLMManager, vector_store_manager: VectorStoreManager):
    """Creates the Advanced RAG graph."""
    workflow = StateGraph(AdvancedRagState)
    
    workflow.add_node("preprocess_query", preprocess_query_node)
    workflow.add_node("retrieve", lambda state: retrieve_node(state, vector_store_manager))
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("compress_context", compress_context_node)
    workflow.add_node("generate", lambda state: generate_node(state, llm_manager))
    
    workflow.set_entry_point("preprocess_query")
    workflow.add_edge("preprocess_query", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "compress_context")
    workflow.add_edge("compress_context", "generate")
    workflow.add_edge("generate", END)
    
    graph = workflow.compile()
    logger.info("Advanced RAG graph compiled successfully.")
    return graph 