"""LangGraph implementation for the Modular RAG system."""
import logging
from typing import TypedDict, List, Dict, Any
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END

from src.rag_systems import modular_rag as modular_rag_utils
from src.utils.llm_manager import LLMManager
from src.utils.vector_store import VectorStoreManager
from src.rag_systems.modular_rag import BM25

logger = logging.getLogger(__name__)

class ModularRagState(TypedDict):
    """The state for the Modular RAG graph."""
    # Inputs
    query: str
    max_iterations: int
    
    # Pre-processing state
    expanded_query: str
    expansion_terms: List[str]
    query_type: str
    classification_confidence: float
    classification_scores: Dict[str, float]
    
    # Iteration state
    iteration: int
    retrieval_k: int
    
    # Retrieval state
    retrieved_docs: List[Document]
    
    # Generation state
    answer: str
    final_confidence: float
    
    # Final result collector
    all_retrieved_docs: List[Document]


def preprocess_node(state: ModularRagState):
    """Expands and classifies the query."""
    logger.info("--- Modular RAG: Pre-processing ---")
    query = state["query"]
    expansion_result = modular_rag_utils.expand_query(query)
    classification_result = modular_rag_utils.classify_query(query)
    
    return {
        "expanded_query": expansion_result["expanded_query"],
        "expansion_terms": expansion_result["expansion_terms"],
        "query_type": classification_result["query_type"],
        "classification_confidence": classification_result["confidence"],
        "classification_scores": classification_result["classification_scores"],
        "iteration": 0,
        "retrieval_k": 8,
        "all_retrieved_docs": [],
    }

def route_query_node(state: ModularRagState) -> str:
    """Decides whether to answer directly or use the RAG pipeline."""
    logger.info("--- Modular RAG: Routing Query ---")
    query_type = state["query_type"]
    
    # Heuristic: If the query is classified as general and is short,
    # it's likely a conversational query that doesn't need retrieval.
    if query_type == "general" and len(state["query"]) < 20:
        logger.info("Query is simple. Routing to direct answer.")
        return "direct_answer"
    else:
        logger.info("Query is complex. Routing to RAG path.")
        return "rag_path"

def retrieve_and_process_node(state: ModularRagState, vector_store_manager: VectorStoreManager, bm25_index: BM25, bm25_docs: List[Document]):
    """Retrieves, merges, and post-processes documents."""
    logger.info(f"--- Modular RAG: Iteration {state['iteration'] + 1} - Retrieval & Processing ---")
    query = state["expanded_query"]
    k = state["retrieval_k"]

    # Hybrid retrieval
    semantic_docs = modular_rag_utils.retrieve_semantic(vector_store_manager, query, k)
    keyword_docs = modular_rag_utils.retrieve_keyword(bm25_index, bm25_docs, query, k)
    
    # Merge and deduplicate
    all_docs = semantic_docs + keyword_docs
    seen = set()
    unique_docs = [doc for doc in all_docs if doc.page_content not in seen and not seen.add(doc.page_content)]
    
    # Post-process
    processed_docs = modular_rag_utils.filter_and_diversify(unique_docs, max_docs=5)
    
    # Append to all retrieved docs
    current_all_docs = state.get("all_retrieved_docs", [])
    updated_all_docs = current_all_docs + processed_docs
    
    return {"retrieved_docs": processed_docs, "all_retrieved_docs": updated_all_docs}

def generate_node(state: ModularRagState, llm_manager: LLMManager):
    """Generates an answer and estimates confidence."""
    logger.info(f"--- Modular RAG: Iteration {state['iteration'] + 1} - Generation ---")
    retrieved_docs = state["retrieved_docs"]
    answer_stream = modular_rag_utils.generate_answer_stream(
        llm_manager, state["query"], retrieved_docs, state["query_type"]
    )
    answer = "".join(list(answer_stream))
    
    confidence = modular_rag_utils.estimate_confidence(
        llm_manager, answer, retrieved_docs, state["classification_confidence"]
    )
    
    return {"answer": answer, "final_confidence": confidence}

def direct_answer_node(state: ModularRagState, llm_manager: LLMManager):
    """Generates a direct response without RAG for simple questions."""
    logger.info("--- Modular RAG: Generating Direct Answer ---")
    query = state["query"]
    prompt = f"Please answer the following question directly and concisely.\n\nQuestion: {query}\n\nAnswer:"
    answer = llm_manager.generate_response(prompt=prompt, context="")
    # Set confidence high to prevent iteration, and empty docs
    return {"answer": answer, "final_confidence": 1.0, "all_retrieved_docs": []}

def iteration_control_node(state: ModularRagState) -> str:
    """Decides whether to continue iterating or end."""
    logger.info(f"--- Modular RAG: Iteration Control ---")
    iteration = state["iteration"]
    confidence = state["final_confidence"]
    max_iterations = state["max_iterations"]
    
    logger.info(f"Checking to stop: confidence={confidence:.2f}, iteration={iteration}, max_iterations={max_iterations}")
    
    if modular_rag_utils.check_iteration_stop(confidence, iteration, max_iterations):
        logger.info("Confidence threshold met or max iterations reached. Ending.")
        return "end"
    else:
        logger.info(f"Confidence {confidence:.2f} is low. Continuing to next iteration.")
        return "continue"

def prepare_next_iteration_node(state: ModularRagState):
    """Updates state for the next iteration."""
    logger.info("--- Modular RAG: Preparing for next iteration ---")
    iteration = state.get("iteration", 0) + 1
    k = state.get("retrieval_k", 8) + 4  # Increase K more aggressively
    return {"iteration": iteration, "retrieval_k": k}

def create_modular_rag_graph(llm_manager: LLMManager, vector_store_manager: VectorStoreManager, bm25_index: BM25, bm25_docs: List[Document]):
    """Creates the Modular RAG graph with an iterative loop."""
    workflow = StateGraph(ModularRagState)
    
    # Add nodes
    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("retrieve_and_process", lambda state: retrieve_and_process_node(state, vector_store_manager, bm25_index, bm25_docs))
    workflow.add_node("generate", lambda state: generate_node(state, llm_manager))
    workflow.add_node("direct_answer", lambda state: direct_answer_node(state, llm_manager))
    workflow.add_node("prepare_next_iteration", prepare_next_iteration_node)
    
    # Build the graph
    workflow.set_entry_point("preprocess")

    # Add conditional routing logic based on query classification
    workflow.add_conditional_edges(
        "preprocess",
        route_query_node,
        {
            "rag_path": "retrieve_and_process",
            "direct_answer": "direct_answer"
        }
    )
    
    # Define the two paths
    
    # 1. Direct answer path
    workflow.add_edge("direct_answer", END)
    
    # 2. RAG path with iteration
    workflow.add_edge("retrieve_and_process", "generate")
    workflow.add_conditional_edges(
        "generate",
        iteration_control_node,
        {
            "continue": "prepare_next_iteration",
            "end": END
        }
    )
    workflow.add_edge("prepare_next_iteration", "retrieve_and_process")
    
    graph = workflow.compile()
    logger.info("Modular RAG graph compiled successfully.")
    return graph 