"""LangGraph implementation for the Modular RAG system."""
import logging
from typing import TypedDict, List, Dict, Any
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END

from src.rag_systems import modular_rag as modular_rag_utils
from src.utils.llm_manager import LLMManager
from src.utils.vector_store import VectorStoreManager
from src.rag_systems.modular_rag import BM25
from src.config.settings import CONFIDENCE_THRESHOLD

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
    logger.info("="*80)
    logger.info("🚀 MODULAR RAG SYSTEM STARTED")
    logger.info("="*80)
    logger.info("--- INITIAL STEP: Query Pre-processing ---")
    query = state["query"]
    
    logger.info(f"🔍 PREPROCESSING DEBUG")
    logger.info(f"   └─ Original Query: '{query}'")
    
    expansion_result = modular_rag_utils.expand_query(query)
    classification_result = modular_rag_utils.classify_query(query)
    
    logger.info(f"🔍 Query Expansion:")
    logger.info(f"   ├─ Expanded Query: '{expansion_result['expanded_query']}'")
    logger.info(f"   └─ Expansion Terms: {expansion_result['expansion_terms']}")
    
    logger.info(f"🏷️ Query Classification:")
    logger.info(f"   ├─ Query Type: {classification_result['query_type']}")
    logger.info(f"   ├─ Classification Confidence: {classification_result['confidence']:.3f}")
    logger.info(f"   └─ All Scores: {classification_result['classification_scores']}")
    
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
    logger.info("-"*80)
    logger.info("--- ROUTING DECISION ---")
    query_type = state["query_type"]
    
    logger.info(f"🔀 Query Routing Analysis:")
    logger.info(f"   ├─ Query type: {query_type}")
    logger.info(f"   ├─ Query length: {len(state['query'])} chars")
    logger.info(f"   └─ Decision criteria: general type + <20 chars = direct answer")
    
    # Heuristic: If the query is classified as general and is short,
    # it's likely a conversational query that doesn't need retrieval.
    if query_type == "general" and len(state["query"]) < 20:
        logger.info("🎯 ROUTING DECISION: Direct Answer (Simple query)")
        logger.info("="*80)
        return "direct_answer"
    else:
        logger.info("🎯 ROUTING DECISION: RAG Pipeline (Complex query)")
        logger.info("="*80)
        return "rag_path"

def retrieve_and_process_node(state: ModularRagState, vector_store_manager: VectorStoreManager, bm25_index: BM25, bm25_docs: List[Document]):
    """Retrieves, merges, and post-processes documents."""
    current_iteration = state['iteration'] + 1
    logger.info("🔄"*20)
    logger.info(f"🔄 ITERATION {current_iteration} STARTED")
    logger.info("🔄"*20)
    logger.info(f"--- ITERATION {current_iteration}: Document Retrieval & Processing ---")
    query = state["expanded_query"]
    k = state["retrieval_k"]

    logger.info(f"🔍 RETRIEVAL DEBUG - Iteration {current_iteration}")
    logger.info(f"   ├─ Query: '{query}'")
    logger.info(f"   └─ Retrieval K: {k}")

    # Hybrid retrieval
    semantic_docs = modular_rag_utils.retrieve_semantic(vector_store_manager, query, k)
    keyword_docs = modular_rag_utils.retrieve_keyword(bm25_index, bm25_docs, query, k)
    
    logger.info(f"📚 Retrieval Results:")
    logger.info(f"   ├─ Semantic docs: {len(semantic_docs)}")
    logger.info(f"   └─ Keyword docs: {len(keyword_docs)}")
    
    # Merge and deduplicate
    all_docs = semantic_docs + keyword_docs
    seen = set()
    unique_docs = [doc for doc in all_docs if doc.page_content not in seen and not seen.add(doc.page_content)]
    
    # Post-process
    processed_docs = modular_rag_utils.filter_and_diversify(unique_docs, max_docs=5)
    
    logger.info(f"🔄 Processing Results:")
    logger.info(f"   ├─ Total retrieved: {len(all_docs)}")
    logger.info(f"   ├─ After deduplication: {len(unique_docs)}")
    logger.info(f"   └─ After filtering & diversifying: {len(processed_docs)}")
    
    # Log document sources for debugging
    if processed_docs:
        sources = [doc.metadata.get("source", "unknown")[:50] + "..." if len(doc.metadata.get("source", "unknown")) > 50 
                  else doc.metadata.get("source", "unknown") for doc in processed_docs]
        logger.info(f"📄 Selected Document Sources:")
        for i, source in enumerate(sources, 1):
            logger.info(f"   {i}. {source}")
    
    # Append to all retrieved docs
    current_all_docs = state.get("all_retrieved_docs", [])
    updated_all_docs = current_all_docs + processed_docs
    
    logger.info(f"📈 Cumulative Progress:")
    logger.info(f"   ├─ Previous total docs: {len(current_all_docs)}")
    logger.info(f"   └─ New total docs: {len(updated_all_docs)}")
    
    return {"retrieved_docs": processed_docs, "all_retrieved_docs": updated_all_docs}

def generate_node(state: ModularRagState, llm_manager: LLMManager):
    """Generates an answer and estimates confidence."""
    current_iteration = state['iteration'] + 1
    logger.info("-"*40)
    logger.info(f"--- ITERATION {current_iteration}: Answer Generation ---")
    retrieved_docs = state["retrieved_docs"]
    
    logger.info(f"📝 GENERATION DEBUG - Iteration {current_iteration}")
    logger.info(f"   ├─ Input docs: {len(retrieved_docs)}")
    logger.info(f"   ├─ Query type: {state['query_type']}")
    logger.info(f"   └─ Classification confidence: {state['classification_confidence']:.3f}")
    
    answer_stream = modular_rag_utils.generate_answer_stream(
        llm_manager, state["query"], retrieved_docs, state["query_type"]
    )
    answer = "".join(list(answer_stream))
    
    logger.info(f"✏️ Generated Answer Preview:")
    logger.info(f"   ├─ Length: {len(answer)} chars")
    logger.info(f"   └─ Preview: '{answer[:150]}{'...' if len(answer) > 150 else ''}'")
    
    confidence = modular_rag_utils.estimate_confidence(
        llm_manager, answer, retrieved_docs, state["classification_confidence"]
    )
    
    logger.info(f"🎯 Generation Summary:")
    logger.info(f"   ├─ Final confidence: {confidence:.3f}")
    logger.info(f"   └─ Threshold needed: {CONFIDENCE_THRESHOLD}")
    
    return {"answer": answer, "final_confidence": confidence}

def direct_answer_node(state: ModularRagState, llm_manager: LLMManager):
    """Generates a direct response without RAG for simple questions."""
    logger.info("💬 DIRECT ANSWER MODE ACTIVATED")
    logger.info("="*80)
    logger.info("📝 Generating Direct Answer (No RAG Pipeline)")
    query = state["query"]
    
    logger.info(f"🔍 Direct Answer Debug:")
    logger.info(f"   ├─ Query: '{query}'")
    logger.info(f"   ├─ Mode: Simple conversational response")
    logger.info(f"   └─ Pipeline: Direct LLM → Answer")
    
    prompt = f"Please answer the following question directly and concisely.\n\nQuestion: {query}\n\nAnswer:"
    answer = llm_manager.generate_response(prompt=prompt, context="")
    
    logger.info(f"✏️ Generated Direct Answer:")
    logger.info(f"   ├─ Length: {len(answer)} chars")
    logger.info(f"   └─ Preview: '{answer[:150]}{'...' if len(answer) > 150 else ''}'")
    
    logger.info("="*80)
    logger.info("✅ MODULAR RAG SYSTEM COMPLETED (Direct Answer)")
    logger.info("="*80)
    
    # Set confidence high to prevent iteration, and empty docs
    return {"answer": answer, "final_confidence": 1.0, "all_retrieved_docs": []}

def iteration_control_node(state: ModularRagState) -> str:
    """Decides whether to continue iterating or end."""
    current_iteration = state["iteration"] + 1
    logger.info("-"*40)
    logger.info(f"--- ITERATION {current_iteration}: Control Decision ---")
    iteration = state["iteration"]
    confidence = state["final_confidence"]
    max_iterations = state["max_iterations"]
    
    logger.info(f"🔍 ITERATION CONTROL DEBUG:")
    logger.info(f"   ├─ Current confidence: {confidence:.3f}")
    logger.info(f"   ├─ Threshold setting: {CONFIDENCE_THRESHOLD}")
    logger.info(f"   ├─ Iteration: {iteration}/{max_iterations-1}")
    logger.info(f"   └─ Comparison: {confidence:.3f} >= {CONFIDENCE_THRESHOLD} = {confidence >= CONFIDENCE_THRESHOLD}")
    
    if modular_rag_utils.check_iteration_stop(confidence, iteration, max_iterations):
        logger.info("🛑 ITERATION DECISION: STOPPING")
        logger.info("   └─ Reason: Confidence threshold met or max iterations reached")
        logger.info("🔄"*20)
        
        # Final summary for completed RAG pipeline
        total_docs = len(state.get("all_retrieved_docs", []))
        final_answer_length = len(state.get("answer", ""))
        
        logger.info("="*80)
        logger.info("✅ MODULAR RAG SYSTEM COMPLETED")
        logger.info("="*80)
        logger.info(f"🎯 Final Summary:")
        logger.info(f"   ├─ Pipeline: Preprocess → Route → [Iterate: Retrieve → Generate → Control]")
        logger.info(f"   ├─ Total iterations completed: {current_iteration}")
        logger.info(f"   ├─ Final confidence: {confidence:.3f}")
        logger.info(f"   ├─ Total documents used: {total_docs}")
        logger.info(f"   ├─ Final answer length: {final_answer_length} chars")
        logger.info(f"   └─ Features: Adaptive retrieval, confidence-based iteration, hybrid search")
        logger.info("="*80)
        return "end"
    else:
        logger.info(f"🔄 ITERATION DECISION: CONTINUING")
        logger.info(f"   └─ Reason: Confidence {confidence:.3f} < {CONFIDENCE_THRESHOLD} threshold")
        logger.info("🔄"*20)
        return "continue"

def prepare_next_iteration_node(state: ModularRagState):
    """Updates state for the next iteration."""
    current_iteration = state.get("iteration", 0) + 1
    next_iteration = current_iteration + 1
    logger.info("🔧 PREPARING FOR NEXT ITERATION")
    logger.info(f"   ├─ Current iteration: {current_iteration}")
    logger.info(f"   ├─ Next iteration: {next_iteration}")
    logger.info(f"   └─ Adjusting search parameters...")
    
    iteration = state.get("iteration", 0) + 1
    current_k = state.get("retrieval_k", 8)
    new_k = current_k + 4  # Increase K more aggressively
    
    logger.info(f"📈 Parameter Updates:")
    logger.info(f"   ├─ Retrieval K: {current_k} → {new_k}")
    logger.info(f"   └─ Strategy: Expand search scope for better results")
    
    return {"iteration": iteration, "retrieval_k": new_k}

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