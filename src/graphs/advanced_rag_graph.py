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
    logger.info("="*80)
    logger.info("🚀 ADVANCED RAG SYSTEM STARTED")
    logger.info("="*80)
    logger.info("--- STEP 1/5: Query Preprocessing ---")
    query = state["query"]
    
    logger.info(f"🔍 PREPROCESSING DEBUG - Advanced RAG")
    logger.info(f"   └─ Original Query: '{query}'")
    
    result = advanced_rag_utils.preprocess_query(query)
    
    logger.info(f"🔍 Query Enhancement:")
    logger.info(f"   ├─ Enhanced Query: '{result['enhanced_query']}'")
    logger.info(f"   ├─ Enhancement Details: {result.get('details', 'N/A')}")
    logger.info(f"   └─ Processing Method: {result.get('method', 'standard')}")
    
    return {
        "preprocessing_details": result,
        "optimized_query": result["enhanced_query"]
    }

def retrieve_node(state: AdvancedRagState, vector_store_manager: VectorStoreManager):
    """Retrieves documents based on the optimized query."""
    logger.info("-"*80)
    logger.info("--- STEP 2/5: Document Retrieval ---")
    
    logger.info(f"🔍 RETRIEVAL DEBUG - Advanced RAG")
    logger.info(f"   ├─ Query: '{state['optimized_query']}'")
    logger.info(f"   └─ Retrieval K: {state['k']}")
    
    docs_with_scores = advanced_rag_utils.retrieve_with_scores(
        vector_store_manager, state["optimized_query"], state["k"]
    )
    
    logger.info(f"📚 Retrieval Results:")
    logger.info(f"   └─ Retrieved docs: {len(docs_with_scores)}")
    
    if docs_with_scores:
        avg_score = sum(score for _, score in docs_with_scores) / len(docs_with_scores)
        max_score = max(score for _, score in docs_with_scores)
        min_score = min(score for _, score in docs_with_scores)
        logger.info(f"📊 Score Statistics:")
        logger.info(f"   ├─ Max score: {max_score:.3f}")
        logger.info(f"   ├─ Min score: {min_score:.3f}")
        logger.info(f"   └─ Avg score: {avg_score:.3f}")
        
        # Log document sources
        sources = [doc.metadata.get("source", "unknown")[:50] + "..." if len(doc.metadata.get("source", "unknown")) > 50 
                  else doc.metadata.get("source", "unknown") for doc, _ in docs_with_scores[:5]]
        logger.info(f"📄 Top Document Sources:")
        for i, source in enumerate(sources, 1):
            logger.info(f"   {i}. {source}")
    
    return {"docs_with_scores": docs_with_scores}

def rerank_node(state: AdvancedRagState):
    """Reranks the retrieved documents."""
    logger.info("-"*80)
    logger.info("--- STEP 3/5: Document Reranking ---")
    
    logger.info(f"🔄 RERANKING DEBUG - Advanced RAG")
    logger.info(f"   ├─ Input docs: {len(state['docs_with_scores'])}")
    logger.info(f"   └─ Target rerank top-k: {state['rerank_top_k']}")
    
    reranked_docs = advanced_rag_utils.rerank_documents(
        state["optimized_query"], state["docs_with_scores"], state["rerank_top_k"]
    )
    
    logger.info(f"🎯 Reranking Results:")
    logger.info(f"   ├─ Reranked docs: {len(reranked_docs)}")
    logger.info(f"   └─ Reduction: {len(state['docs_with_scores'])} → {len(reranked_docs)}")
    
    if reranked_docs:
        # Log reranked document sources
        sources = [doc.metadata.get("source", "unknown")[:50] + "..." if len(doc.metadata.get("source", "unknown")) > 50 
                  else doc.metadata.get("source", "unknown") for doc in reranked_docs]
        logger.info(f"📄 Reranked Document Sources:")
        for i, source in enumerate(sources, 1):
            logger.info(f"   {i}. {source}")
    
    return {"reranked_docs": reranked_docs}
    
def compress_context_node(state: AdvancedRagState):
    """Compresses the context from reranked documents."""
    logger.info("-"*80)
    logger.info("--- STEP 4/5: Context Compression ---")
    
    original_length = sum(len(doc.page_content) for doc in state["reranked_docs"])
    logger.info(f"🗜️ COMPRESSION DEBUG - Advanced RAG")
    logger.info(f"   ├─ Input docs: {len(state['reranked_docs'])}")
    logger.info(f"   ├─ Query: '{state['query']}'")
    logger.info(f"   └─ Original context length: {original_length} chars")
    
    result = advanced_rag_utils.compress_context(state["reranked_docs"], state["query"])
    compressed_length = len(result["compressed_context"])
    
    logger.info(f"📉 Compression Results:")
    logger.info(f"   ├─ Compressed length: {compressed_length} chars")
    logger.info(f"   ├─ Compression ratio: {result['compression_ratio']:.3f}")
    logger.info(f"   └─ Space saved: {original_length - compressed_length} chars ({(1-result['compression_ratio'])*100:.1f}%)")
    
    logger.info(f"📝 Compressed Context Preview:")
    preview = result["compressed_context"][:200] + "..." if len(result["compressed_context"]) > 200 else result["compressed_context"]
    logger.info(f"   └─ '{preview}'")
    
    return {
        "compressed_context": result["compressed_context"],
        "compression_ratio": result["compression_ratio"]
    }

def generate_node(state: AdvancedRagState, llm_manager: LLMManager):
    """Generates the final answer."""
    logger.info("-"*80)
    logger.info("--- STEP 5/5: Answer Generation ---")
    
    logger.info(f"📝 GENERATION DEBUG - Advanced RAG")
    logger.info(f"   ├─ Original query: '{state['query']}'")
    logger.info(f"   ├─ Optimized query: '{state['optimized_query']}'")
    logger.info(f"   └─ Context length: {len(state['compressed_context'])} chars")
    
    answer_stream = advanced_rag_utils.generate_answer_stream(
        llm_manager, state["query"], state["compressed_context"]
    )
    full_answer = "".join(list(answer_stream))
    
    logger.info(f"✏️ Generated Answer:")
    logger.info(f"   ├─ Length: {len(full_answer)} chars")
    logger.info(f"   └─ Preview: '{full_answer[:150]}{'...' if len(full_answer) > 150 else ''}'")
    
    logger.info("="*80)
    logger.info("✅ ADVANCED RAG SYSTEM COMPLETED")
    logger.info("="*80)
    logger.info(f"🎯 Final Summary:")
    logger.info(f"   ├─ Pipeline: Preprocess → Retrieve → Rerank → Compress → Generate")
    logger.info(f"   ├─ Final docs used: {len(state.get('reranked_docs', []))}")
    logger.info(f"   ├─ Compression ratio: {state.get('compression_ratio', 'N/A')}")
    logger.info(f"   ├─ Final answer length: {len(full_answer)} chars")
    logger.info(f"   └─ Features: Query optimization, TF-IDF reranking, context compression")
    logger.info("="*80)
    
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