"""LangGraph implementation for the Naive RAG system."""
import logging
from typing import TypedDict, List
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END
from langgraph.config import get_stream_writer

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

# ========== invoke ==========

def retrieve_node(state: NaiveRagState, vector_store_manager: VectorStoreManager):
    """Retrieves documents."""
    logger.info("="*80)
    logger.info("🚀 NAIVE RAG SYSTEM STARTED")
    logger.info("="*80)
    logger.info("--- STEP 1/2: Document Retrieval ---")

    logger.info(f"🔍 RETRIEVAL DEBUG - Naive RAG")
    logger.info(f"   ├─ Query: '{state['query']}'")
    logger.info(f"   └─ Retrieval K: {state['k']}")

    documents = naive_rag_utils.retrieve_naive(vector_store_manager, state["query"], state["k"])

    logger.info(f"📚 Retrieval Results:")
    logger.info(f"   └─ Retrieved docs: {len(documents)}")

    if documents:
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_chars = total_chars / len(documents)
        logger.info(f"📊 Document Statistics:")
        logger.info(f"   ├─ Total characters: {total_chars}")
        logger.info(f"   └─ Avg chars per doc: {avg_chars:.0f}")

        # Log document sources
        sources = [doc.metadata.get("source", "unknown")[:50] + "..." if len(doc.metadata.get("source", "unknown")) > 50
                  else doc.metadata.get("source", "unknown") for doc in documents]
        logger.info(f"📄 Document Sources:")
        for i, source in enumerate(sources, 1):
            logger.info(f"   {i}. {source}")

    return {"documents": documents}

def generate_node(state: NaiveRagState, llm_manager: LLMManager):
    """Generates the answer."""
    logger.info("-"*80)
    logger.info("--- STEP 2/2: Answer Generation ---")

    context_length = sum(len(doc.page_content) for doc in state["documents"])
    logger.info(f"📝 GENERATION DEBUG - Naive RAG")
    logger.info(f"   ├─ Query: '{state['query']}'")
    logger.info(f"   ├─ Input docs: {len(state['documents'])}")
    logger.info(f"   └─ Total context length: {context_length} chars")

    answer_stream = naive_rag_utils.generate_naive_answer_stream(
        llm_manager, state["query"], state["documents"]
    )
    full_answer = "".join(list(answer_stream))

    logger.info(f"✏️ Generated Answer:")
    logger.info(f"   ├─ Length: {len(full_answer)} chars")
    logger.info(f"   └─ Preview: '{full_answer[:150]}{'...' if len(full_answer) > 150 else ''}'")

    logger.info("="*80)
    logger.info("✅ NAIVE RAG SYSTEM COMPLETED")
    logger.info("="*80)
    logger.info(f"🎯 Final Summary:")
    logger.info(f"   ├─ Pipeline: Retrieve → Generate (Simple & Direct)")
    logger.info(f"   ├─ Total docs used: {len(state['documents'])}")
    logger.info(f"   ├─ Final answer length: {len(full_answer)} chars")
    logger.info(f"   └─ Features: No preprocessing, reranking, or compression")
    logger.info("="*80)

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

# ========== stream ==========

def generate_stream_node(state: NaiveRagState, llm_manager: LLMManager):
    """Generates the answer."""
    logger.info("-"*80)
    logger.info("--- STEP 2/2: Answer Generation ---")

    context_length = sum(len(doc.page_content) for doc in state["documents"])
    logger.info(f"📝 GENERATION DEBUG - Naive RAG")
    logger.info(f"   ├─ Query: '{state['query']}'")
    logger.info(f"   ├─ Input docs: {len(state['documents'])}")
    logger.info(f"   └─ Total context length: {context_length} chars")

    answer_stream = naive_rag_utils.generate_naive_answer_stream(
        llm_manager, state["query"], state["documents"]
    )

    writer = get_stream_writer()
    full_answer = ""
    for chunk in answer_stream:
        writer(chunk)
        full_answer += chunk

    logger.info(f"✏️ Generated Answer:")
    logger.info(f"   ├─ Length: {len(full_answer)} chars")
    logger.info(f"   └─ Preview: '{full_answer[:150]}{'...' if len(full_answer) > 150 else ''}'")

    logger.info("="*80)
    logger.info("✅ NAIVE RAG SYSTEM COMPLETED")
    logger.info("="*80)
    logger.info(f"🎯 Final Summary:")
    logger.info(f"   ├─ Pipeline: Retrieve → Generate (Simple & Direct)")
    logger.info(f"   ├─ Total docs used: {len(state['documents'])}")
    logger.info(f"   ├─ Final answer length: {len(full_answer)} chars")
    logger.info(f"   └─ Features: No preprocessing, reranking, or compression")
    logger.info("="*80)

    return {"answer": full_answer}

def create_naive_rag_stream_graph(llm_manager: LLMManager, vector_store_manager: VectorStoreManager):
    """Creates the Naive RAG graph."""
    workflow = StateGraph(NaiveRagState)

    workflow.add_node("retrieve", lambda state: retrieve_node(state, vector_store_manager))
    workflow.add_node("generate", lambda state: generate_stream_node(state, llm_manager))

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    graph = workflow.compile()
    logger.info("Naive RAG graph compiled successfully.")
    return graph