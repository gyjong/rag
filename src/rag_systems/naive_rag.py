"""Naive RAG utilities."""

import logging
from typing import List, Dict, Any, Iterator
from langchain_core.documents import Document

from ..utils.vector_store import VectorStoreManager
from ..utils.llm_manager import LLMManager

logger = logging.getLogger(__name__)

def retrieve_naive(vector_store_manager: VectorStoreManager, query: str, k: int = 5) -> List[Document]:
    """
    Retrieve relevant documents using simple similarity search.
    """
    vector_store = vector_store_manager.get_vector_store()
    if vector_store is None:
        logger.error("Vector store is not initialized.")
        return []
            
    try:
        logger.info(f"Retrieving documents for query: {query}")
        documents = vector_store.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(documents)} documents.")
        return documents
    except Exception as e:
        logger.error(f"Failed to retrieve documents: {e}", exc_info=True)
        return []

def generate_naive_answer_stream(llm_manager: LLMManager, query: str, context_docs: List[Document]) -> Iterator[str]:
    """
    Generate answer using retrieved context, streaming the response.
    """
    if not context_docs:
        yield "관련 문서를 찾을 수 없습니다."
        return
            
    context = "\\n\\n".join([doc.page_content for doc in context_docs])
    
    max_context_length = 4000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."
            
    try:
        logger.info("Generating answer from context...")
        response_stream = llm_manager.generate_response_stream(
            prompt=query,
            context=context
        )
        for chunk in response_stream:
            yield chunk
        logger.info("Answer generation complete.")
    except Exception as e:
        logger.error(f"Failed to generate answer: {e}", exc_info=True)
        yield "답변 생성 중 오류가 발생했습니다."

def get_naive_rag_system_info() -> Dict[str, Any]:
    """
    Get information about the Naive RAG system.
    """
    return {
        "name": "Naive RAG",
        "description": "기본적인 RAG 시스템: 단순 유사도 검색 + 생성",
        "components": [
            "Vector Store (Similarity Search)",
            "LLM (Direct Generation)"
        ],
        "features": [
            "단순 유사도 기반 검색",
            "직접적인 답변 생성",
            "빠른 처리 속도"
        ],
        "limitations": [
            "검색 품질 제한",
            "컨텍스트 재정렬 없음",
            "쿼리 최적화 없음"
        ]
    } 