"""Naive RAG implementation."""

from typing import List, Dict, Any, Optional
import time
import streamlit as st
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM

from ..utils.vector_store import VectorStoreManager
from ..utils.llm_manager import LLMManager


class NaiveRAG:
    """Naive RAG implementation with simple retrieval and generation."""
    
    def __init__(self, vector_store_manager: VectorStoreManager, llm_manager: LLMManager):
        """Initialize Naive RAG system.
        
        Args:
            vector_store_manager: Vector store manager instance
            llm_manager: LLM manager instance
        """
        self.vector_store_manager = vector_store_manager
        self.llm_manager = llm_manager
        self.name = "Naive RAG"
        self.description = "기본적인 RAG 시스템: 단순 유사도 검색 + 생성"
        
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents using simple similarity search.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        vector_store = self.vector_store_manager.get_vector_store()
        if vector_store is None:
            st.error("벡터 스토어가 초기화되지 않았습니다.")
            return []
            
        try:
            with st.spinner("문서 검색 중..."):
                start_time = time.time()
                documents = vector_store.similarity_search(query, k=k)
                retrieval_time = time.time() - start_time
                
            st.success(f"검색 완료: {len(documents)}개 문서 ({retrieval_time:.2f}초)")
            return documents
            
        except Exception as e:
            st.error(f"문서 검색 실패: {str(e)}")
            return []
    
    def generate(self, query: str, context_docs: List[Document]) -> str:
        """Generate answer using retrieved context.
        
        Args:
            query: User query
            context_docs: Retrieved context documents
            
        Returns:
            Generated answer
        """
        if not context_docs:
            return "관련 문서를 찾을 수 없습니다."
            
        # Combine context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Limit context length to avoid token limits
        max_context_length = 4000
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
            
        st.write("**답변 생성 중...**")
        answer_placeholder = st.empty()
        
        start_time = time.time()
        full_response = ""
        
        # Stream the response
        for chunk in self.llm_manager.generate_response_stream(
            prompt=query,
            context=context
        ):
            full_response += chunk
            answer_placeholder.markdown(full_response + "▌")
        
        generation_time = time.time() - start_time
        answer_placeholder.markdown(full_response)
        
        st.success(f"답변 생성 완료 ({generation_time:.2f}초)")
        return full_response
    
    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Process a query end-to-end.
        
        Args:
            question: User question
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with results and metadata
        """
        start_time = time.time()
        
        # Step 1: Retrieve relevant documents
        st.subheader("🔍 1단계: 문서 검색")
        retrieved_docs = self.retrieve(question, k=k)
        
        if not retrieved_docs:
            return {
                "question": question,
                "answer": "관련 문서를 찾을 수 없습니다.",
                "retrieved_docs": [],
                "total_time": time.time() - start_time,
                "rag_type": self.name
            }
        
        # Display retrieved documents
        with st.expander(f"검색된 문서 ({len(retrieved_docs)}개)"):
            for i, doc in enumerate(retrieved_docs):
                st.write(f"**문서 {i+1}:**")
                st.write(f"출처: {doc.metadata.get('source', 'Unknown')}")
                st.write(f"내용: {doc.page_content[:200]}...")
                st.divider()
        
        # Step 2: Generate answer
        st.subheader("🤖 2단계: 답변 생성")
        answer = self.generate(question, retrieved_docs)
        
        total_time = time.time() - start_time
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "total_time": total_time,
            "rag_type": self.name,
            "metadata": {
                "num_retrieved": len(retrieved_docs),
                "retrieval_method": "similarity_search",
                "generation_method": "simple"
            }
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the Naive RAG system.
        
        Returns:
            Dictionary with system information
        """
        return {
            "name": self.name,
            "description": self.description,
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