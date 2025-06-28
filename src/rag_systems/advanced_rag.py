"""Advanced RAG implementation with enhanced retrieval and processing."""

from typing import List, Dict, Any, Optional, Tuple
import time
import re
import streamlit as st
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from ..utils.vector_store import VectorStoreManager
from ..utils.llm_manager import LLMManager


class AdvancedRAG:
    """Advanced RAG implementation with pre-retrieval optimization and post-retrieval processing."""
    
    def __init__(self, vector_store_manager: VectorStoreManager, llm_manager: LLMManager):
        """Initialize Advanced RAG system.
        
        Args:
            vector_store_manager: Vector store manager instance
            llm_manager: LLM manager instance
        """
        self.vector_store_manager = vector_store_manager
        self.llm_manager = llm_manager
        self.name = "Advanced RAG"
        self.description = "향상된 RAG 시스템: 쿼리 최적화 + 재순위화 + 컨텍스트 압축"
        
    def preprocess_query(self, query: str) -> str:
        """Preprocess and optimize the query.
        
        Args:
            query: Original user query
            
        Returns:
            Optimized query
        """
        # Basic query cleaning
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Add query expansion keywords (simple approach)
        expanded_terms = []
        
        # Add synonyms and related terms for common concepts
        if "AI" in query or "인공지능" in query:
            expanded_terms.extend(["machine learning", "딥러닝", "neural network"])
        if "트렌드" in query or "trend" in query:
            expanded_terms.extend(["동향", "전망", "미래"])
        if "업무" in query or "work" in query:
            expanded_terms.extend(["직장", "비즈니스", "productivity"])
            
        if expanded_terms:
            expanded_query = f"{query} {' '.join(expanded_terms[:2])}"
            return expanded_query
            
        return query
    
    def retrieve_with_scores(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Retrieve documents with similarity scores.
        
        Args:
            query: Search query
            k: Number of documents to retrieve (before reranking)
            
        Returns:
            List of (document, score) tuples
        """
        vector_store = self.vector_store_manager.get_vector_store()
        if vector_store is None:
            st.error("벡터 스토어가 초기화되지 않았습니다.")
            return []
            
        try:
            docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
            return docs_with_scores
            
        except Exception as e:
            st.error(f"문서 검색 실패: {str(e)}")
            return []
    
    def rerank_documents(self, query: str, docs_with_scores: List[Tuple[Document, float]], top_k: int = 5) -> List[Document]:
        """Rerank documents using TF-IDF similarity.
        
        Args:
            query: Original query
            docs_with_scores: Documents with initial scores
            top_k: Number of top documents to return after reranking
            
        Returns:
            Reranked documents
        """
        if not docs_with_scores:
            return []
            
        documents = [doc for doc, _ in docs_with_scores]
        
        # Prepare texts for TF-IDF
        texts = [doc.page_content for doc in documents]
        texts.append(query)  # Add query as the last document
        
        try:
            # Calculate TF-IDF similarity
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Calculate similarity between query and documents
            query_vector = tfidf_matrix[-1]  # Last vector is the query
            doc_vectors = tfidf_matrix[:-1]  # All vectors except the last one
            
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            
            # Combine original scores with TF-IDF scores
            combined_scores = []
            for i, (doc, original_score) in enumerate(docs_with_scores):
                tfidf_score = similarities[i]
                # Weighted combination (70% TF-IDF, 30% original embedding score)
                combined_score = 0.7 * tfidf_score + 0.3 * (1 - original_score)  # Lower distance = higher score
                combined_scores.append((doc, combined_score))
            
            # Sort by combined score (descending)
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k documents
            reranked_docs = [doc for doc, _ in combined_scores[:top_k]]
            return reranked_docs
            
        except Exception as e:
            st.warning(f"재순위화 실패, 원본 순서 사용: {str(e)}")
            return [doc for doc, _ in docs_with_scores[:top_k]]
    
    def compress_context(self, docs: List[Document], max_length: int = 3000) -> str:
        """Compress context by selecting most relevant sentences.
        
        Args:
            docs: List of documents
            max_length: Maximum context length
            
        Returns:
            Compressed context
        """
        if not docs:
            return ""
            
        # Combine all text
        full_context = "\n\n".join([doc.page_content for doc in docs])
        
        if len(full_context) <= max_length:
            return full_context
            
        # Split into sentences
        sentences = re.split(r'[.!?]\s+', full_context)
        
        # Simple sentence scoring based on length and content
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) < 20:  # Skip very short sentences
                continue
                
            # Score based on keywords and length
            score = len(sentence) * 0.1
            if any(keyword in sentence.lower() for keyword in ['ai', '인공지능', '트렌드', '미래', '업무']):
                score *= 1.5
                
            scored_sentences.append((sentence.strip(), score))
        
        # Sort by score and select top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        compressed_context = ""
        for sentence, _ in scored_sentences:
            if len(compressed_context) + len(sentence) <= max_length:
                compressed_context += sentence + ". "
            else:
                break
                
        return compressed_context.strip()
    
    def generate_with_reasoning(self, query: str, context: str) -> str:
        """Generate answer with explicit reasoning steps.
        
        Args:
            query: User query
            context: Retrieved and processed context
            
        Returns:
            Generated answer with reasoning
        """
        st.write("**답변 생성 중 (추론 기반)...**")
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
        
        st.success(f"추론 기반 답변 생성 완료 ({generation_time:.2f}초)")
        return full_response
    
    def query(self, question: str, k: int = 10, rerank_top_k: int = 5) -> Dict[str, Any]:
        """Process a query end-to-end with advanced techniques.
        
        Args:
            question: User question
            k: Number of documents to retrieve initially
            rerank_top_k: Number of documents after reranking
            
        Returns:
            Dictionary with results and metadata
        """
        start_time = time.time()
        
        # Step 1: Query preprocessing
        st.subheader("🔧 1단계: 쿼리 전처리")
        optimized_query = self.preprocess_query(question)
        if optimized_query != question:
            st.info(f"최적화된 쿼리: {optimized_query}")
        else:
            st.info("쿼리 최적화: 변경사항 없음")
        
        # Step 2: Retrieve with scores
        st.subheader("🔍 2단계: 문서 검색")
        docs_with_scores = self.retrieve_with_scores(optimized_query, k=k)
        
        if not docs_with_scores:
            return {
                "question": question,
                "answer": "관련 문서를 찾을 수 없습니다.",
                "retrieved_docs": [],
                "total_time": time.time() - start_time,
                "rag_type": self.name
            }
        
        st.success(f"초기 검색: {len(docs_with_scores)}개 문서")
        
        # Step 3: Rerank documents
        st.subheader("📊 3단계: 문서 재순위화")
        reranked_docs = self.rerank_documents(optimized_query, docs_with_scores, rerank_top_k)
        st.success(f"재순위화 완료: 상위 {len(reranked_docs)}개 문서 선택")
        
        # Display reranked documents
        with st.expander(f"재순위화된 문서 ({len(reranked_docs)}개)"):
            for i, doc in enumerate(reranked_docs):
                st.write(f"**문서 {i+1}:**")
                st.write(f"출처: {doc.metadata.get('source', 'Unknown')}")
                st.write(f"내용: {doc.page_content[:200]}...")
                st.divider()
        
        # Step 4: Context compression
        st.subheader("🗜️ 4단계: 컨텍스트 압축")
        compressed_context = self.compress_context(reranked_docs)
        compression_ratio = len(compressed_context) / sum(len(doc.page_content) for doc in reranked_docs) if reranked_docs else 0
        st.info(f"압축률: {compression_ratio:.2%}")
        
        # Step 5: Generate answer with reasoning
        st.subheader("🤖 5단계: 답변 생성")
        answer = self.generate_with_reasoning(question, compressed_context)
        
        total_time = time.time() - start_time
        
        return {
            "question": question,
            "optimized_query": optimized_query,
            "answer": answer,
            "retrieved_docs": reranked_docs,
            "total_time": total_time,
            "rag_type": self.name,
            "metadata": {
                "initial_retrieved": len(docs_with_scores),
                "final_retrieved": len(reranked_docs),
                "compression_ratio": compression_ratio,
                "retrieval_method": "similarity_search + reranking",
                "generation_method": "reasoning-based"
            }
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the Advanced RAG system.
        
        Returns:
            Dictionary with system information
        """
        return {
            "name": self.name,
            "description": self.description,
            "components": [
                "Query Preprocessor",
                "Vector Store (Enhanced Search)",
                "Document Reranker (TF-IDF)",
                "Context Compressor",
                "LLM (Reasoning-based Generation)"
            ],
            "features": [
                "쿼리 확장 및 최적화",
                "TF-IDF 기반 문서 재순위화",
                "컨텍스트 압축",
                "추론 기반 답변 생성"
            ],
            "advantages": [
                "향상된 검색 정확도",
                "효율적인 컨텍스트 활용",
                "더 정확한 답변 생성"
            ]
        } 