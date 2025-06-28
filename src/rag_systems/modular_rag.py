"""Modular RAG implementation with flexible component-based architecture."""

from typing import List, Dict, Any, Optional, Callable
import time
import streamlit as st
from langchain_core.documents import Document
from enum import Enum

from ..utils.vector_store import VectorStoreManager
from ..utils.llm_manager import LLMManager


class ModuleType(Enum):
    """Types of modules in the Modular RAG system."""
    INDEXING = "indexing"
    PRE_RETRIEVAL = "pre_retrieval"
    RETRIEVAL = "retrieval"
    POST_RETRIEVAL = "post_retrieval"
    GENERATION = "generation"
    ORCHESTRATION = "orchestration"


class ModularRAG:
    """Modular RAG implementation with component-based architecture."""
    
    def __init__(self, vector_store_manager: VectorStoreManager, llm_manager: LLMManager):
        """Initialize Modular RAG system."""
        self.vector_store_manager = vector_store_manager
        self.llm_manager = llm_manager
        self.name = "Modular RAG"
        self.description = "모듈형 RAG 시스템: 유연한 컴포넌트 기반 아키텍처"
        
        # Initialize modules
        self.modules = {}
        self._setup_modules()
        
    def _setup_modules(self):
        """Setup default modules for the system."""
        self.modules = {
            ModuleType.PRE_RETRIEVAL: [
                self._query_expansion_module,
                self._query_classification_module
            ],
            ModuleType.RETRIEVAL: [
                self._semantic_retrieval_module,
                self._keyword_retrieval_module
            ],
            ModuleType.POST_RETRIEVAL: [
                self._relevance_filtering_module,
                self._diversity_module
            ],
            ModuleType.GENERATION: [
                self._answer_generation_module,
                self._confidence_estimation_module
            ],
            ModuleType.ORCHESTRATION: [
                self._routing_module,
                self._iteration_control_module
            ]
        }
    
    def _query_expansion_module(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Expand query with related terms."""
        expanded_terms = []
        
        # Simple keyword-based expansion
        keywords_map = {
            "AI": ["artificial intelligence", "machine learning", "딥러닝"],
            "인공지능": ["AI", "머신러닝", "neural network"],
            "트렌드": ["동향", "전망", "trend", "future"],
            "업무": ["work", "business", "직장", "productivity"]
        }
        
        for keyword, expansions in keywords_map.items():
            if keyword.lower() in query.lower():
                expanded_terms.extend(expansions[:2])
        
        expanded_query = f"{query} {' '.join(expanded_terms)}" if expanded_terms else query
        
        return {
            "original_query": query,
            "expanded_query": expanded_query,
            "expansion_terms": expanded_terms
        }
    
    def _query_classification_module(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Classify query type for routing."""
        query_lower = query.lower()
        
        # Simple rule-based classification
        if any(word in query_lower for word in ["what", "무엇", "어떤"]):
            query_type = "factual"
        elif any(word in query_lower for word in ["how", "어떻게", "방법"]):
            query_type = "procedural"
        elif any(word in query_lower for word in ["why", "왜", "이유"]):
            query_type = "causal"
        elif any(word in query_lower for word in ["when", "언제", "시점"]):
            query_type = "temporal"
        else:
            query_type = "general"
            
        return {
            "query_type": query_type,
            "confidence": 0.8  # Simple confidence score
        }
    
    def _semantic_retrieval_module(self, query: str, context: Dict[str, Any]) -> List[Document]:
        """Perform semantic retrieval using vector similarity."""
        vector_store = self.vector_store_manager.get_vector_store()
        if vector_store is None:
            return []
            
        try:
            k = context.get("retrieval_k", 8)
            docs = vector_store.similarity_search(query, k=k)
            return docs
        except Exception as e:
            st.error(f"Semantic retrieval failed: {str(e)}")
            return []
    
    def _keyword_retrieval_module(self, query: str, context: Dict[str, Any]) -> List[Document]:
        """Perform keyword-based retrieval (simplified)."""
        # In a real implementation, this would use a keyword search engine
        # For now, we'll use semantic search with query modification
        vector_store = self.vector_store_manager.get_vector_store()
        if vector_store is None:
            return []
            
        try:
            # Extract keywords and create keyword-focused query
            keywords = [word for word in query.split() if len(word) > 2]
            keyword_query = " ".join(keywords)
            
            k = context.get("retrieval_k", 5)
            docs = vector_store.similarity_search(keyword_query, k=k)
            return docs
        except Exception:
            return []
    
    def _relevance_filtering_module(self, docs: List[Document], context: Dict[str, Any]) -> List[Document]:
        """Filter documents based on relevance threshold."""
        if not docs:
            return docs
            
        # Simple length-based filtering (in practice, would use relevance scores)
        min_length = context.get("min_doc_length", 50)
        filtered_docs = [doc for doc in docs if len(doc.page_content) >= min_length]
        
        return filtered_docs[:context.get("max_docs", 5)]
    
    def _diversity_module(self, docs: List[Document], context: Dict[str, Any]) -> List[Document]:
        """Ensure diversity in retrieved documents."""
        if len(docs) <= 3:
            return docs
            
        # Simple diversity based on source
        diverse_docs = []
        seen_sources = set()
        
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            if source not in seen_sources or len(diverse_docs) < 2:
                diverse_docs.append(doc)
                seen_sources.add(source)
                
            if len(diverse_docs) >= context.get("max_diverse_docs", 4):
                break
                
        return diverse_docs
    
    def _answer_generation_module(self, query: str, docs: List[Document], context: Dict[str, Any]) -> str:
        """Generate answer using retrieved documents."""
        if not docs:
            return "관련 문서를 찾을 수 없습니다."
            
        doc_context = "\n\n".join([doc.page_content for doc in docs])
        
        # Limit context length
        max_length = context.get("max_context_length", 3500)
        if len(doc_context) > max_length:
            doc_context = doc_context[:max_length] + "..."
            
        query_type = context.get("query_type", "general")
        
        # Customize prompt based on query type
        if query_type == "factual":
            prompt_template = """다음 컨텍스트에서 사실적 정보를 바탕으로 질문에 답해주세요.
정확한 정보만 제공하고, 확실하지 않으면 모른다고 답해주세요.

컨텍스트: {context}
질문: {question}
답변:"""
        elif query_type == "procedural":
            prompt_template = """다음 컨텍스트를 바탕으로 단계별 방법을 설명해주세요.
가능하면 순서대로 정리해서 답변해주세요.

컨텍스트: {context}
질문: {question}
답변:"""
        else:
            prompt_template = """다음 컨텍스트를 바탕으로 질문에 답해주세요.
답변은 명확하고 도움이 되도록 작성해주세요.

컨텍스트: {context}
질문: {question}
답변:"""
        
        st.write("**모듈형 답변 생성 중...**")
        answer_placeholder = st.empty()
        
        start_time = time.time()
        full_response = ""
        
        # Stream the response
        for chunk in self.llm_manager.generate_response_stream(
            prompt=query,
            context=doc_context
        ):
            full_response += chunk
            answer_placeholder.markdown(full_response + "▌")
        
        generation_time = time.time() - start_time
        answer_placeholder.markdown(full_response)
        
        st.success(f"모듈형 답변 생성 완료 ({generation_time:.2f}초)")
        return full_response
    
    def _confidence_estimation_module(self, answer: str, context: Dict[str, Any]) -> float:
        """Estimate confidence in the generated answer."""
        # Simple heuristic-based confidence estimation
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on answer length and structure
        if len(answer) > 50:
            confidence += 0.2
        if len(answer) > 100:
            confidence += 0.1
            
        # Decrease confidence for uncertainty indicators
        uncertainty_words = ["모른다", "확실하지", "아마도", "maybe", "uncertain"]
        if any(word in answer.lower() for word in uncertainty_words):
            confidence -= 0.3
            
        # Increase confidence if answer has specific details
        if any(word in answer for word in ["2024", "2025", "구체적", "specific"]):
            confidence += 0.1
            
        return max(0.0, min(1.0, confidence))
    
    def _routing_module(self, query: str, context: Dict[str, Any]) -> str:
        """Route query to appropriate processing path."""
        query_type = context.get("query_type", "general")
        
        if query_type == "factual":
            return "precise_path"
        elif query_type == "procedural":
            return "step_by_step_path"
        elif query_type == "causal":
            return "reasoning_path"
        else:
            return "standard_path"
    
    def _iteration_control_module(self, context: Dict[str, Any]) -> bool:
        """Control whether to iterate or stop."""
        confidence = context.get("confidence", 0.0)
        iteration_count = context.get("iteration_count", 0)
        max_iterations = context.get("max_iterations", 2)
        
        # Continue if confidence is low and we haven't exceeded max iterations
        return confidence < 0.7 and iteration_count < max_iterations
    
    def query(self, question: str, max_iterations: int = 2) -> Dict[str, Any]:
        """Process query using modular approach."""
        start_time = time.time()
        context = {
            "iteration_count": 0,
            "max_iterations": max_iterations,
            "retrieval_k": 8,
            "max_docs": 5,
            "max_context_length": 3500
        }
        
        st.subheader("🧩 모듈형 RAG 처리 과정")
        
        # Step 1: Pre-retrieval processing
        st.write("**1단계: 사전 검색 처리**")
        
        # Query expansion
        expansion_result = self._query_expansion_module(question, context)
        context.update(expansion_result)
        if expansion_result["expansion_terms"]:
            st.info(f"쿼리 확장: {expansion_result['expanded_query']}")
        
        # Query classification
        classification_result = self._query_classification_module(question, context)
        context.update(classification_result)
        st.info(f"쿼리 유형: {classification_result['query_type']}")
        
        # Routing
        processing_path = self._routing_module(question, context)
        context["processing_path"] = processing_path
        st.info(f"처리 경로: {processing_path}")
        
        final_answer = ""
        all_retrieved_docs = []
        
        # Iterative processing
        while context["iteration_count"] < max_iterations:
            iteration = context["iteration_count"] + 1
            st.write(f"**반복 {iteration}:**")
            
            # Step 2: Retrieval
            with st.spinner("문서 검색 중..."):
                query_to_use = context.get("expanded_query", question)
                
                # Combine semantic and keyword retrieval
                semantic_docs = self._semantic_retrieval_module(query_to_use, context)
                keyword_docs = self._keyword_retrieval_module(query_to_use, context)
                
                # Merge and deduplicate
                all_docs = semantic_docs + keyword_docs
                seen_content = set()
                unique_docs = []
                for doc in all_docs:
                    content_hash = hash(doc.page_content[:100])
                    if content_hash not in seen_content:
                        unique_docs.append(doc)
                        seen_content.add(content_hash)
                
                st.success(f"검색 완료: {len(unique_docs)}개 문서")
            
            # Step 3: Post-retrieval processing
            filtered_docs = self._relevance_filtering_module(unique_docs, context)
            diverse_docs = self._diversity_module(filtered_docs, context)
            all_retrieved_docs.extend(diverse_docs)
            
            st.info(f"후처리 완료: {len(diverse_docs)}개 문서 선택")
            
            # Step 4: Generation
            with st.spinner("답변 생성 중..."):
                answer = self._answer_generation_module(question, diverse_docs, context)
                
            # Step 5: Confidence estimation
            confidence = self._confidence_estimation_module(answer, context)
            context["confidence"] = confidence
            context["iteration_count"] = iteration
            
            st.info(f"신뢰도: {confidence:.2f}")
            
            # Check if we should continue iterating
            if not self._iteration_control_module(context):
                final_answer = answer
                break
            else:
                st.warning(f"신뢰도가 낮아 다음 반복을 시도합니다 (신뢰도: {confidence:.2f})")
                # Adjust parameters for next iteration
                context["retrieval_k"] = min(context["retrieval_k"] + 2, 15)
                final_answer = answer
        
        # Display final results
        with st.expander(f"최종 검색된 문서 ({len(all_retrieved_docs)}개)"):
            for i, doc in enumerate(all_retrieved_docs[-5:]):  # Show last 5
                st.write(f"**문서 {i+1}:**")
                st.write(f"출처: {doc.metadata.get('source', 'Unknown')}")
                st.write(f"내용: {doc.page_content[:200]}...")
                st.divider()
        
        total_time = time.time() - start_time
        
        return {
            "question": question,
            "answer": final_answer,
            "retrieved_docs": all_retrieved_docs,
            "total_time": total_time,
            "rag_type": self.name,
            "metadata": {
                "iterations": context["iteration_count"],
                "final_confidence": context.get("confidence", 0.0),
                "query_type": context.get("query_type", "general"),
                "processing_path": context.get("processing_path", "standard"),
                "total_retrieved": len(all_retrieved_docs),
                "expansion_terms": context.get("expansion_terms", [])
            }
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the Modular RAG system."""
        return {
            "name": self.name,
            "description": self.description,
            "components": [
                "Pre-retrieval Modules (Query Expansion, Classification)",
                "Retrieval Modules (Semantic + Keyword)",
                "Post-retrieval Modules (Filtering, Diversity)",
                "Generation Modules (Answer Generation, Confidence)",
                "Orchestration Modules (Routing, Iteration Control)"
            ],
            "features": [
                "모듈형 아키텍처",
                "쿼리 유형별 라우팅",
                "반복적 개선",
                "신뢰도 기반 제어",
                "다중 검색 전략"
            ],
            "advantages": [
                "높은 유연성과 확장성",
                "상황별 최적화",
                "점진적 품질 개선",
                "투명한 처리 과정"
            ]
        } 