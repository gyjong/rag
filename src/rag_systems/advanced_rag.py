"""Advanced RAG utilities with enhanced retrieval and processing."""

import logging
from typing import List, Dict, Any, Iterator, Tuple
import re
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from ..utils.vector_store import VectorStoreManager
from ..utils.llm_manager import LLMManager

logger = logging.getLogger(__name__)

def preprocess_query(query: str) -> Dict[str, Any]:
    """Preprocess and optimize the query with comprehensive expansion and enhancement.
    
    Args:
        query: Original user query
        
    Returns:
        A dictionary containing the original query, expansion details, and the enhanced query.
    """
    cleaned_query = re.sub(r'\s+', ' ', query.strip())
    
    # Comprehensive query expansion with multiple strategies
    expanded_terms = []
    
    # Domain-specific keyword expansions
    domain_keyword_map = {
        "AI": (["ai", "인공지능", "artificial intelligence", "머신러닝", "machine learning"],
               ["딥러닝", "deep learning", "neural network", "신경망", "자동화", "automation", "알고리즘", "algorithm", "데이터 분석", "data analysis", "예측 모델", "predictive modeling"]),
        "Business": (["업무", "work", "직장", "business", "비즈니스", "회사"],
                     ["생산성", "productivity", "효율성", "efficiency", "업무 프로세스", "work process", "자동화", "automation", "디지털 전환", "digital transformation", "혁신", "innovation"]),
        "Trend": (["트렌드", "trend", "동향", "전망", "미래", "future"],
                  ["시장 동향", "market trend", "기술 동향", "technology trend", "발전 방향", "development direction", "변화", "change", "혁신", "innovation", "진화", "evolution"]),
        "Industry": (["산업", "industry", "시장", "market", "기업", "company"],
                     ["시장 분석", "market analysis", "경쟁", "competition", "성장", "growth", "투자", "investment", "전략", "strategy"]),
        "Analysis": (["분석", "analysis", "연구", "research", "조사", "survey"],
                     ["데이터 분석", "data analysis", "통계", "statistics", "조사 결과", "survey results", "연구 보고서", "research report"]),
        "Strategy": (["도입", "implementation", "전략", "strategy", "방안", "plan"],
                     ["실행 계획", "execution plan", "로드맵", "roadmap", "단계별 접근", "step-by-step approach", "성공 사례", "success case"]),
        "Performance": (["성능", "performance", "품질", "quality", "효율성", "efficiency"],
                        ["최적화", "optimization", "개선", "improvement", "측정", "measurement", "평가", "evaluation", "벤치마크", "benchmark"]),
        "Impact": (["영향", "impact", "효과", "effect", "변화", "change"],
                   ["결과", "result", "성과", "outcome", "개선 효과", "improvement effect", "변화 분석", "change analysis", "영향 평가", "impact assessment"]),
        "Tech App": (["자동화", "automation", "디지털화", "digitalization", "혁신", "innovation"],
                     ["스마트 팩토리", "smart factory", "IoT", "인터넷 of things", "클라우드", "cloud", "빅데이터", "big data", "블록체인", "blockchain"]),
        "Temporal": (["현재", "current", "미래", "future", "과거", "past", "비교", "compare"],
                     ["시계열 분석", "time series analysis", "트렌드 비교", "trend comparison", "변화 추이", "change trend", "예측", "prediction", "전망", "outlook"])
    }

    query_lower = cleaned_query.lower()
    for _, (triggers, terms) in domain_keyword_map.items():
        if any(keyword.lower() in query_lower for keyword in triggers):
            expanded_terms.extend(terms)

    unique_terms = list(dict.fromkeys(expanded_terms))

    # Dynamic expansion based on query complexity
    matched_domains = sum(1 for _, (keywords, _) in domain_keyword_map.items() if any(k in query_lower for k in keywords))
    
    question_marks = query.count('?') + query.count('？')
    word_count = len(cleaned_query.split())
    
    base_expansion = min(matched_domains * 2, 6)
    if question_marks > 1: base_expansion += 2
    if word_count > 15: base_expansion += 1
        
    max_terms = max(3, min(base_expansion, 8))
    selected_terms = unique_terms[:max_terms]
    
    enhanced_query = f"{cleaned_query} {' '.join(selected_terms)}" if selected_terms else cleaned_query

    return {
        "original_query": query,
        "cleaned_query": cleaned_query,
        "enhanced_query": enhanced_query,
        "matched_domains": matched_domains,
        "selected_terms": selected_terms
    }

def retrieve_with_scores(vector_store_manager: VectorStoreManager, query: str, k: int = 10) -> List[Tuple[Document, float]]:
    """Retrieve documents with similarity scores.
    
    Args:
        vector_store_manager: Vector store manager instance
        query: Search query
        k: Number of documents to retrieve
        
    Returns:
        List of (document, score) tuples
    """
    vector_store = vector_store_manager.get_vector_store()
    if not vector_store:
        logger.error("Vector store is not initialized.")
        return []
            
    try:
        return vector_store.similarity_search_with_score(query, k=k)
    except Exception as e:
        logger.error(f"Failed to retrieve documents with scores: {e}", exc_info=True)
        return []

def rerank_documents(query: str, docs_with_scores: List[Tuple[Document, float]], top_k: int = 5) -> List[Document]:
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
    texts = [doc.page_content for doc in documents]
    
    if not texts:
        return []

    texts_with_query = texts + [query]
    
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(texts_with_query)
        
        query_vector = tfidf_matrix[-1]
        doc_vectors = tfidf_matrix[:-1]
        
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        
        combined_scores = []
        for i, (doc, original_score) in enumerate(docs_with_scores):
            tfidf_score = similarities[i]
            combined_score = 0.7 * tfidf_score + 0.3 * (1 - original_score)
            combined_scores.append((doc, combined_score))
        
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        reranked_docs = [doc for doc, _ in combined_scores[:top_k]]
        return reranked_docs
            
    except Exception as e:
        logger.warning(f"Reranking failed, returning original order: {e}", exc_info=True)
        return [doc for doc, _ in docs_with_scores[:top_k]]

def compress_context(docs: List[Document], max_length: int = 3000) -> Dict[str, Any]:
    """Compress context by selecting most relevant sentences.
    
    Args:
        docs: List of documents
        max_length: Maximum context length
        
    Returns:
        A dictionary with the compressed context and compression ratio.
    """
    if not docs:
        return {"compressed_context": "", "compression_ratio": 0}
            
    full_context = "\\n\\n".join([doc.page_content for doc in docs])
    original_length = len(full_context)
    
    if original_length <= max_length:
        return {"compressed_context": full_context, "compression_ratio": 1.0}
            
    sentences = re.split(r'(?<=[.!?])\s+', full_context)
    
    scored_sentences = []
    for sentence in sentences:
        if len(sentence.strip()) < 20: continue
            
        score = len(sentence) * 0.1
        if any(keyword in sentence.lower() for keyword in ['ai', '인공지능', '트렌드', '미래', '업무']):
            score *= 1.5
        scored_sentences.append((sentence.strip(), score))
    
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    compressed_context = ""
    for sentence, _ in scored_sentences:
        if len(compressed_context) + len(sentence) <= max_length:
            compressed_context += sentence + " "
        else:
            break
    
    final_length = len(compressed_context.strip())
    compression_ratio = final_length / original_length if original_length > 0 else 0
            
    return {"compressed_context": compressed_context.strip(), "compression_ratio": compression_ratio}

def generate_answer_stream(llm_manager: LLMManager, query: str, context: str) -> Iterator[str]:
    """Generate answer with explicit reasoning steps.
    
    Args:
        llm_manager: LLM manager instance
        query: User query
        context: Retrieved and processed context
        
    Yields:
        Generated answer chunks
    """
    try:
        logger.info("Generating answer from compressed context...")
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

def get_advanced_rag_system_info() -> Dict[str, Any]:
    """Get information about the Advanced RAG system.
    
    Returns:
        Dictionary with system information
    """
    return {
        "name": "Advanced RAG",
        "description": "향상된 RAG 시스템: 쿼리 최적화 + 재순위화 + 컨텍스트 압축",
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