"""Advanced RAG utilities with enhanced retrieval and processing."""

import logging
from typing import List, Dict, Any, Iterator, Tuple
import re
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from ..utils.vector_store import VectorStoreManager
from ..utils.llm_manager import LLMManager
from ..config.settings import QUERY_EXPANSION_COUNT, CONTEXT_COMPRESSION_MAX_LENGTH, ADVANCED_RAG_DOMAIN_KEYWORD_MAP, KOREAN_STOP_WORDS

logger = logging.getLogger(__name__)

def _get_combined_stop_words() -> List[str]:
    """Get combined English and Korean stop words.
    
    Returns:
        List of combined stop words
    """
    # Combine English and Korean stop words
    combined_stop_words = list(ENGLISH_STOP_WORDS) + KOREAN_STOP_WORDS
    return combined_stop_words

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
    
    # Use centralized domain keyword map from settings
    query_lower = cleaned_query.lower()
    for _, (triggers, terms) in ADVANCED_RAG_DOMAIN_KEYWORD_MAP.items():
        if any(keyword.lower() in query_lower for keyword in triggers):
            expanded_terms.extend(terms)

    unique_terms = list(dict.fromkeys(expanded_terms))

    # Dynamic expansion based on query complexity
    matched_domains = sum(1 for _, (keywords, _) in ADVANCED_RAG_DOMAIN_KEYWORD_MAP.items() if any(k in query_lower for k in keywords))
    
    question_marks = query.count('?') + query.count('？')
    word_count = len(cleaned_query.split())
    
    base_expansion = min(matched_domains * 2, 6)
    if question_marks > 1: base_expansion += 2
    if word_count > 15: base_expansion += 1
        
    max_terms = max(3, min(base_expansion, QUERY_EXPANSION_COUNT))
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
        # Use combined English and Korean stop words for better multilingual support
        combined_stop_words = _get_combined_stop_words()
        vectorizer = TfidfVectorizer(stop_words=combined_stop_words, max_features=1000)
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

def _extract_relevant_keywords(query: str) -> Dict[str, List[str]]:
    """Extract relevant keywords from query using domain keyword mapping.
    
    Args:
        query: User query
        
    Returns:
        Dictionary with matched domains and their keywords
    """
    # Use centralized domain keyword map from settings
    query_lower = query.lower()
    matched_keywords = {}
    
    for domain, (triggers, expansion_terms) in ADVANCED_RAG_DOMAIN_KEYWORD_MAP.items():
        if any(keyword.lower() in query_lower for keyword in triggers):
            # Increase expansion for compression context (more keywords for better matching)
            max_terms = min(len(expansion_terms), QUERY_EXPANSION_COUNT * 2)  # Double the expansion
            matched_keywords[domain] = expansion_terms[:max_terms]
    
    return matched_keywords

def _calculate_keyword_boost(sentence_lower: str, relevant_keywords: Dict[str, List[str]]) -> float:
    """Calculate keyword boost score for a sentence.
    
    Args:
        sentence_lower: Sentence in lowercase
        relevant_keywords: Dictionary of domain keywords
        
    Returns:
        Boost multiplier for the sentence score
    """
    if not relevant_keywords:
        return 1.0  # No boost if no relevant keywords
    
    boost = 1.0
    total_matches = 0
    total_keywords = sum(len(keywords) for keywords in relevant_keywords.values())
    
    # Count keyword matches across all domains
    for domain, keywords in relevant_keywords.items():
        domain_matches = sum(1 for keyword in keywords if keyword.lower() in sentence_lower)
        total_matches += domain_matches
        
        # Domain-specific boost weights
        domain_weights = {
            "AI": 1.8,
            "Legal": 1.7,
            "Business": 1.5,
            "Industry": 1.5,
            "Strategy": 1.4,
            "Trend": 1.4,
            "Performance": 1.3,
            "Analysis": 1.3,
            "Impact": 1.2,
            "Tech App": 1.2,
            "Temporal": 1.1
        }
        
        if domain_matches > 0:
            domain_boost = domain_weights.get(domain, 1.2)
            boost *= domain_boost ** min(domain_matches / len(keywords), 0.5)  # Cap the exponential growth
    
    # Additional boost based on match density
    if total_keywords > 0:
        match_ratio = total_matches / total_keywords
        density_boost = 1.0 + (match_ratio * 0.5)  # Up to 1.5x boost for high density
        boost *= density_boost
    
    # Cap maximum boost to prevent extreme values
    return min(boost, 3.0)

def compress_context(docs: List[Document], query: str = "", max_length: int = CONTEXT_COMPRESSION_MAX_LENGTH) -> Dict[str, Any]:
    """Compress context by selecting most relevant sentences using dynamic keyword extraction.
    
    Args:
        docs: List of documents
        query: Original query to extract relevant keywords
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
    
    # Extract relevant keywords from query using domain_keyword_map
    relevant_keywords = _extract_relevant_keywords(query)
    
    # Log detected keywords for debugging
    if relevant_keywords and query:
        logger.info(f"🔍 Compression Keywords Detected:")
        for domain, keywords in relevant_keywords.items():
            logger.info(f"   ├─ {domain}: {keywords[:5]}{'...' if len(keywords) > 5 else ''}")
        logger.info(f"   └─ Total keywords: {sum(len(kw) for kw in relevant_keywords.values())}")
    elif query:
        logger.info(f"🔍 No domain keywords detected in query: '{query}'")
    
    sentences = re.split(r'(?<=[.!?])\s+', full_context)
    
    scored_sentences = []
    for sentence in sentences:
        if len(sentence.strip()) < 20: continue
            
        score = len(sentence) * 0.1  # Base score from length
        
        # Dynamic keyword scoring based on query context
        keyword_boost = _calculate_keyword_boost(sentence.lower(), relevant_keywords)
        score *= keyword_boost
        
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