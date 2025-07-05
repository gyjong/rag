"""Modular RAG utilities with a flexible component-based architecture."""

import logging
from typing import List, Dict, Any, Iterator, Tuple
import re
import numpy as np
from langchain_core.documents import Document
from enum import Enum

from ..utils.vector_store import VectorStoreManager
from ..utils.llm_manager import LLMManager
from ..config.settings import CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

# --- BM25 Implementation (Utility Class) ---
class BM25:
    """BM25 ranking algorithm implementation."""
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus or []
        self.corpus_size = len(self.corpus)
        
        if self.corpus_size == 0:
            self._initialize_empty()
            return

        try:
            self.doc_tokens = [self._tokenize(doc) if doc else [] for doc in self.corpus]
            self.doc_len = [len(tokens) for tokens in self.doc_tokens]
            self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0
            
            self.vocab = list(set(token for tokens in self.doc_tokens for token in tokens))
            
            self.df = {term: sum(1 for tokens in self.doc_tokens if term in tokens) for term in self.vocab}
            self.idf = {term: np.log((self.corpus_size - self.df.get(term, 0) + 0.5) / (self.df.get(term, 0) + 0.5) + 1) for term in self.vocab}

        except Exception as e:
            logger.error(f"BM25 initialization failed: {e}", exc_info=True)
            self._initialize_empty()
            raise ValueError(f"BM25 초기화 실패: {str(e)}")

    def _initialize_empty(self):
        self.doc_tokens = []
        self.doc_len = []
        self.avgdl = 0
        self.vocab = []
        self.df = {}
        self.idf = {}

    def _tokenize(self, text: str) -> List[str]:
        if not text or not isinstance(text, str): return []
        text = text.lower()
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        return [token for token in text.split() if len(token) > 1]

    def get_scores(self, query: str) -> List[float]:
        if self.corpus_size == 0: return []
        query_tokens = self._tokenize(query)
        if not query_tokens: return [0.0] * self.corpus_size

        scores = np.zeros(self.corpus_size)
        for term in query_tokens:
            if term not in self.vocab: continue
            
            q_freq = np.array([doc.count(term) for doc in self.doc_tokens])
            
            numerator = q_freq * (self.k1 + 1)
            denominator = q_freq + self.k1 * (1 - self.b + self.b * np.array(self.doc_len) / (self.avgdl + 1e-8))
            
            scores += self.idf.get(term, 0) * (numerator / (denominator + 1e-8))
            
        return scores.tolist()

    def get_top_k_indices(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        scores = self.get_scores(query)
        if not scores: return []
        
        top_indices = np.argsort(scores)[::-1][:k]
        return [(i, scores[i]) for i in top_indices if scores[i] > 0]

# --- RAG Module Definitions ---
class ModuleType(Enum):
    INDEXING = "indexing"
    PRE_RETRIEVAL = "pre_retrieval"
    RETRIEVAL = "retrieval"
    POST_RETRIEVAL = "post_retrieval"
    GENERATION = "generation"
    ORCHESTRATION = "orchestration"

# --- Pre-retrieval Modules ---
def expand_query(query: str) -> Dict[str, Any]:
    """Expand query with related terms."""
    keywords_map = {
        # Technology & AI
        "AI": ["artificial intelligence", "machine learning", "deep learning", "neural networks", "automation"],
        "인공지능": ["AI", "머신러닝", "딥러닝", "신경망", "자동화"],
        "머신러닝": ["machine learning", "ML", "AI", "인공지능", "데이터 분석"],
        "딥러닝": ["deep learning", "neural networks", "AI", "머신러닝"],
        
        # Business & Trends
        "트렌드": ["동향", "전망", "추세", "방향성", "미래"],
        "동향": ["trend", "트렌드", "전망", "추세", "방향성"],
        "전망": ["outlook", "forecast", "예측", "미래", "트렌드"],
        "시장": ["market", "business", "산업", "경제", "상업"],
        
        # Time & Future
        "미래": ["future", "전망", "향후", "앞으로", "다가올"],
        "2025년": ["2025", "올해", "현재", "최신", "최근"],
        "올해": ["2025", "현재", "최신", "최근", "이번년도"],
        
        # Work & Productivity
        "업무": ["work", "business", "직무", "일", "업무환경"],
        "생산성": ["productivity", "efficiency", "효율성", "성과", "업무효율"],
        "직장": ["workplace", "office", "회사", "직장환경", "업무환경"],
        
        # Industry & Sectors
        "기업": ["company", "corporation", "business", "회사", "기업환경"],
        "산업": ["industry", "sector", "분야", "업계", "산업계"],
        "스타트업": ["startup", "신생기업", "벤처", "창업", "신규기업"],
        
        # Digital & Technology
        "디지털": ["digital", "온라인", "전자", "정보기술", "IT"],
        "온라인": ["online", "인터넷", "웹", "디지털", "가상"],
        "모바일": ["mobile", "스마트폰", "앱", "휴대용", "이동"],
        
        # Data & Analytics
        "데이터": ["data", "정보", "분석", "통계", "인사이트"],
        "분석": ["analysis", "analytics", "데이터분석", "통계", "연구"],
        "통계": ["statistics", "데이터", "수치", "분석", "조사"],
        
        # Innovation & Development
        "혁신": ["innovation", "창의성", "새로운", "발전", "개선"],
        "개발": ["development", "연구", "개발", "기술개발", "제품개발"],
        "연구": ["research", "조사", "분석", "개발", "탐구"],
        
        # Communication & Collaboration
        "소통": ["communication", "의사소통", "대화", "협업", "팀워크"],
        "협업": ["collaboration", "팀워크", "협력", "공동작업", "소통"],
        "회의": ["meeting", "컨퍼런스", "토론", "협의", "소통"],
        
        # Skills & Learning
        "기술": ["skill", "능력", "전문성", "역량", "실력"],
        "학습": ["learning", "교육", "훈련", "개발", "성장"],
        "교육": ["education", "학습", "훈련", "강의", "교육과정"],
        
        # Environment & Culture
        "환경": ["environment", "상황", "조건", "분위기", "문화"],
        "문화": ["culture", "전통", "가치관", "습관", "환경"],
        "변화": ["change", "변화", "전환", "발전", "혁신"]
    }
    expanded_terms = []
    for keyword, expansions in keywords_map.items():
        if keyword.lower() in query.lower():
            expanded_terms.extend(expansions)
            if len(expanded_terms) >= 4: break
    
    unique_terms = list(dict.fromkeys(expanded_terms))[:4]
    enhanced_query = f"{query} {' '.join(unique_terms)}" if unique_terms else query
    
    return {"expanded_query": enhanced_query, "expansion_terms": unique_terms}

def classify_query(query: str) -> Dict[str, Any]:
    """Classify the query type with confidence scoring."""
    query_lower = query.lower().strip()
    patterns = {
        "factual": (["무엇", "what", "어떤"], ["정의", "설명"], 1.0),
        "procedural": (["어떻게", "how", "방법"], ["과정", "절차"], 1.0),
        "causal": (["왜", "why", "이유"], ["원인", "결과"], 1.0),
        "temporal": (["언제", "when"], ["년도", "미래", "과거"], 1.0),
        "comparative": (["비교", "차이", "vs"], ["장단점"], 1.0),
        "quantitative": (["얼마", "수량", "개수"], ["통계", "%"], 1.0),
        "general": (["대해", "관련", "about"], ["설명", "알려줘"], 0.8)
    }
    scores = {}
    for q_type, (primary, secondary, weight) in patterns.items():
        score = sum(1.0 for p in primary if p in query_lower) + sum(0.5 for p in secondary if p in query_lower)
        scores[q_type] = score * weight

    if max(scores.values()) > 0:
        query_type = max(scores, key=scores.get)
        confidence = min(0.95, 0.3 + (scores[query_type] * 0.3))
    else:
        query_type = "general"
        confidence = 0.5
        
    return {"query_type": query_type, "confidence": confidence, "classification_scores": scores}

# --- Retrieval Modules ---
def retrieve_semantic(vector_store_manager: VectorStoreManager, query: str, k: int) -> List[Document]:
    """Perform semantic retrieval using vector similarity."""
    vector_store = vector_store_manager.get_vector_store()
    if not vector_store: return []
    try:
        return vector_store.similarity_search(query, k=k)
    except Exception as e:
        logger.error(f"Semantic retrieval failed: {e}", exc_info=True)
        return []

def retrieve_keyword(bm25_index: BM25, all_bm25_docs: List[Document], query: str, k: int) -> List[Document]:
    """Perform BM25-based keyword retrieval."""
    if not bm25_index or not all_bm25_docs: return []
    try:
        top_results = bm25_index.get_top_k_indices(query, k=k)
        return [all_bm25_docs[i] for i, score in top_results]
    except Exception as e:
        logger.error(f"Keyword retrieval failed: {e}", exc_info=True)
        return []

# --- Post-retrieval Modules ---
def filter_and_diversify(docs: List[Document], max_docs: int = 5) -> List[Document]:
    """Filter by relevance (simple length check) and ensure diversity by source."""
    if not docs: return []
    
    # Filter by min length
    filtered = [doc for doc in docs if len(doc.page_content) >= 50]
    
    # Diversify by source
    if len(filtered) <= 3: return filtered[:max_docs]
        
    diverse_docs, seen_sources = [], set()
    for doc in filtered:
        source = doc.metadata.get("source", "unknown")
        if source not in seen_sources or len(diverse_docs) < 2:
            diverse_docs.append(doc)
            seen_sources.add(source)
        if len(diverse_docs) >= max_docs: break
            
    # If not enough diverse docs, fill with remaining filtered docs
    if len(diverse_docs) < max_docs:
        remaining_docs = [doc for doc in filtered if doc not in diverse_docs]
        diverse_docs.extend(remaining_docs[:max_docs - len(diverse_docs)])

    return diverse_docs

# --- Generation Modules ---
def generate_answer_stream(llm_manager: LLMManager, query: str, docs: List[Document], query_type: str, max_context_length: int = 3500) -> Iterator[str]:
    """Generate an answer stream based on the query type and context."""
    if not docs:
        yield "관련 문서를 찾을 수 없습니다."
        return
        
    context = "\\n\\n".join([doc.page_content for doc in docs])
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."
        
    prompt_templates = {
        "factual": "다음 컨텍스트에서 사실 정보만을 바탕으로 질문에 답해주세요. 정확하지 않으면 '확실하지 않습니다'라고 답하세요.",
        "procedural": "다음 컨텍스트를 바탕으로 단계별 방법을 체계적으로 설명해주세요.",
        "causal": "다음 컨텍스트를 바탕으로 원인과 이유를 논리적으로 분석해서 답변해주세요.",
        "default": "다음 컨텍스트를 바탕으로 종합적으로 답변해주세요."
    }
    system_prompt = prompt_templates.get(query_type, prompt_templates["default"])
    enhanced_prompt = f"{system_prompt}\\n\\n컨텍스트:\\n{context}\\n\\n질문: {query}\\n\\n답변:"
    
    try:
        for chunk in llm_manager.generate_response_stream(prompt=enhanced_prompt, context=""):
            yield chunk
    except Exception as e:
        logger.error(f"Answer generation failed: {e}", exc_info=True)
        yield "답변 생성 중 오류가 발생했습니다."

def estimate_confidence(llm_manager: LLMManager, answer: str, docs: List[Document], classification_confidence: float) -> float:
    """
    Estimate confidence by checking for uncertainty and factual consistency (groundedness).
    """
    logger.info("🔍 === CONFIDENCE CALCULATION DEBUG ===")
    
    # 1. Base score from query classification and answer length
    base_score = 0.3
    classification_bonus = classification_confidence * 0.1
    length_bonus = 0.1 if len(answer) > 50 else 0.0
    
    confidence = base_score + classification_bonus + length_bonus
    
    logger.info(f"📊 Base Score Calculation:")
    logger.info(f"   ├─ Base score: {base_score}")
    logger.info(f"   ├─ Classification confidence: {classification_confidence:.3f} → bonus: {classification_bonus:.3f}")
    logger.info(f"   ├─ Answer length: {len(answer)} chars → bonus: {length_bonus}")
    logger.info(f"   └─ Subtotal: {confidence:.3f}")

    # 2. Penalize for uncertainty keywords
    uncertainty_keywords = ["모른다", "확실하지", "아마도"]
    found_uncertainty_words = [word for word in uncertainty_keywords if word in answer.lower()]
    uncertainty_penalty = 0.3 if found_uncertainty_words else 0.0
    
    confidence -= uncertainty_penalty
    
    logger.info(f"🚫 Uncertainty Penalty:")
    logger.info(f"   ├─ Found uncertainty words: {found_uncertainty_words}")
    logger.info(f"   ├─ Penalty: -{uncertainty_penalty}")
    logger.info(f"   ├─ Answer preview: '{answer[:100]}{'...' if len(answer) > 100 else ''}'")
    logger.info(f"   └─ After penalty: {confidence:.3f}")

    # 3. Check for factual consistency using an LLM call
    faithfulness_bonus = 0.0
    if docs:
        context = "\\n\\n".join([doc.page_content for doc in docs])
        prompt = f"""
        Please act as a fact-checker. You will be given a context and a statement.
        Your task is to determine if the statement is fully supported by the information in the context.
        Answer with only "Yes" or "No".

        Context:
        ---
        {context}
        ---

        Statement:
        ---
        {answer}
        ---

        Is the statement fully supported by the context? Answer "Yes" or "No".
        """
        try:
            faithfulness_response = llm_manager.generate_response(prompt=prompt, context="").strip().lower()
            logger.info(f"Faithfulness check response: {faithfulness_response}")
            if "yes" in faithfulness_response:
                faithfulness_bonus = 0.3  # Big bonus for being grounded
            else:
                faithfulness_bonus = -0.4 # Big penalty for hallucination
        except Exception as e:
            logger.error(f"Faithfulness check failed: {e}")
            faithfulness_bonus = -0.1 # Small penalty if check fails
    else:
        logger.info("No documents available for faithfulness check")

    confidence += faithfulness_bonus
    final_confidence = max(0.0, min(1.0, confidence))
    
    logger.info(f"✅ Faithfulness Check:")
    logger.info(f"   ├─ Documents available: {len(docs) if docs else 0}")
    logger.info(f"   ├─ Faithfulness bonus/penalty: {faithfulness_bonus:+.3f}")
    logger.info(f"   └─ After faithfulness: {confidence:.3f}")
    
    logger.info(f"🎯 Final Confidence Calculation:")
    logger.info(f"   ├─ Before clamping: {confidence:.3f}")
    logger.info(f"   ├─ After clamping [0.0, 1.0]: {final_confidence:.3f}")
    logger.info(f"   └─ Formula: 0.3 + ({classification_confidence:.3f}*0.1) + {length_bonus} - {uncertainty_penalty} + {faithfulness_bonus:+.3f} = {final_confidence:.3f}")
    
    logger.info("🔍 === END CONFIDENCE DEBUG ===")
    
    return final_confidence

# --- Orchestration Modules ---
def check_iteration_stop(confidence: float, iteration: int, max_iterations: int) -> bool:
    """
    Control whether to iterate or stop.
    Stops if confidence is high OR if the number of completed iterations reaches max_iterations - 1.
    """
    # iteration is 0-indexed (0 means 1st iteration completed)
    return confidence >= CONFIDENCE_THRESHOLD or iteration >= max_iterations - 1

# --- System Info ---
def get_modular_rag_system_info() -> Dict[str, Any]:
    """Get information about the Modular RAG system."""
    return {
        "name": "Modular RAG",
        "description": "모듈형 RAG 시스템: 유연한 컴포넌트 기반 아키텍처",
        "components": [
            "Pre-retrieval (Query Expansion, Classification)",
            "Retrieval (Semantic + BM25 Keyword)",
            "Post-retrieval (Filtering, Diversity)",
            "Generation (Answer Generation, Confidence)",
            "Orchestration (Iteration Control)"
        ],
        "features": [
            "모듈형 아키텍처", "7가지 쿼리 유형 분류", "하이브리드 검색 (벡터 + BM25)",
            "반복적 개선", "신뢰도 기반 제어"
        ],
        "advantages": [
            "높은 유연성과 확장성", "질문 유형별 맞춤 처리", "의미적 + 키워드 하이브리드 검색",
            "점진적 품질 개선", "투명한 처리 과정"
        ],
        "retrieval_methods": {
            "semantic": "Dense vector similarity (embeddings)",
            "keyword": "BM25 sparse retrieval",
            "hybrid": "Combination of both methods"
        }
    } 