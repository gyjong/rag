"""Modular RAG implementation with flexible component-based architecture."""

from typing import List, Dict, Any, Optional, Callable, Tuple
import time
import re
import numpy as np
import streamlit as st
import pandas as pd
from langchain_core.documents import Document
from enum import Enum

from ..utils.vector_store import VectorStoreManager
from ..utils.llm_manager import LLMManager

# BM25 Implementation (standalone to avoid external dependencies)
class BM25:
    """BM25 ranking algorithm implementation."""
    
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        """Initialize BM25 with corpus.
        
        Args:
            corpus: List of documents (strings)
            k1: Term frequency normalization parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.corpus_size = len(corpus)
        
        # Tokenize and process corpus
        self.doc_tokens = [self._tokenize(doc) for doc in corpus]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0
        
        # Build vocabulary and document frequency
        self.vocab = set()
        for tokens in self.doc_tokens:
            self.vocab.update(tokens)
        self.vocab = list(self.vocab)
        
        # Calculate document frequencies
        self.df = {}
        for term in self.vocab:
            self.df[term] = sum(1 for tokens in self.doc_tokens if term in tokens)
        
        # Pre-compute IDF scores
        self.idf = {}
        for term in self.vocab:
            self.idf[term] = np.log((self.corpus_size - self.df[term] + 0.5) / (self.df[term] + 0.5))
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (can be enhanced with proper NLP tokenizer)."""
        # Basic tokenization: lowercase, remove punctuation, split by spaces
        text = text.lower()
        text = re.sub(r'[^\w\s가-힣]', ' ', text)  # Keep Korean characters
        tokens = text.split()
        # Filter short tokens
        tokens = [token for token in tokens if len(token) > 1]
        return tokens
    
    def get_scores(self, query: str) -> List[float]:
        """Calculate BM25 scores for query against all documents."""
        query_tokens = self._tokenize(query)
        scores = []
        
        for i, doc_tokens in enumerate(self.doc_tokens):
            score = 0.0
            doc_len = self.doc_len[i]
            
            # Count term frequencies in document
            tf = {}
            for token in doc_tokens:
                tf[token] = tf.get(token, 0) + 1
            
            # Calculate BM25 score
            for term in query_tokens:
                if term in tf:
                    # TF component
                    tf_component = tf[term] * (self.k1 + 1)
                    tf_component /= (tf[term] + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
                    
                    # IDF component
                    idf_component = self.idf.get(term, 0)
                    
                    score += tf_component * idf_component
            
            scores.append(score)
        
        return scores
    
    def get_top_k(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """Get top-k documents with scores."""
        scores = self.get_scores(query)
        # Get indices and scores, sort by score (descending)
        scored_docs = [(i, score) for i, score in enumerate(scores)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:k]


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
        
        # BM25 index for keyword-based retrieval (lazy initialization)
        self.bm25_index = None
        self.bm25_documents = []
        
        # Initialize modules
        self.modules = {}
        self._setup_modules()
    
    def _initialize_bm25_index(self):
        """Initialize BM25 index for keyword-based retrieval."""
        if self.bm25_index is not None:
            return  # Already initialized
            
        # Check if we have a cached index in session state
        if "bm25_index" in st.session_state and "bm25_documents" in st.session_state:
            self.bm25_index = st.session_state.bm25_index
            self.bm25_documents = st.session_state.bm25_documents
            return
        
        vector_store = self.vector_store_manager.get_vector_store()
        if vector_store is None:
            st.warning("⚠️ 벡터 스토어가 없어 BM25 인덱스를 생성할 수 없습니다.")
            return
            
        try:
            with st.spinner("🔍 BM25 키워드 검색 인덱스 생성 중..."):
                # Get all documents from vector store
                # Note: This is a simplified approach - in production, you'd want a more efficient method
                sample_docs = vector_store.similarity_search("AI", k=100)  # Get many docs
                
                if not sample_docs:
                    st.warning("⚠️ 문서가 없어 BM25 인덱스를 생성할 수 없습니다.")
                    return
                
                # Extract text content
                corpus = [doc.page_content for doc in sample_docs]
                self.bm25_documents = sample_docs
                
                # Create BM25 index
                self.bm25_index = BM25(corpus)
                
                # Cache in session state
                st.session_state.bm25_index = self.bm25_index
                st.session_state.bm25_documents = self.bm25_documents
                
                st.success(f"✅ BM25 인덱스 생성 완료! ({len(corpus)}개 문서 인덱싱)")
                
        except Exception as e:
            st.error(f"❌ BM25 인덱스 생성 실패: {str(e)}")
        
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
            # AI & Technology
            "AI": ["artificial intelligence", "machine learning", "딥러닝", "neural networks", "automation", "intelligent systems"],
            "인공지능": ["AI", "머신러닝", "neural network", "자동화", "지능형 시스템", "algorithm"],
            "머신러닝": ["machine learning", "AI", "인공지능", "data science", "predictive modeling", "ML"],
            "딥러닝": ["deep learning", "neural networks", "AI", "인공지능", "computer vision", "NLP"],
            
            # Trends & Future
            "트렌드": ["동향", "전망", "trend", "future", "방향성", "흐름", "tendency", "outlook"],
            "동향": ["trend", "트렌드", "현황", "상황", "direction", "movement"],
            "전망": ["outlook", "prospect", "예측", "미래", "forecast", "prediction"],
            "미래": ["future", "prospect", "전망", "outlook", "upcoming", "forthcoming"],
            
            # Business & Work
            "업무": ["work", "business", "직장", "productivity", "task", "operation", "job", "profession"],
            "직장": ["workplace", "office", "business", "업무", "career", "employment"],
            "비즈니스": ["business", "enterprise", "commerce", "업무", "corporate", "commercial"],
            "생산성": ["productivity", "efficiency", "performance", "output", "effectiveness"],
            
            # Technology Applications
            "자동화": ["automation", "자동", "streamlining", "optimization", "efficiency"],
            "디지털화": ["digitalization", "digital transformation", "digitization", "modernization"],
            "혁신": ["innovation", "breakthrough", "advancement", "progress", "development"],
            "기술": ["technology", "tech", "technique", "method", "approach"],
            
            # Industry & Sectors
            "산업": ["industry", "sector", "field", "domain", "market"],
            "시장": ["market", "industry", "sector", "business", "commercial"],
            "기업": ["company", "enterprise", "corporation", "business", "organization"],
            
            # Analysis & Research
            "분석": ["analysis", "research", "study", "investigation", "examination"],
            "연구": ["research", "study", "investigation", "analysis", "exploration"],
            "조사": ["survey", "investigation", "research", "study", "examination"],
            
            # Impact & Effects
            "영향": ["impact", "effect", "influence", "consequence", "result"],
            "효과": ["effect", "impact", "result", "outcome", "consequence"],
            "변화": ["change", "transformation", "shift", "evolution", "development"],
            
            # Implementation & Strategy
            "도입": ["implementation", "adoption", "introduction", "deployment", "integration"],
            "전략": ["strategy", "plan", "approach", "method", "tactic"],
            "방안": ["plan", "strategy", "approach", "solution", "method"],
            "방법": ["method", "approach", "technique", "way", "procedure"],
            
            # Performance & Quality
            "성능": ["performance", "efficiency", "capability", "effectiveness", "quality"],
            "품질": ["quality", "standard", "excellence", "performance", "caliber"],
            "효율성": ["efficiency", "productivity", "effectiveness", "performance", "optimization"]
        }
        
        # Limit expansion to avoid overwhelming the query
        for keyword, expansions in keywords_map.items():
            if keyword.lower() in query.lower():
                expanded_terms.extend(expansions[:2])
                # Limit total expansion terms to prevent query bloat
                if len(expanded_terms) >= 4:
                    break
        
        expanded_query = f"{query} {' '.join(expanded_terms[:4])}" if expanded_terms else query
        
        return {
            "original_query": query,
            "expanded_query": expanded_query,
            "expansion_terms": expanded_terms
        }
    
    def _query_classification_module(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced query classification with detailed analysis and confidence scoring."""
        query_lower = query.lower().strip()
        
        # Enhanced pattern matching with scoring
        classification_patterns = {
            "factual": {
                "primary": ["무엇", "what", "어떤", "어떠한", "뭔가", "뭐가"],
                "secondary": ["정의", "definition", "의미", "meaning", "개념", "설명"],
                "weight": 1.0
            },
            "procedural": {
                "primary": ["어떻게", "how", "방법", "방식", "과정", "절차"],
                "secondary": ["단계", "step", "순서", "프로세스", "process", "구현"],
                "weight": 1.0
            },
            "causal": {
                "primary": ["왜", "why", "이유", "원인", "cause", "reason"],
                "secondary": ["때문", "because", "결과", "영향", "effect", "impact"],
                "weight": 1.0
            },
            "temporal": {
                "primary": ["언제", "when", "시점", "시기", "timing"],
                "secondary": ["년도", "year", "미래", "future", "과거", "past", "현재"],
                "weight": 1.0
            },
            "comparative": {
                "primary": ["비교", "차이", "compare", "difference", "vs", "대비"],
                "secondary": ["장단점", "pros", "cons", "좋은", "나쁜", "better", "worse"],
                "weight": 1.0
            },
            "quantitative": {
                "primary": ["얼마", "how much", "how many", "수량", "개수", "비율"],
                "secondary": ["퍼센트", "percent", "%", "통계", "statistics", "수치"],
                "weight": 1.0
            }
        }
        
        # Calculate scores for each query type
        type_scores = {}
        matched_keywords = {}
        
        for q_type, patterns in classification_patterns.items():
            score = 0.0
            keywords = []
            
            # Check primary patterns (higher weight)
            for pattern in patterns["primary"]:
                if pattern in query_lower:
                    score += 1.0
                    keywords.append(f"[주요] {pattern}")
            
            # Check secondary patterns (lower weight)  
            for pattern in patterns["secondary"]:
                if pattern in query_lower:
                    score += 0.5
                    keywords.append(f"[보조] {pattern}")
            
            # Question mark bonus
            if "?" in query or "?" in query:
                score += 0.2
                
            type_scores[q_type] = score * patterns["weight"]
            matched_keywords[q_type] = keywords
        
        # Determine best classification
        if max(type_scores.values()) > 0:
            query_type = max(type_scores, key=type_scores.get)
            confidence = min(0.95, 0.3 + (type_scores[query_type] * 0.3))
        else:
            query_type = "general"
            confidence = 0.5
        
        # Display detailed classification analysis
        st.write("**🔍 질문 분류 상세 분석:**")
        
        # Create classification table
        classification_data = []
        for q_type, score in type_scores.items():
            status = "✅ 선택됨" if q_type == query_type else ""
            keywords_str = ", ".join(matched_keywords[q_type]) if matched_keywords[q_type] else "매칭 없음"
            
            classification_data.append({
                "유형": q_type,
                "점수": f"{score:.2f}",
                "매칭 키워드": keywords_str,
                "상태": status
            })
        
        # Sort by score
        classification_data.sort(key=lambda x: float(x["점수"]), reverse=True)
        
        import pandas as pd
        df = pd.DataFrame(classification_data)
        st.dataframe(df, use_container_width=True)
        
        # Show confidence meter
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**최종 분류:** `{query_type}` (신뢰도: {confidence:.2%})")
        with col2:
            # Simple confidence visualization
            if confidence >= 0.8:
                st.success(f"높음 {confidence:.1%}")
            elif confidence >= 0.6:
                st.warning(f"보통 {confidence:.1%}")
            else:
                st.error(f"낮음 {confidence:.1%}")
        
        # Show classification impact
        classification_effects = {
            "factual": "🎯 정확한 사실 정보 위주로 답변",
            "procedural": "📋 단계별 방법론 중심으로 답변", 
            "causal": "🤔 원인과 이유 분석 중심으로 답변",
            "temporal": "⏰ 시간순 정보 중심으로 답변",
            "comparative": "⚖️ 비교 분석 중심으로 답변",
            "quantitative": "📊 수치와 통계 중심으로 답변",
            "general": "📖 종합적이고 포괄적으로 답변"
        }
        
        st.info(f"**처리 방식:** {classification_effects.get(query_type, '일반적 처리')}")
        
        return {
            "query_type": query_type,
            "confidence": confidence,
            "all_scores": type_scores,
            "matched_keywords": matched_keywords[query_type],
            "classification_details": classification_data
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
        """Perform BM25-based keyword retrieval."""
        st.write("**🔍 BM25 키워드 검색:**")
        
        # Initialize BM25 index if not already done
        self._initialize_bm25_index()
        
        if self.bm25_index is None or not self.bm25_documents:
            st.warning("⚠️ BM25 인덱스가 없어 벡터 검색으로 대체합니다.")
            # Fallback to vector search
            vector_store = self.vector_store_manager.get_vector_store()
            if vector_store:
                k = context.get("retrieval_k", 5)
                return vector_store.similarity_search(query, k=k)
            return []
            
        try:
            k = context.get("retrieval_k", 5)
            
            # Perform BM25 search
            with st.spinner(f"BM25 알고리즘으로 상위 {k}개 문서 검색 중..."):
                start_time = time.time()
                top_docs = self.bm25_index.get_top_k(query, k=k)
                search_time = time.time() - start_time
            
            # Extract documents and scores
            retrieved_docs = []
            search_results = []
            
            for rank, (doc_idx, score) in enumerate(top_docs, 1):
                if score > 0:  # Only include documents with positive scores
                    doc = self.bm25_documents[doc_idx]
                    retrieved_docs.append(doc)
                    search_results.append({
                        "순위": rank,
                        "BM25 점수": f"{score:.3f}",
                        "문서 출처": doc.metadata.get('source', 'Unknown')[:30] + "...",
                        "내용 미리보기": doc.page_content[:100] + "..."
                    })
            
            # Display search results
            if search_results:
                st.write("**📊 BM25 검색 결과:**")
                df = pd.DataFrame(search_results)
                st.dataframe(df, use_container_width=True)
                
                # Search statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("검색 시간", f"{search_time:.3f}초")
                with col2:
                    st.metric("검색된 문서", f"{len(retrieved_docs)}개")
                with col3:
                    st.metric("최고 점수", f"{top_docs[0][1]:.3f}" if top_docs else "0.000")
                with col4:
                    st.metric("평균 점수", f"{np.mean([score for _, score in top_docs]):.3f}" if top_docs else "0.000")
                
                # Query analysis
                query_tokens = self.bm25_index._tokenize(query)
                st.info(f"**분석된 쿼리 토큰:** {', '.join(query_tokens)}")
                
                # Show detailed results for top documents
                with st.expander(f"🔍 상위 {min(3, len(retrieved_docs))}개 문서 상세보기"):
                    for i, (doc, (_, score)) in enumerate(zip(retrieved_docs[:3], top_docs[:3])):
                        st.write(f"**#{i+1} 문서 (BM25: {score:.3f})**")
                        st.write(f"**출처:** {doc.metadata.get('source', 'Unknown')}")
                        st.write(f"**내용:** {doc.page_content[:200]}...")
                        st.divider()
                
                st.success(f"✅ BM25 키워드 검색 완료: {len(retrieved_docs)}개 문서 ({search_time:.3f}초)")
                
            else:
                st.warning("⚠️ 검색된 문서가 없습니다.")
                
            return retrieved_docs
            
        except Exception as e:
            st.error(f"❌ BM25 검색 실패: {str(e)}")
            # Fallback to vector search
            vector_store = self.vector_store_manager.get_vector_store()
            if vector_store:
                k = context.get("retrieval_k", 5)
                return vector_store.similarity_search(query, k=k)
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
        """Enhanced answer generation with query-type specific processing."""
        if not docs:
            return "관련 문서를 찾을 수 없습니다."
            
        doc_context = "\n\n".join([doc.page_content for doc in docs])
        
        # Limit context length
        max_length = context.get("max_context_length", 3500)
        if len(doc_context) > max_length:
            doc_context = doc_context[:max_length] + "..."
            
        query_type = context.get("query_type", "general")
        
        # Enhanced prompt templates with clear differentiation
        prompt_templates = {
            "factual": {
                "instruction": "사실적 정보 중심으로 정확하게 답변해주세요.",
                "system_prompt": """다음 컨텍스트에서 정확한 사실 정보만을 바탕으로 질문에 답해주세요.
- 확실한 정보만 제공하세요
- 추측이나 의견은 배제하세요  
- 정확하지 않으면 '확실하지 않습니다'라고 답해주세요
- 구체적인 수치나 데이터가 있으면 포함하세요""",
                "style": "정확성 우선"
            },
            "procedural": {
                "instruction": "단계별 방법론 중심으로 체계적으로 답변해주세요.",
                "system_prompt": """다음 컨텍스트를 바탕으로 단계별 방법을 체계적으로 설명해주세요.
- 순서대로 번호를 매겨 설명하세요
- 각 단계별로 구체적인 방법을 제시하세요
- 주의사항이나 팁이 있으면 포함하세요
- 실행 가능한 구체적인 방법을 중심으로 답변하세요""",
                "style": "단계별 체계화"
            },
            "causal": {
                "instruction": "원인과 이유 분석 중심으로 논리적으로 답변해주세요.",
                "system_prompt": """다음 컨텍스트를 바탕으로 원인과 이유를 논리적으로 분석해서 답변해주세요.
- '왜냐하면...', '그 이유는...' 등의 표현을 사용하세요
- 원인과 결과의 관계를 명확히 설명하세요
- 배경 상황과 맥락을 포함하세요
- 다양한 요인들 간의 연관성을 설명하세요""",
                "style": "논리적 인과관계"
            },
            "temporal": {
                "instruction": "시간순 정보와 발전 과정 중심으로 답변해주세요.",
                "system_prompt": """다음 컨텍스트를 바탕으로 시간순으로 정리해서 답변해주세요.
- 연도, 시기, 순서를 명확히 표시하세요
- 과거→현재→미래 순으로 설명하세요  
- 발전 과정이나 변화 추이를 포함하세요
- 특정 시점의 중요한 사건이나 변화를 강조하세요""",
                "style": "시간순 정리"
            },
            "comparative": {
                "instruction": "비교 분석 중심으로 장단점을 명확히 답변해주세요.",
                "system_prompt": """다음 컨텍스트를 바탕으로 비교 분석해서 답변해주세요.
- 공통점과 차이점을 명확히 구분하세요
- 장점과 단점을 균형있게 제시하세요
- 표나 목록 형태로 정리해서 설명하세요
- 각각의 특징을 상대적으로 비교해주세요""",
                "style": "비교 분석"
            },
            "quantitative": {
                "instruction": "수치와 통계 정보 중심으로 데이터 기반 답변해주세요.",
                "system_prompt": """다음 컨텍스트에서 수치, 통계, 데이터 중심으로 답변해주세요.
- 구체적인 숫자와 퍼센트를 포함하세요
- 통계 데이터가 있으면 활용하세요
- 양적 변화나 규모를 강조하세요
- 그래프나 차트로 표현 가능한 정보를 제시하세요""",
                "style": "데이터 기반"
            },
            "general": {
                "instruction": "종합적이고 포괄적으로 답변해주세요.",
                "system_prompt": """다음 컨텍스트를 바탕으로 종합적으로 답변해주세요.
- 다양한 관점에서 설명하세요
- 전체적인 개요부터 세부사항까지 포함하세요
- 균형있고 포괄적인 정보를 제공하세요""",
                "style": "종합적 설명"
            }
        }
        
        # Get prompt configuration for current query type
        prompt_config = prompt_templates.get(query_type, prompt_templates["general"])
        
        # Display answer generation strategy
        st.write("**🤖 답변 생성 전략:**")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"📝 {prompt_config['instruction']}")
        with col2:
            st.success(f"🎯 {prompt_config['style']}")
        
        # Create enhanced prompt
        enhanced_prompt = f"""
{prompt_config['system_prompt']}

컨텍스트:
{doc_context}

질문: {query}

답변:"""
        
        st.write("**모듈형 답변 생성 중...**")
        answer_placeholder = st.empty()
        
        start_time = time.time()
        full_response = ""
        
        # Stream the response with enhanced prompt
        for chunk in self.llm_manager.generate_response_stream(
            prompt=enhanced_prompt,
            context=""  # Context already included in enhanced_prompt
        ):
            full_response += chunk
            answer_placeholder.markdown(full_response + "▌")
        
        generation_time = time.time() - start_time
        answer_placeholder.markdown(full_response)
        
        # Show generation summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("생성 시간", f"{generation_time:.2f}초")
        with col2:
            st.metric("답변 길이", f"{len(full_response)}자")
        with col3:
            st.metric("질문 유형", query_type)
            
        st.success(f"✅ {prompt_config['style']} 방식으로 답변 생성 완료!")
        
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
        # Get BM25 index status
        bm25_status = "✅ 활성화" if self.bm25_index is not None else "⏳ 대기중"
        bm25_docs_count = len(self.bm25_documents) if self.bm25_documents else 0
        
        return {
            "name": self.name,
            "description": self.description,
            "components": [
                "Pre-retrieval Modules (Query Expansion, Classification)",
                "Retrieval Modules (Semantic + BM25 Keyword)",
                "Post-retrieval Modules (Filtering, Diversity)",
                "Generation Modules (Answer Generation, Confidence)",
                "Orchestration Modules (Routing, Iteration Control)"
            ],
            "features": [
                "모듈형 아키텍처",
                "7가지 쿼리 유형 분류 (factual, procedural, causal, temporal, comparative, quantitative, general)",
                "하이브리드 검색 (벡터 + BM25)",
                "반복적 개선",
                "신뢰도 기반 제어",
                f"BM25 키워드 검색 {bm25_status} ({bm25_docs_count}개 문서)"
            ],
            "advantages": [
                "높은 유연성과 확장성",
                "질문 유형별 맞춤 처리",
                "의미적 + 키워드 하이브리드 검색",
                "점진적 품질 개선",
                "투명한 처리 과정"
            ],
            "retrieval_methods": {
                "semantic": "Dense vector similarity (embeddings)",
                "keyword": f"BM25 sparse retrieval ({bm25_docs_count} indexed docs)",
                "hybrid": "Combination of both methods"
            }
        } 