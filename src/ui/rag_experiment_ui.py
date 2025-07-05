"""RAG experiment UI module for the RAG application."""

import streamlit as st
import time
from typing import Dict, Any, Optional, List

from ..config import *
from ..config.settings import MAX_ITERATIONS, RERANK_TOP_K
from ..utils.llm_manager import LLMManager
from ..utils.vector_store import VectorStoreManager
from ..rag_systems.naive_rag import get_naive_rag_system_info
from ..rag_systems.advanced_rag import get_advanced_rag_system_info
from ..rag_systems.modular_rag import get_modular_rag_system_info, BM25
from ..graphs.naive_rag_graph import create_naive_rag_graph
from ..graphs.advanced_rag_graph import create_advanced_rag_graph
from ..graphs.modular_rag_graph import create_modular_rag_graph


class RAGExperimentUI:
    """UI components for RAG experiment functionality."""
    
    @staticmethod
    def display_rag_experiment_tab():
        """RAG experiment tab with various systems."""
        st.header("🧪 RAG 시스템 실험")
        
        # 벡터 스토어 상태를 확인하고 UI에 표시
        is_ready = RAGExperimentUI._check_vector_store()
        
        # 벡터 스토어가 준비되었을 때만 RAG 시스템 초기화
        if is_ready:
            RAGExperimentUI._initialize_rag_systems()
        
        # 항상 실험 인터페이스를 표시
        RAGExperimentUI._display_experiment_interface(is_ready)
    
    @staticmethod
    def _check_vector_store():
        """벡터 스토어 상태를 확인하고 UI에 표시하며, 준비 여부를 반환합니다."""
        vector_store_manager = st.session_state.get("vector_store_manager")
        vector_store = None
        
        if vector_store_manager:
            try:
                vector_store = vector_store_manager.get_vector_store()
            except Exception as e:
                st.warning(f"⚠️ 기존 벡터 스토어 확인 실패: {str(e)}")
        
        if vector_store is None:
            st.warning("📋 벡터 스토어가 필요합니다.")
            st.info("**다음 중 하나를 수행하세요:**\n"
                     "1. **📚 문서 로딩** 탭에서 문서를 로드한 후 **🔍 벡터 스토어** 탭에서 새 벡터 스토어 생성\n"
                     "2. **🔍 벡터 스토어** 탭에서 기존에 저장된 벡터 스토어 로딩")
            return False
        
        st.success("✅ 벡터 스토어 준비 완료!")
        RAGExperimentUI._display_vector_store_status(vector_store)
        return True
    
    @staticmethod
    def _display_vector_store_status(vector_store):
        """Display current vector store status with detailed information."""
        with st.expander("📊 현재 벡터 스토어 정보", expanded=True):
            try:
                # Get vector store manager from session state
                vector_store_manager = st.session_state.get("vector_store_manager")
                
                if vector_store_manager:
                    # Get collection stats
                    stats = vector_store_manager.get_collection_stats()
                    
                    # Get vector store metadata from both sources
                    manager_metadata = getattr(vector_store_manager, '_metadata', {})
                    session_metadata = st.session_state.get('vector_store_metadata', {})
                    
                    # Merge metadata (session state takes precedence for newer info)
                    metadata = {**manager_metadata, **session_metadata}
                    
                    # Display basic stats in columns
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        doc_count = stats.get("document_count", metadata.get("document_count", "N/A"))
                        st.metric("📄 문서 수", doc_count)
                    
                    with col2:
                        vs_type = getattr(vector_store_manager, 'vector_store_type', 'Unknown')
                        st.metric("🔧 벡터 스토어 타입", vs_type.upper())
                    
                    with col3:
                        collection_name = getattr(vector_store_manager, 'collection_name', 'N/A')
                        st.metric("🗂️ 컬렉션", collection_name)
                    
                    with col4:
                        source = st.session_state.get('vector_store_source', 'unknown')
                        source_emoji = {"created": "🆕", "loaded": "📥", "manual_loaded": "🔧", "auto_created": "⚙️"}.get(source, "❓")
                        st.metric("📍 출처", f"{source_emoji} {source}")
                    
                    with col5:
                        telemetry_status = stats.get("telemetry_status", "비활성화")
                        st.metric("🚫 텔레메트리", telemetry_status)
                    
                    # Show vector store sync status
                    current_vs_id = st.session_state.get("vector_store_id")
                    rag_vs_id = st.session_state.get("last_rag_vector_store_id")
                    
                    if current_vs_id and rag_vs_id:
                        if current_vs_id == rag_vs_id:
                            st.success("🔄 **RAG 시스템과 벡터 스토어가 동기화됨**")
                        else:
                            st.warning("⚠️ **벡터 스토어가 변경되었습니다.** 실험을 다시 시작하면 새로운 벡터 스토어가 적용됩니다.")
                    else:
                        st.info("ℹ️ **RAG 시스템이 아직 초기화되지 않았습니다.** 실험을 시작하면 현재 벡터 스토어로 초기화됩니다.")
                    
                    # Detailed information table
                    st.write("### 📋 상세 정보")
                    
                    # Check if we have stored metadata from vector store creation/loading
                    vector_store_info = []
                    
                    # Try to get metadata from different sources
                    if metadata:
                        # From stored metadata
                        vector_store_info.extend([
                            ("📅 생성 시간", metadata.get('created_at', 'N/A')),
                            ("🏷️ 저장 이름", metadata.get('store_name', 'N/A')),
                            ("📊 총 문자 수", f"{metadata.get('total_characters', 0):,}"),
                            ("📚 소스 파일 수", metadata.get('source_count', 'N/A')),
                            ("📏 평균 청크 크기", f"{metadata.get('avg_chunk_size', 0):.0f}"),
                            ("🤖 임베딩 모델", metadata.get('embedding_model', 'N/A')),
                            ("🔪 청크 크기", metadata.get('chunk_size', 'N/A')),
                            ("🔗 청크 오버랩", metadata.get('chunk_overlap', 'N/A'))
                        ])
                    else:
                        # Basic information when no metadata available
                        from ..config import EMBEDDING_MODEL
                        
                        vector_store_info.extend([
                            ("🤖 임베딩 모델", EMBEDDING_MODEL),
                            ("🔧 벡터 스토어 상태", stats.get("status", "활성")),
                            ("🔪 현재 청크 크기", st.session_state.get("chunk_size", "N/A")),
                            ("🔗 현재 청크 오버랩", st.session_state.get("chunk_overlap", "N/A")),
                            ("🔍 기본 검색 수", st.session_state.get("top_k", "N/A"))
                        ])
                    
                    # Display information in a clean format
                    for i in range(0, len(vector_store_info), 2):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if i < len(vector_store_info):
                                key, value = vector_store_info[i]
                                st.write(f"**{key}:** {value}")
                        
                        with col2:
                            if i + 1 < len(vector_store_info):
                                key, value = vector_store_info[i + 1]
                                st.write(f"**{key}:** {value}")
                    
                    # Sample document information
                    st.write("### 🔍 샘플 문서 정보")
                    sample_docs = vector_store.similarity_search("test", k=3)
                    if sample_docs:
                        st.success(f"✅ 검색 테스트 성공 - {len(sample_docs)}개 문서 발견")
                        
                        # Show sample documents in a table
                        sample_data = []
                        for i, doc in enumerate(sample_docs):
                            sample_data.append({
                                "순번": i + 1,
                                "출처": doc.metadata.get('source', 'Unknown'),
                                "페이지": doc.metadata.get('page_number', 'N/A'),
                                "문자 수": len(doc.page_content),
                                "내용 미리보기": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                            })
                        
                        import pandas as pd
                        df_samples = pd.DataFrame(sample_data)
                        st.dataframe(df_samples, use_container_width=True)
                        
                        # Show unique sources
                        unique_sources = set(doc.metadata.get('source', 'Unknown') for doc in sample_docs)
                        if len(unique_sources) > 1:
                            st.info(f"📚 **발견된 소스 파일:** {', '.join(sorted(unique_sources))}")
                        else:
                            st.info(f"📚 **주요 소스 파일:** {list(unique_sources)[0]}")
                    else:
                        st.warning("⚠️ 벡터 스토어가 비어있거나 접근할 수 없습니다.")
                        
                else:
                    st.error("❌ 벡터 스토어 매니저를 찾을 수 없습니다.")
                    
            except Exception as e:
                st.error(f"❌ 벡터 스토어 상태 확인 실패: {str(e)}")
                
                # Show basic fallback information
                st.write("### ⚠️ 기본 정보")
                st.write(f"**🤖 임베딩 모델:** {st.session_state.get('selected_embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')}")
                st.write(f"**🔧 벡터 스토어 타입:** {st.session_state.get('vector_store_type', 'faiss').upper()}")
                st.write(f"**🔪 청크 크기:** {st.session_state.get('chunk_size', 'N/A')}")
                st.write(f"**🔗 청크 오버랩:** {st.session_state.get('chunk_overlap', 'N/A')}")
    
    @staticmethod
    def _initialize_rag_systems():
        """Initialize RAG systems if not already done."""
        # Check if vector store has changed
        current_vector_store_id = st.session_state.get("vector_store_id")
        last_rag_vector_store_id = st.session_state.get("last_rag_vector_store_id")
        
        # Get existing RAG systems safely
        existing_rag_systems = st.session_state.get("rag_systems", {})
        
        # Reset RAG systems if vector store changed or if systems don't exist
        if (not existing_rag_systems or 
            current_vector_store_id != last_rag_vector_store_id):
            
            # Clear existing systems if vector store changed
            if (current_vector_store_id != last_rag_vector_store_id and 
                existing_rag_systems):
                st.info("🔄 **벡터 스토어가 변경되었습니다.** RAG 시스템을 재초기화합니다.")
                # Clear previous results safely
                if "experiment_results" not in st.session_state:
                    st.session_state.experiment_results = []
                else:
                    st.session_state.experiment_results = []
            
            selected_model = st.session_state.get("selected_llm_model", DEFAULT_LLM_MODEL)
            llm_temperature = st.session_state.get("llm_temperature", 0.1)
            llm_manager = LLMManager(selected_model, OLLAMA_BASE_URL, temperature=llm_temperature)
            
            vector_store_manager = st.session_state.vector_store_manager
            
            st.session_state.rag_systems = {
                "Naive RAG": get_naive_rag_system_info(),
                "Advanced RAG": get_advanced_rag_system_info(),
                "Modular RAG": get_modular_rag_system_info()
            }
            
            # Track the vector store ID used for RAG systems
            st.session_state.last_rag_vector_store_id = current_vector_store_id
    
    @staticmethod
    def _display_experiment_interface(is_ready: bool):
        """Display the main experiment interface."""
        # System selection
        st.subheader("🎯 실험 설정")

        # Disable inputs if the system is not ready
        with st.container(border=True):
            if not is_ready:
                st.info("실험을 진행하려면 먼저 위의 안내에 따라 벡터 스토어를 준비해주세요.")

            rag_systems = st.session_state.get("rag_systems", {
                "Naive RAG": get_naive_rag_system_info(),
                "Advanced RAG": get_advanced_rag_system_info(),
                "Modular RAG": get_modular_rag_system_info()
            })

            col1, col2 = st.columns(2)
            with col1:
                selected_systems = st.multiselect(
                    "테스트할 RAG 시스템 선택:",
                    list(rag_systems.keys()),
                    default=list(rag_systems.keys()),
                    disabled=not is_ready
                )
            
            with col2:
                retrieval_k = st.slider("검색할 문서 수 (k):", 1, 15, st.session_state.get("top_k", 5), disabled=not is_ready)
            
            # BM25 Indexing for Modular RAG
            if "Modular RAG" in selected_systems:
                st.subheader("🔑 Modular RAG: BM25 인덱싱")
                if is_ready:
                    RAGExperimentUI._manage_bm25_indexing()
                else:
                    st.info("BM25 인덱싱은 벡터 스토어가 준비된 후에 가능합니다.")

            # Display sample queries
            RAGExperimentUI._display_sample_queries()
            
            # Query input
            query = RAGExperimentUI._display_query_input()
            
            # Run experiment
            if st.button("🚀 실험 실행", type="primary", use_container_width=True):
                if not is_ready:
                    st.error("❌ 벡터 스토어가 준비되지 않았습니다. 실험을 실행할 수 없습니다.")
                elif not query:
                    st.warning("⚠️ 질문을 입력해주세요.")
                elif not selected_systems:
                    st.warning("⚠️ 하나 이상의 RAG 시스템을 선택해주세요.")
                else:
                    RAGExperimentUI._run_experiment(query, selected_systems, retrieval_k)

    @staticmethod
    def _display_sample_queries():
        """Display sample queries categorized by type."""
        st.write("**샘플 질문 (질문 유형별):**")
        sample_queries = [
            "2025년 AI 트렌드는 무엇인가요?",                    # factual
            "직장에서 AI를 어떻게 활용할 수 있나요?",            # procedural
            "왜 AI가 업무 생산성에 중요한가요?",                # causal
            "AI 기술은 언제부터 발전하기 시작했나요?",          # temporal
            "생성형 AI와 기존 AI의 차이점은 무엇인가요?",        # comparative
            "AI 시장 규모는 얼마나 되나요?",                   # quantitative
            "인공지능에 대해 알려주세요.",                     # general
            "머신러닝 모델을 구축하는 방법은?",                # procedural
            "AI 개발에는 어떤 비용이 드나요?",                 # quantitative
            "딥러닝이 주목받는 이유는 무엇인가요?"              # causal
        ]
        
        # Display categorized sample questions
        st.write("**🏷️ 질문 유형 예시:**")
        st.markdown("""
        - **사실형(factual)**: "무엇", "어떤" → 정확한 정보 위주
        - **방법형(procedural)**: "어떻게", "방법" → 단계별 설명  
        - **원인형(causal)**: "왜", "이유" → 논리적 분석
        - **시간형(temporal)**: "언제", "시점" → 시간순 정리
        - **비교형(comparative)**: "차이", "비교" → 비교 분석
        - **수치형(quantitative)**: "얼마", "규모" → 데이터 기반
        - **일반형(general)**: 기타 → 종합적 설명
        """)
        
        st.write("**📝 테스트 질문 목록:**")
        
        # Sample query types for reference
        query_types = ["factual", "procedural", "causal", "temporal", "comparative", "quantitative", "general", "procedural", "quantitative", "causal"]
        
        cols = st.columns(2)
        for i, sample_query in enumerate(sample_queries):
            col = cols[i % 2]
            query_type = query_types[i] if i < len(query_types) else "general"
            type_emoji = {
                "factual": "🎯", "procedural": "📋", "causal": "🤔", 
                "temporal": "⏰", "comparative": "⚖️", "quantitative": "📊", "general": "📖"
            }
            emoji = type_emoji.get(query_type, "📝")
            
            if col.button(f"{emoji} {sample_query}", key=f"sample_{i}", help=f"질문 유형: {query_type}"):
                st.session_state.text_area_value = sample_query
                st.rerun()
    
    @staticmethod
    def _display_query_input():
        """Display query input area."""
        # Query input - 샘플 질문 선택 시 해당 질문이 입력창에 표시됨
        query = st.text_area(
            "질문을 입력하세요:",
            value=st.session_state.get("text_area_value", ""),
            placeholder="예: 2025년 AI 트렌드는 무엇인가요?",
            height=100,
            key="query_input"
        )
        return query
    
    @staticmethod
    def _run_experiment(query, selected_systems, retrieval_k):
        """Run the RAG experiment with selected systems."""
        # This check is now the primary guard before running.
        if "vector_store_manager" not in st.session_state or not st.session_state.vector_store_manager.get_vector_store():
            st.error("❌ 벡터 스토어가 설정되지 않았습니다. 먼저 벡터 스토어를 생성하거나 로드해주세요.")
            return

        results = []
        
        # Get RAG systems safely
        rag_systems = st.session_state.get("rag_systems", {})
        
        if not rag_systems:
            st.error("❌ RAG 시스템이 초기화되지 않았습니다.")
            return
        
        for system_name in selected_systems:
            st.write(f"## {system_name} 실행 중...")
            
            if system_name not in rag_systems:
                st.error(f"❌ {system_name} 시스템을 찾을 수 없습니다.")
                continue
                
            rag_system = rag_systems[system_name]
            
            try:
                with st.spinner(f"{system_name} 실행 중... 답변을 생성하고 있습니다."):
                    if system_name == "Naive RAG":
                        selected_model = st.session_state.get("selected_llm_model", DEFAULT_LLM_MODEL)
                        llm_temperature = st.session_state.get("llm_temperature", 0.1)
                        llm_manager = LLMManager(selected_model, OLLAMA_BASE_URL, temperature=llm_temperature)
                        vector_store_manager = st.session_state.vector_store_manager
                        
                        result = RAGExperimentUI._run_naive_rag(query, retrieval_k, llm_manager, vector_store_manager)

                    elif system_name == "Advanced RAG":
                        selected_model = st.session_state.get("selected_llm_model", DEFAULT_LLM_MODEL)
                        llm_temperature = st.session_state.get("llm_temperature", 0.1)
                        llm_manager = LLMManager(selected_model, OLLAMA_BASE_URL, temperature=llm_temperature)
                        vector_store_manager = st.session_state.vector_store_manager

                        result = RAGExperimentUI._run_advanced_rag(query, retrieval_k * 2, RERANK_TOP_K, llm_manager, vector_store_manager)
                        
                    elif system_name == "Modular RAG":
                        if "bm25_index" not in st.session_state or "bm25_documents" not in st.session_state:
                            st.error("❌ Modular RAG를 실행하려면 먼저 BM25 인덱스를 생성해야 합니다.")
                            continue

                        selected_model = st.session_state.get("selected_llm_model", DEFAULT_LLM_MODEL)
                        llm_temperature = st.session_state.get("llm_temperature", 0.1)
                        llm_manager = LLMManager(selected_model, OLLAMA_BASE_URL, temperature=llm_temperature)
                        vector_store_manager = st.session_state.vector_store_manager
                        bm25_index = st.session_state.bm25_index
                        bm25_documents = st.session_state.bm25_documents

                        result = RAGExperimentUI._run_modular_rag(
                            query, MAX_ITERATIONS, llm_manager, vector_store_manager, bm25_index, bm25_documents
                        )
                    else:
                        # Fallback for any other system that might still use the old class structure
                        result = rag_system.query(query, k=retrieval_k)
                
                results.append(result)
                
                # Display individual result
                st.write(f"**답변:**")
                st.info(result.get('answer', '답변을 생성하지 못했습니다.'))
                st.write(f"**처리 시간:** {result.get('total_time', 0):.2f}초")
                
                retrieved_docs = result.get("retrieved_docs", [])
                if retrieved_docs:
                    with st.expander(f"검색된 문서 ({len(retrieved_docs)}개)"):
                        for i, doc in enumerate(retrieved_docs):
                            st.write(f"**문서 {i+1} (출처: {doc.metadata.get('source', 'Unknown')})**")
                            st.text(doc.page_content[:200] + "...")
                            st.divider()
                
            except Exception as e:
                st.error(f"{system_name} 실행 실패: {str(e)}")
                import traceback
                st.error(f"상세 오류 정보: {traceback.format_exc()}")
                continue
            
            st.divider()
        
        # Store results
        if results:
            st.session_state.experiment_results = results
            st.success("✅ 모든 실험 완료!")
    
    @staticmethod
    def _manage_bm25_indexing():
        """UI for managing the BM25 index required for Modular RAG."""
        bm25_index = st.session_state.get("bm25_index")
        bm25_docs_count = len(st.session_state.get("bm25_documents", []))

        if bm25_index:
            st.success(f"✅ BM25 인덱스가 준비되었습니다. ({bm25_docs_count}개 문서 인덱싱됨)")
            if st.button("🔄 BM25 인덱스 재생성", key="experiment_regenerate_bm25"):
                st.session_state.pop("bm25_index", None)
                st.session_state.pop("bm25_documents", None)
                st.rerun()
        else:
            st.warning("BM25 키워드 검색을 위해 인덱스 생성이 필요합니다.")
            if st.button("🚀 BM25 인덱스 생성", key="experiment_create_bm25"):
                try:
                    with st.spinner("벡터 스토어에서 문서를 로드하여 BM25 인덱스를 생성합니다..."):
                        vector_store_manager = st.session_state.get("vector_store_manager")
                        if not vector_store_manager:
                            st.error("벡터 스토어 매니저를 찾을 수 없습니다.")
                            return

                        vector_store = vector_store_manager.get_vector_store()
                        
                        stats = vector_store_manager.get_collection_stats()
                        total_docs = stats.get("document_count", 1000)
                        
                        if total_docs == 0:
                            st.error("인덱싱할 문서가 벡터 스토어에 없습니다.")
                            return

                        docs = vector_store.similarity_search("", k=total_docs)
                        
                        if not docs:
                            st.error("벡터 스토어에서 문서를 가져오지 못했습니다.")
                            return
                        
                        corpus = [doc.page_content for doc in docs]
                        st.session_state.bm25_index = BM25(corpus)
                        st.session_state.bm25_documents = docs
                    st.rerun()
                except Exception as e:
                    st.error(f"BM25 인덱스 생성 실패: {e}")
                    if "bm25_index" in st.session_state:
                        del st.session_state.bm25_index
                    if "bm25_documents" in st.session_state:
                        del st.session_state.bm25_documents
    
    @staticmethod
    def _run_modular_rag(query: str, max_iterations: int, llm_manager: LLMManager, vector_store_manager: VectorStoreManager, bm25_index: BM25, bm25_docs: List[Any]) -> Dict[str, Any]:
        """Run the Modular RAG using the compiled graph and display progress via streaming."""
        start_time = time.time()
        
        modular_rag_graph = create_modular_rag_graph(llm_manager, vector_store_manager, bm25_index, bm25_docs)
        inputs = {"query": query, "max_iterations": max_iterations}
        
        st.subheader("🧩 모듈형 RAG 처리 과정")
        
        # Placeholders for real-time updates
        preprocess_placeholder = st.empty()
        iteration_placeholder = st.empty()
        final_summary_placeholder = st.empty()
        
        final_state = {}
        with st.spinner("Modular RAG 그래프 실행 중..."):
            for state in modular_rag_graph.stream(inputs, config={"callbacks": [langfuse_handler]}):
                node_name, node_output = list(state.items())[0]
                final_state.update(node_output)

                with preprocess_placeholder.container(border=True):
                    st.write("**1단계: 사전 검색 처리**")
                    st.info(f"쿼리 확장: {final_state.get('expanded_query', '...')}")
                    st.info(f"쿼리 유형: {final_state.get('query_type', '...')} (신뢰도: {final_state.get('classification_confidence', 0):.2f})")

                with iteration_placeholder.container(border=True):
                    st.write(f"**2단계: 반복적 개선 (현재 {final_state.get('iteration', 0) + 1}번째 실행 중)**")
                    if node_name == "retrieve_and_process":
                        st.success(f"문서 검색 및 처리 완료: {len(final_state.get('retrieved_docs', []))}개 문서 선택됨")
                    if node_name == "generate":
                        st.success("답변 생성 완료")
                        st.info(f"중간 답변: {final_state.get('answer', '')[:100]}...")
                        st.warning(f"현재 신뢰도: {final_state.get('final_confidence', 0):.2f}")

        # Final display after streaming is complete
        iteration_placeholder.empty()
        preprocess_placeholder.empty()

        st.write("**1단계: 사전 검색 처리**")
        st.info(f"쿼리 확장: {final_state.get('expanded_query')}")
        st.info(f"쿼리 유형: {final_state.get('query_type')} (신뢰도: {final_state.get('classification_confidence', 0):.2f})")

        st.write(f"**2단계: 반복적 개선 (총 {final_state.get('iteration', 0) + 1}회 실행)**")

        st.subheader("🤖 최종 답변")
        answer = final_state.get("answer", "답변을 생성하지 못했습니다.")
        st.markdown(answer)
        
        st.subheader("📈 최종 결과 요약")
        total_time = time.time() - start_time
        all_retrieved_docs = final_state.get("all_retrieved_docs", [])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("총 처리 시간", f"{total_time:.2f}초")
        col2.metric("총 반복 횟수", final_state.get('iteration', 0) + 1)
        col3.metric("최종 신뢰도", f"{final_state.get('final_confidence', 0):.2f}")

        with st.expander(f"최종 검색된 문서 보기 ({len(all_retrieved_docs)}개)"):
            for i, doc in enumerate(all_retrieved_docs):
                st.write(f"**문서 {i+1} (출처: {doc.metadata.get('source', 'Unknown')})**")
                st.text(doc.page_content[:200] + "...")
                st.divider()

        system_info = get_modular_rag_system_info()
        return {
            "question": query,
            "answer": answer,
            "retrieved_docs": all_retrieved_docs,
            "total_time": total_time,
            "rag_type": system_info["name"],
            "metadata": {
                "iterations": final_state.get("iteration", 0),
                "final_confidence": final_state.get("final_confidence", 0.0),
                "query_type": final_state.get("query_type", "general"),
                "total_retrieved": len(all_retrieved_docs),
                "expansion_terms": final_state.get("expansion_terms", [])
            }
        }

    @staticmethod
    def _run_advanced_rag(query: str, k: int, rerank_top_k: int, llm_manager: LLMManager, vector_store_manager: VectorStoreManager) -> Dict[str, Any]:
        """Run the Advanced RAG using the compiled graph and display progress."""
        start_time = time.time()
        
        advanced_rag_graph = create_advanced_rag_graph(llm_manager, vector_store_manager)
        
        inputs = {"query": query, "k": k, "rerank_top_k": rerank_top_k}
        
        final_state = {}
        with st.spinner("Advanced RAG 그래프 실행 중..."):
            final_state = advanced_rag_graph.invoke(inputs, config={"callbacks": [langfuse_handler]})

        # 1. 쿼리 전처리 결과 표시
        st.subheader("🔧 1단계: 쿼리 전처리")
        preprocess_details = final_state.get("preprocessing_details", {})
        optimized_query = final_state.get("optimized_query")

        if optimized_query != query:
            st.success(f"쿼리 최적화 완료")
            with st.expander("쿼리 확장 상세 정보 보기"):
                st.write(f"- **원본 쿼리:** `{preprocess_details.get('original_query')}`")
                st.write(f"- **선택된 확장 용어:** `{preprocess_details.get('selected_terms')}`")
                st.write(f"- **최종 확장 쿼리:** `{optimized_query}`")
        else:
            st.info("쿼리 최적화: 변경사항 없음")
        
        # 2. 문서 검색 결과
        st.subheader("🔍 2단계: 문서 검색")
        docs_with_scores = final_state.get("docs_with_scores", [])
        st.success(f"초기 검색: {len(docs_with_scores)}개 문서")

        # 3. 문서 재순위화 결과
        st.subheader("📊 3단계: 문서 재순위화")
        reranked_docs = final_state.get("reranked_docs", [])
        st.success(f"재순위화 완료: 상위 {len(reranked_docs)}개 문서 선택")

        with st.expander(f"재순위화된 문서 ({len(reranked_docs)}개)"):
            for i, doc in enumerate(reranked_docs):
                st.write(f"**문서 {i+1} (출처: {doc.metadata.get('source', 'Unknown')})**")
                st.text(doc.page_content[:200] + "...")
                st.divider()

        # 4. 컨텍스트 압축 결과
        st.subheader("🗜️ 4단계: 컨텍스트 압축")
        compression_ratio = final_state.get("compression_ratio", 0)
        st.info(f"압축률: {compression_ratio:.2%}")
        
        # 5. 최종 답변
        st.subheader("🤖 5단계: 답변 생성")
        answer = final_state.get("answer", "답변을 생성하지 못했습니다.")
        
        total_time = time.time() - start_time

        system_info = get_advanced_rag_system_info()
        return {
            "question": query,
            "answer": answer,
            "retrieved_docs": reranked_docs,
            "total_time": total_time,
            "rag_type": system_info["name"],
            "metadata": {
                "optimized_query": optimized_query,
                "initial_retrieved": len(docs_with_scores),
                "final_retrieved": len(reranked_docs),
                "compression_ratio": compression_ratio,
                "retrieval_method": "similarity_search + reranking",
                "generation_method": "reasoning-based"
            }
        }

    @staticmethod
    def _run_naive_rag(query: str, k: int, llm_manager: LLMManager, vector_store_manager: VectorStoreManager) -> Dict[str, Any]:
        """Run the Naive RAG using the compiled graph and return results."""
        start_time = time.time()
        
        naive_rag_graph = create_naive_rag_graph(llm_manager, vector_store_manager)
        
        inputs = {"query": query, "k": k}
        final_state = naive_rag_graph.invoke(inputs, config={"callbacks": [langfuse_handler]})

        end_time = time.time()
        total_time = end_time - start_time

        retrieved_docs = final_state.get("documents", [])
        system_info = get_naive_rag_system_info()
        
        return {
            "question": query,
            "answer": final_state.get("answer"),
            "retrieved_docs": retrieved_docs,
            "total_time": total_time,
            "rag_type": system_info["name"],
            "metadata": {
                "num_retrieved": len(retrieved_docs),
                "retrieval_method": "similarity_search",
                "generation_method": "simple"
            }
        }