"""RAG experiment UI module for the RAG application."""

import streamlit as st

from ..config import *
from ..utils.llm_manager import LLMManager
from ..rag_systems.naive_rag import NaiveRAG
from ..rag_systems.advanced_rag import AdvancedRAG
from ..rag_systems.modular_rag import ModularRAG


class RAGExperimentUI:
    """UI components for RAG experiment functionality."""
    
    @staticmethod
    def display_rag_experiment_tab():
        """RAG experiment tab with various systems."""
        st.header("🧪 RAG 시스템 실험")
        
        # Check vector store availability
        if not RAGExperimentUI._check_vector_store():
            return
        
        # Initialize RAG systems
        RAGExperimentUI._initialize_rag_systems()
        
        # Display experiment interface
        RAGExperimentUI._display_experiment_interface()
    
    @staticmethod
    def _check_vector_store():
        """Check if vector store is available."""
        # First check if we have a vector store manager with actual vector store
        vector_store_manager = st.session_state.get("vector_store_manager")
        vector_store = None
        
        if vector_store_manager:
            try:
                vector_store = vector_store_manager.get_vector_store()
            except Exception as e:
                st.warning(f"⚠️ 기존 벡터 스토어 확인 실패: {str(e)}")
                vector_store = None
        
        # If no vector store exists, show warning
        if vector_store is None:
            st.warning("📋 벡터 스토어가 필요합니다.")
            st.info("**다음 중 하나를 수행하세요:**")
            st.markdown("""
            1. **📚 문서 로딩** 탭에서 문서를 로드한 후 **🔍 벡터 스토어** 탭에서 새 벡터 스토어 생성
            2. **🔍 벡터 스토어** 탭에서 기존에 저장된 벡터 스토어 로딩
            """)
            return False
        
        # Display vector store info
        st.success("✅ 벡터 스토어 준비 완료!")
        
        # Show current vector store status
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
                "Naive RAG": NaiveRAG(vector_store_manager, llm_manager),
                "Advanced RAG": AdvancedRAG(vector_store_manager, llm_manager),
                "Modular RAG": ModularRAG(vector_store_manager, llm_manager)
            }
            
            # Track the vector store ID used for RAG systems
            st.session_state.last_rag_vector_store_id = current_vector_store_id
    
    @staticmethod
    def _display_experiment_interface():
        """Display the main experiment interface."""
        # System selection
        st.subheader("🎯 실험 설정")
        
        # Get RAG systems safely
        rag_systems = st.session_state.get("rag_systems", {})
        
        if not rag_systems:
            st.warning("⚠️ RAG 시스템이 초기화되지 않았습니다. 벡터 스토어를 먼저 확인해주세요.")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            selected_systems = st.multiselect(
                "테스트할 RAG 시스템 선택:",
                list(rag_systems.keys()),
                default=list(rag_systems.keys())
            )
        
        with col2:
            retrieval_k = st.slider("검색할 문서 수 (k):", 1, 15, st.session_state.top_k)
        
        # Display sample queries
        RAGExperimentUI._display_sample_queries()
        
        # Query input
        query = RAGExperimentUI._display_query_input()
        
        # Run experiment
        if query and selected_systems and st.button("🚀 실험 실행", type="primary"):
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
                if system_name == "Advanced RAG":
                    result = rag_system.query(query, k=retrieval_k*2, rerank_top_k=retrieval_k)
                elif system_name == "Modular RAG":
                    result = rag_system.query(query, max_iterations=2)
                else:
                    result = rag_system.query(query, k=retrieval_k)
                
                results.append(result)
                
                # Display individual result
                st.write(f"**답변:** {result['answer']}")
                st.write(f"**처리 시간:** {result['total_time']:.2f}초")
                
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