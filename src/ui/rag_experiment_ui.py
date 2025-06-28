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
        """Display current vector store status."""
        with st.expander("📊 현재 벡터 스토어 정보"):
            try:
                # Get sample documents to check vector store
                sample_docs = vector_store.similarity_search("test", k=1)
                if sample_docs:
                    st.info(f"📄 로드된 문서 수: 추정 {len(sample_docs)} 개 이상")
                    st.write(f"**샘플 문서 출처:** {sample_docs[0].metadata.get('source', 'Unknown')}")
                else:
                    st.warning("⚠️ 벡터 스토어가 비어있습니다.")
            except Exception as e:
                st.warning(f"⚠️ 벡터 스토어 상태 확인 실패: {str(e)}")
    
    @staticmethod
    def _initialize_rag_systems():
        """Initialize RAG systems if not already done."""
        if not st.session_state.rag_systems:
            selected_model = st.session_state.get("selected_llm_model", DEFAULT_LLM_MODEL)
            llm_temperature = st.session_state.get("llm_temperature", 0.1)
            llm_manager = LLMManager(selected_model, OLLAMA_BASE_URL, temperature=llm_temperature)
            
            vector_store_manager = st.session_state.vector_store_manager
            
            st.session_state.rag_systems = {
                "Naive RAG": NaiveRAG(vector_store_manager, llm_manager),
                "Advanced RAG": AdvancedRAG(vector_store_manager, llm_manager),
                "Modular RAG": ModularRAG(vector_store_manager, llm_manager)
            }
    
    @staticmethod
    def _display_experiment_interface():
        """Display the main experiment interface."""
        # System selection
        st.subheader("🎯 실험 설정")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_systems = st.multiselect(
                "테스트할 RAG 시스템 선택:",
                list(st.session_state.rag_systems.keys()),
                default=list(st.session_state.rag_systems.keys())
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
        
        for system_name in selected_systems:
            st.write(f"## {system_name} 실행 중...")
            
            rag_system = st.session_state.rag_systems[system_name]
            
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