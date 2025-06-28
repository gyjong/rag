"""Main Streamlit application for RAG systems comparison."""

import streamlit as st
import os
from pathlib import Path
import base64

# Import our modules
from src.config import *
from src.utils.document_processor import DocumentProcessor
from src.utils.embeddings import EmbeddingManager
from src.utils.vector_store import VectorStoreManager
from src.utils.llm_manager import LLMManager
from src.utils.font_utils import apply_custom_css
from src.rag_systems.naive_rag import NaiveRAG
from src.rag_systems.advanced_rag import AdvancedRAG
from src.rag_systems.modular_rag import ModularRAG
from src.ui.comparison_ui import ComparisonUI


def load_custom_font():
    """Load custom font if available."""
    return apply_custom_css()


def initialize_session_state():
    """Initialize session state variables."""
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "vector_store_created" not in st.session_state:
        st.session_state.vector_store_created = False
    if "rag_systems" not in st.session_state:
        st.session_state.rag_systems = {}
    if "experiment_results" not in st.session_state:
        st.session_state.experiment_results = []
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "document_chunks" not in st.session_state:
        st.session_state.document_chunks = []


def setup_page():
    """Setup page configuration."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom font globally first
    font_loaded = apply_custom_css()
    
    # Display title with custom styling
    if font_loaded:
        st.markdown('<h1 class="main-title">🤖 RAG Systems Comparison Tool</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">단계별 Naive RAG, Advanced RAG, Modular RAG 비교 실험 애플리케이션</p>', unsafe_allow_html=True)
    else:
        st.title("🤖 RAG Systems Comparison Tool")
        st.subheader("단계별 Naive RAG, Advanced RAG, Modular RAG 비교 실험 애플리케이션")


def setup_sidebar():
    """Setup sidebar with system information."""
    st.sidebar.title("⚙️ 시스템 설정")
    
    # Model Selection
    st.sidebar.subheader("🤖 모델 선택")
    selected_model_name = st.sidebar.selectbox(
        "LLM 모델 선택:",
        options=list(AVAILABLE_LLM_MODELS.keys()),
        index=list(AVAILABLE_LLM_MODELS.values()).index(DEFAULT_LLM_MODEL) if DEFAULT_LLM_MODEL in AVAILABLE_LLM_MODELS.values() else 0
    )
    selected_model = AVAILABLE_LLM_MODELS[selected_model_name]
    
    # LLM Temperature
    st.sidebar.subheader("🔥 LLM Temperature")
    temperature = st.sidebar.slider(
        "Temperature (창의성)", min_value=0.0, max_value=1.0, value=st.session_state.get("llm_temperature", 0.1), step=0.05
    )
    st.session_state.llm_temperature = temperature
    st.sidebar.write(f"현재 값: {temperature}")
    
    # Vector Store Type
    st.sidebar.subheader("🗄️ 벡터 스토어 타입")
    vector_store_type = st.sidebar.radio(
        "Vector Store 엔진 선택:",
        options=["faiss", "chroma"],
        index=["faiss", "chroma"].index(st.session_state.get("vector_store_type", "faiss"))
    )
    st.session_state.vector_store_type = vector_store_type
    st.sidebar.write(f"현재 선택: {vector_store_type}")
    
    # Chunk Size
    st.sidebar.subheader("🔪 청크 크기 (Chunk Size)")
    chunk_size = st.sidebar.slider(
        "청크 크기:", min_value=256, max_value=4096, value=st.session_state.get("chunk_size", CHUNK_SIZE), step=64
    )
    st.session_state.chunk_size = chunk_size
    st.sidebar.write(f"현재 값: {chunk_size}")
    
    # Chunk Overlap
    st.sidebar.subheader("🔗 청크 오버랩 (Chunk Overlap)")
    chunk_overlap = st.sidebar.slider(
        "오버랩:", min_value=0, max_value=1024, value=st.session_state.get("chunk_overlap", CHUNK_OVERLAP), step=16
    )
    st.session_state.chunk_overlap = chunk_overlap
    st.sidebar.write(f"현재 값: {chunk_overlap}")
    
    # Top-K (검색 수)
    st.sidebar.subheader("🔍 검색 수 (Top-K)")
    top_k = st.sidebar.slider(
        "검색할 문서 수:", min_value=1, max_value=30, value=st.session_state.get("top_k", DEFAULT_K), step=1
    )
    st.session_state.top_k = top_k
    st.sidebar.write(f"현재 값: {top_k}")
    
    # Store selected model in session state
    if "selected_llm_model" not in st.session_state:
        st.session_state.selected_llm_model = selected_model
    elif st.session_state.selected_llm_model != selected_model:
        st.session_state.selected_llm_model = selected_model
        # Clear cached LLM when model changes
        if "llm_manager" in st.session_state:
            del st.session_state.llm_manager
    
    # LLM Status
    st.sidebar.subheader("🧠 LLM 상태")
    llm_manager = LLMManager(st.session_state.selected_llm_model, OLLAMA_BASE_URL, temperature=st.session_state.llm_temperature)
    llm_info = llm_manager.get_model_info()
    
    if llm_info["connection_status"]:
        st.sidebar.success("✅ Ollama 서버 연결됨")
        if llm_info["model_available"]:
            st.sidebar.success(f"✅ 모델 '{st.session_state.selected_llm_model}' 사용 가능")
        else:
            st.sidebar.error(f"❌ 모델 '{st.session_state.selected_llm_model}' 없음")
            if st.sidebar.button("모델 다운로드"):
                with st.spinner("모델 다운로드 중..."):
                    if llm_manager.pull_model():
                        st.sidebar.success("모델 다운로드 완료!")
                        st.rerun()
                    else:
                        st.sidebar.error("모델 다운로드 실패")
    else:
        st.sidebar.error("❌ Ollama 서버 연결 실패")
        st.sidebar.code(f"ollama serve")
    
    # Document Status
    st.sidebar.subheader("📚 문서 상태")
    if st.session_state.documents_loaded:
        st.sidebar.success(f"✅ {len(st.session_state.documents)}개 문서 로딩됨")
        st.sidebar.success(f"✅ {len(st.session_state.document_chunks)}개 청크 생성됨")
    else:
        st.sidebar.warning("⚠️ 문서가 로딩되지 않음")
    
    # Vector Store Status
    if st.session_state.vector_store_created:
        st.sidebar.success("✅ 벡터 스토어 생성됨")
    else:
        st.sidebar.warning("⚠️ 벡터 스토어 미생성")
    
    # Configuration
    st.sidebar.subheader("🔧 설정")
    st.sidebar.write(f"**임베딩 모델:** {EMBEDDING_MODEL}")
    st.sidebar.write(f"**청크 크기:** {chunk_size}")
    st.sidebar.write(f"**청크 오버랩:** {chunk_overlap}")
    st.sidebar.write(f"**기본 검색 수:** {top_k}")


def load_documents_tab():
    """Document loading tab."""
    st.header("📚 문서 로딩 및 전처리")
    
    # Document folder info
    st.info(f"문서 폴더: {DOCS_FOLDER}")
    
    if not DOCS_FOLDER.exists():
        st.error(f"문서 폴더가 존재하지 않습니다: {DOCS_FOLDER}")
        return
    
    # List available documents
    pdf_files = list(DOCS_FOLDER.glob("*.pdf"))
    st.write(f"사용 가능한 PDF 파일 ({len(pdf_files)}개):")
    for pdf_file in pdf_files:
        file_size = pdf_file.stat().st_size / (1024 * 1024)  # MB
        st.write(f"• {pdf_file.name} ({file_size:.1f} MB)")
    
    # Load documents button
    if st.button("📖 문서 로딩 시작", type="primary"):
        if not pdf_files:
            st.warning("PDF 파일이 없습니다.")
            return
        
        # Initialize document processor
        doc_processor = DocumentProcessor(st.session_state.chunk_size, st.session_state.chunk_overlap)
        
        # Load documents
        documents = doc_processor.load_documents_from_folder(DOCS_FOLDER)
        
        if documents:
            st.session_state.documents = documents
            
            # Display document statistics
            stats = doc_processor.get_document_stats(documents)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 문서 수", stats["total_documents"])
            with col2:
                st.metric("총 문자 수", f"{stats['total_characters']:,}")
            with col3:
                st.metric("평균 문자/문서", f"{stats['average_chars_per_doc']:,.0f}")
            
            # Split documents
            chunks = doc_processor.split_documents(documents)
            st.session_state.document_chunks = chunks
            st.session_state.documents_loaded = True
            
            st.success("✅ 문서 로딩 및 전처리 완료!")
        else:
            st.error("문서 로딩에 실패했습니다.")
    
    # Display loaded documents info
    if st.session_state.documents_loaded:
        st.subheader("📋 로딩된 문서 정보")
        
        # Document chunks preview
        with st.expander("문서 청크 미리보기"):
            if st.session_state.document_chunks:
                chunk = st.session_state.document_chunks[0]
                st.write(f"**첫 번째 청크 (ID: {chunk.metadata.get('chunk_id', 'N/A')}):**")
                st.write(f"**출처:** {chunk.metadata.get('source', 'Unknown')}")
                st.write(f"**크기:** {len(chunk.page_content)} 문자")
                st.write(f"**내용:**")
                st.write(chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content)


def create_vector_store_tab():
    """Vector store creation tab."""
    st.header("🔍 벡터 스토어 생성")
    
    if not st.session_state.documents_loaded:
        st.warning("먼저 문서를 로딩해주세요.")
        return
    
    st.info(f"임베딩 모델: {EMBEDDING_MODEL}")
    
    # Create vector store button
    if st.button("🚀 벡터 스토어 생성", type="primary"):
        try:
            # Initialize embedding manager with models folder
            embedding_manager = EmbeddingManager(EMBEDDING_MODEL, MODELS_FOLDER)
            embeddings = embedding_manager.get_embeddings()
            
            # Display embedding model info
            embed_info = embedding_manager.get_model_info()
            st.write("**임베딩 모델 정보:**")
            for key, value in embed_info.items():
                st.write(f"• {key}: {value}")
            
            # Create vector store (use selected type)
            vector_store_type = st.session_state.get("vector_store_type", "faiss")
            vector_store_manager = VectorStoreManager(embeddings, vector_store_type)
            vector_store = vector_store_manager.create_vector_store(st.session_state.document_chunks)
            
            # Store in session state
            st.session_state.vector_store_manager = vector_store_manager
            st.session_state.embedding_manager = embedding_manager
            st.session_state.vector_store_created = True
            
            # Display collection stats
            stats = vector_store_manager.get_collection_stats()
            st.write("**벡터 스토어 통계:**")
            for key, value in stats.items():
                st.write(f"• {key}: {value}")
            
            st.success("✅ 벡터 스토어 생성 완료!")
            
        except Exception as e:
            st.error(f"벡터 스토어 생성 실패: {str(e)}")
    
    # Test search functionality
    if st.session_state.vector_store_created:
        st.subheader("🔍 검색 테스트")
        test_query = st.text_input("테스트 검색어를 입력하세요:", placeholder="예: AI 트렌드")
        
        if test_query and st.button("검색 테스트"):
            vector_store_manager = st.session_state.vector_store_manager
            try:
                docs = vector_store_manager.similarity_search(test_query, k=st.session_state.top_k)
                st.write(f"**검색 결과 ({len(docs)}개):**")
                
                for i, doc in enumerate(docs):
                    with st.expander(f"문서 {i+1}: {doc.metadata.get('source', 'Unknown')}"):
                        st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        
            except Exception as e:
                st.error(f"검색 테스트 실패: {str(e)}")


def rag_experiment_tab():
    """RAG experiment tab."""
    st.header("🧪 RAG 시스템 실험")
    
    if not st.session_state.vector_store_created:
        st.warning("먼저 벡터 스토어를 생성해주세요.")
        return
    
    # Initialize RAG systems if not already done
    if not st.session_state.rag_systems:
        vector_store_manager = st.session_state.vector_store_manager
        selected_model = st.session_state.get("selected_llm_model", DEFAULT_LLM_MODEL)
        llm_temperature = st.session_state.get("llm_temperature", 0.1)
        llm_manager = LLMManager(selected_model, OLLAMA_BASE_URL, temperature=llm_temperature)
        
        st.session_state.rag_systems = {
            "Naive RAG": NaiveRAG(vector_store_manager, llm_manager),
            "Advanced RAG": AdvancedRAG(vector_store_manager, llm_manager),
            "Modular RAG": ModularRAG(vector_store_manager, llm_manager)
        }
    
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
    
    # Sample queries - 분류 테스트용 다양한 유형
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
    
    # Query input - 샘플 질문 선택 시 해당 질문이 입력창에 표시됨
    query = st.text_area(
        "질문을 입력하세요:",
        value=st.session_state.get("text_area_value", ""),
        placeholder="예: 2025년 AI 트렌드는 무엇인가요?",
        height=100,
        key="query_input"
    )
    
    # Run experiment
    if query and selected_systems and st.button("🚀 실험 실행", type="primary"):
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
                continue
            
            st.divider()
        
        # Store results
        if results:
            st.session_state.experiment_results = results
            st.success("✅ 모든 실험 완료!")


def comparison_tab():
    """Comparison and analysis tab."""
    st.header("📊 결과 비교 및 분석")
    
    if not st.session_state.experiment_results:
        st.warning("먼저 RAG 실험을 실행해주세요.")
        return
    
    results = st.session_state.experiment_results
    
    # System information comparison
    if st.session_state.rag_systems:
        systems_info = []
        for system_name, rag_system in st.session_state.rag_systems.items():
            systems_info.append(rag_system.get_system_info())
        
        ComparisonUI.display_system_comparison(systems_info)
        st.divider()
    
    # Performance comparison
    ComparisonUI.display_performance_comparison(results)
    st.divider()
    
    # Answer comparison
    ComparisonUI.display_answer_comparison(results)
    st.divider()
    
    # Detailed metrics for each system
    for result in results:
        ComparisonUI.display_detailed_metrics(result)
        ComparisonUI.create_processing_flow_diagram(result["rag_type"])
        st.divider()
    
    # Summary report
    ComparisonUI.create_summary_report(results)


def about_tab():
    """About and documentation tab."""
    st.header("ℹ️ RAG 시스템 소개")
    
    st.markdown("""
    ## 🎯 애플리케이션 목적
    
    이 애플리케이션은 세 가지 주요 RAG (Retrieval-Augmented Generation) 패러다임을 비교하고 실험할 수 있도록 설계되었습니다:
    
    ### 1. 📚 Naive RAG
    - **특징**: 가장 기본적인 RAG 구현
    - **구성**: 단순 벡터 유사도 검색 + 직접 생성
    - **장점**: 빠른 처리 속도, 간단한 구조
    - **단점**: 제한된 검색 품질, 컨텍스트 최적화 없음
    
    ### 2. 🔧 Advanced RAG
    - **특징**: 향상된 전처리 및 후처리 기법 적용
    - **구성**: 쿼리 최적화 + 문서 재순위화 + 컨텍스트 압축
    - **장점**: 높은 검색 정확도, 효율적인 컨텍스트 활용
    - **단점**: 복잡한 구조, 상대적으로 느린 처리
    
    ### 3. 🧩 Modular RAG
    - **특징**: 유연한 모듈 기반 아키텍처
    - **구성**: 독립적 모듈들의 조합, 반복적 개선
    - **장점**: 높은 확장성, 상황별 최적화, 투명한 처리 과정
    - **단점**: 복잡한 설계, 많은 계산 리소스 필요
    
    ## 🛠️ 기술 스택
    
    - **Frontend**: Streamlit
    - **LLM**: Ollama (Gemma 3 Models)
    - **Embeddings**: HuggingFace Multilingual E5
    - **Vector Store**: ChromaDB
    - **Framework**: LangChain, LangGraph
    - **Package Management**: pip/uv
    
    ## 📖 사용 방법
    
    1. **문서 로딩**: PDF 문서들을 로딩하고 청크로 분할
    2. **벡터 스토어 생성**: 문서 임베딩 생성 및 인덱싱
    3. **RAG 실험**: 다양한 RAG 시스템으로 질문 답변 테스트
    4. **결과 비교**: 성능, 정확도, 처리 시간 등을 비교 분석
    
    ## 🎨 UI 특징
    
    - 단계별 진행 상황 표시
    - 실시간 성능 메트릭
    - 시각적 비교 차트
    - 상세한 처리 과정 로그
    - 실시간 스트리밍 답변
    """)
    
    # Display system requirements
    st.subheader("⚙️ 시스템 요구사항")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**필수 요구사항:**")
        st.write("• Python 3.11+")
        st.write("• Ollama 설치 및 실행")
        st.write("• 4GB+ RAM")
        st.write("• 2GB+ 디스크 공간")
    
    with col2:
        st.write("**권장 사양:**")
        st.write("• 8GB+ RAM")
        st.write("• GPU 지원 (선택사항)")
        st.write("• SSD 저장장치")
        st.write("• 안정적인 인터넷 연결")


def main():
    """Main application function."""
    # Apply custom font globally first (before any other UI elements)
    apply_custom_css()
    
    # Setup
    setup_page()
    initialize_session_state()
    setup_sidebar()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 문서 로딩",
        "🔍 벡터 스토어",
        "🧪 RAG 실험",
        "📊 결과 비교",
        "ℹ️ 소개"
    ])
    
    with tab1:
        load_documents_tab()
    
    with tab2:
        create_vector_store_tab()
    
    with tab3:
        rag_experiment_tab()
    
    with tab4:
        comparison_tab()
    
    with tab5:
        about_tab()


if __name__ == "__main__":
    main()