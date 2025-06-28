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
    
    이 애플리케이션은 **최신 RAG 기술을 활용한** 세 가지 차별화된 RAG (Retrieval-Augmented Generation) 시스템을 
    직접 비교하고 실험할 수 있는 **종합 벤치마킹 플랫폼**입니다.
    """)
    
    # Create detailed system comparison
    with st.expander("📚 **Naive RAG** - 기본형 시스템", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            ### 🔧 **구성**
            - **검색**: 단순 벡터 유사도 (Dense Retrieval)
            - **생성**: 직접 LLM 호출
            - **최적화**: 없음
            
            ### ✨ **특징**
            - ⚡ **초고속 처리**: 최소한의 오버헤드
            - 🎯 **단순함**: 이해하기 쉬운 구조
            - 📦 **경량화**: 최소 리소스 사용
            """)
        with col2:
            st.info("**⚡ 속도 우선**\n\n빠른 프로토타이핑이나 \n실시간 응답이 필요한 \n환경에 최적화")
    
    with st.expander("🔧 **Advanced RAG** - Enhanced Search 시스템", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            ### 🔧 **Enhanced Search 구성**
            - **1단계**: 🔍 쿼리 전처리 & 최적화
            - **2단계**: 🧠 벡터 유사도 검색 (확장 범위)
            - **3단계**: 🔀 **하이브리드 재순위화**
              - TF-IDF 점수 (70%) + 벡터 유사도 (30%)
            - **4단계**: 📦 **스마트 컨텍스트 압축**
              - 키워드 기반 중요도 스코어링
              - 동적 길이 조절
            - **5단계**: 🤖 **실시간 스트리밍 생성**
            
            ### ✨ **핵심 혁신**
            - 🔀 **하이브리드 재순위화**: 의미적 + 통계적 검색 결합
            - 📦 **지능형 압축**: 중요 정보 보존하며 토큰 효율화
            - ⚡ **스트리밍**: 실시간 답변 생성 및 표시
            """)
        with col2:
            st.success("**📦 균형 최적화**\n\n정확성과 효율성의 \n완벽한 밸런스를 \n추구하는 시스템")
    
    with st.expander("🧩 **Modular RAG** - 지능형 모듈러 시스템", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            ### 🧠 **Pre-Retrieval Modules**
            - **🎯 Query Classification**: 7가지 질문 유형 분류
              - factual(사실형), procedural(방법형), causal(원인형)
              - temporal(시간형), comparative(비교형), quantitative(수치형), general(일반형)
            - **🔍 Query Expansion**: 키워드 매핑 기반 확장
            
            ### 🔍 **Retrieval Modules**
            - **🧠 Semantic Retrieval**: Dense 벡터 검색
            - **🔤 BM25 Keyword Retrieval**: Sparse 키워드 검색
              - 독립적인 BM25 구현 (k1=1.5, b=0.75)
              - 한국어 토크나이징 지원
            
            ### 🔧 **Post-Retrieval Modules**
            - **📊 Relevance Filtering**: 관련도 기반 필터링
            - **🎲 Diversity Module**: 중복 제거 및 다양성 보장
            
            ### 🤖 **Generation Modules**
            - **🎯 Type-Specific Generation**: 질문 유형별 맞춤 프롬프트
            - **📊 Confidence Estimation**: 답변 신뢰도 평가
            
            ### 🔄 **Orchestration Modules**
            - **🛤️ Routing**: 질문 유형별 최적 처리 경로 선택
            - **🔄 Iteration Control**: 신뢰도 기반 반복 개선 (< 0.7시 재시도)
            """)
        with col2:
            st.warning("**🎯 정밀 최적화**\n\n복잡한 질문과 \n높은 정확성이 \n요구되는 고급 용도")
    
    st.markdown("---")
    
    # Technical innovations section
    st.markdown("""
    ## 🚀 **기술적 혁신사항**
    
    ### 🔀 **하이브리드 검색 (Advanced & Modular)**
    - **Dense + Sparse 결합**: 의미적 검색과 키워드 매칭의 시너지
    - **동적 가중치**: TF-IDF 70% + 벡터 유사도 30% 최적 비율
    - **BM25 완전 구현**: 외부 의존성 없는 순수 Python 구현
    
    ### 🎯 **지능형 질문 분류 (Modular)**
    - **7가지 질문 유형**: 세밀한 의도 파악 및 맞춤 처리
    - **동적 신뢰도 계산**: 매칭 키워드 패턴 분석
    - **시각화 지원**: 분류 과정의 투명한 표시
    
    ### 📦 **스마트 컨텍스트 관리 (Advanced)**
    - **키워드 기반 압축**: 중요 문장 자동 추출
    - **동적 길이 조절**: 질문 복잡도에 따른 컨텍스트 크기 최적화
    - **토큰 효율성**: 최대 50% 압축률로 비용 절감
    
    ### 🔄 **반복적 개선 (Modular)**
    - **신뢰도 임계값**: 0.7 미만시 자동 재시도
    - **매개변수 조정**: 검색 범위 동적 확대 (k → k+2)
    - **최대 반복 제한**: 무한 루프 방지
    """)
    
    st.markdown("---")
    
    # Enhanced tech stack
    st.markdown("""
    ## 🛠️ **고급 기술 스택**
    
    ### 🧠 **AI/ML 코어**
    - **LLM**: Ollama Gemma 3 (1B~27B, QAT 최적화)
    - **Embeddings**: Multilingual E5-Large-Instruct (다국어 지원)
    - **검색 알고리즘**: 
      - Dense Retrieval (FAISS/ChromaDB)
      - BM25 Sparse Retrieval (순수 구현)
      - TF-IDF 통계 분석
    
    ### 🔧 **프레임워크 & 도구**
    - **Frontend**: Streamlit (실시간 UI)
    - **Backend**: LangChain + 커스텀 모듈
    - **벡터 DB**: ChromaDB (영구 저장)
    - **시각화**: Plotly (인터랙티브 차트)
    - **폰트**: Paperlogy (커스텀 디자인)
    
    ### 📊 **성능 모니터링**
    - **실시간 메트릭**: 처리 시간, 검색 점수, 신뢰도
    - **상세 분석**: 검색 점수 분포, 압축률, 반복 횟수
    - **비교 시각화**: 시스템별 성능 벤치마킹
    """)
    
    st.markdown("---")
    
    # Usage guide
    st.markdown("""
    ## 📖 **단계별 사용 가이드**
    
    ### 🎯 **질문 유형별 최적 시스템 선택**
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("""
        **⚡ Naive RAG**
        
        ✅ **적합한 경우:**
        - 빠른 프로토타이핑
        - 실시간 응답 필요
        - 단순한 정보 조회
        - 리소스 제약 환경
        
        📝 **예시 질문:**
        - "AI란 무엇인가요?"
        - "현재 트렌드는?"
        """)
    
    with col2:
        st.success("""
        **📦 Advanced RAG**
        
        ✅ **적합한 경우:**
        - 정확성과 속도 균형
        - 중급 복잡도 질문
        - 효율적 토큰 사용
        - 비용 최적화 필요
        
        📝 **예시 질문:**
        - "AI가 비즈니스에 미치는 영향은?"
        - "머신러닝과 딥러닝의 차이점?"
        """)
    
    with col3:
        st.warning("""
        **🎯 Modular RAG**
        
        ✅ **적합한 경우:**
        - 복잡한 분석 질문
        - 최고 정확도 요구
        - 다양한 질문 유형
        - 상세한 프로세스 추적
        
        📝 **예시 질문:**
        - "AI 도입 전략을 단계별로 설명해주세요"
        - "2025년과 2024년 AI 트렌드를 비교 분석해주세요"
        """)
    
    st.markdown("""
    ### 🚀 **실험 진행 단계**
    
    1. **📚 문서 로딩** → PDF 문서 자동 파싱 및 청크 분할
    2. **🔍 벡터 스토어** → 임베딩 생성 및 인덱스 구축  
    3. **🧪 RAG 실험** → 시스템별 질문 처리 및 답변 생성
    4. **📊 결과 비교** → 성능 메트릭 분석 및 시각화
    
    ### 🎨 **고급 UI 기능**
    
    - **🎯 질문 유형 가이드**: 10가지 샘플 질문 (유형별 분류)
    - **📊 실시간 메트릭**: 처리 시간, 신뢰도, 검색 점수
    - **🔍 상세 분석**: 검색 과정 시각화, 점수 분포 차트
    - **⚡ 스트리밍**: Advanced RAG 실시간 답변 생성
    - **🔄 프로세스 추적**: Modular RAG 모듈별 처리 과정
    """)
    
    st.markdown("---")
    
    # System requirements
    st.subheader("⚙️ **시스템 요구사항**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 📋 **필수 요구사항**
        - **Python**: 3.11+ (Poetry 권장)
        - **Ollama**: 설치 및 실행 중
        - **RAM**: 4GB+ (BM25 인덱싱용)
        - **Storage**: 2GB+ (모델 및 벡터 저장)
        - **Models**: Gemma 3 다운로드 필요
        
        ### 🚀 **권장 사양 (고성능)**
        - **RAM**: 8GB+ (대용량 문서 처리)
        - **GPU**: CUDA 지원 (선택사항)
        - **SSD**: 빠른 벡터 검색
        - **Network**: 안정적 연결 (모델 다운로드)
        """)
    
    with col2:
        st.markdown("""
        ### 🔧 **설치 명령어**
        ```bash
        # Ollama 설치 (macOS)
        brew install ollama
        
        # Gemma 3 모델 다운로드
        ollama pull gemma3:4b-it-qat
        ollama pull gemma3:12b-it-qat
        
        # 애플리케이션 실행
        poetry install
        poetry run streamlit run app.py
        ```
        
        ### 📊 **성능 벤치마크**
        - **Naive RAG**: ~1-2초 (경량)
        - **Advanced RAG**: ~3-5초 (균형)  
        - **Modular RAG**: ~5-8초 (정밀)
        
        *Gemma 3 4B 모델 기준
        """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    ## 🎉 **이 애플리케이션의 가치**
    
    📚 **교육적 가치**: RAG 기술의 발전 과정을 단계적으로 체험  
    🔬 **연구 도구**: 다양한 RAG 접근법의 성능 비교 분석  
    ⚡ **실용성**: 실제 비즈니스 환경에서의 RAG 시스템 선택 가이드  
    🚀 **혁신성**: 최신 RAG 기술들의 실제 구현 및 벤치마킹  
    
    > **"단순한 벡터 검색부터 고도화된 모듈러 아키텍처까지,  
    > RAG 기술의 모든 것을 한 곳에서 경험해보세요!"**
    """)
    
    # Fun stats
    with st.expander("📊 **구현 통계**"):
        st.markdown("""
        - 🧩 **총 모듈 수**: 12개 (Modular RAG)
        - 🎯 **질문 분류**: 7가지 유형
        - 🔍 **검색 방법**: 3가지 (Dense, Sparse, Hybrid)
        - 📊 **성능 메트릭**: 15+ 지표
        - ⚡ **최적화 기법**: 5가지 (압축, 재순위화, 반복 등)
        - 🎨 **UI 컴포넌트**: 20+ 인터랙티브 요소
        """)
    
    st.balloons()  # 소개 탭 방문 축하! 🎉


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