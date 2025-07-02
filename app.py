"""Main Streamlit application for RAG systems comparison."""

import streamlit as st
import os
from pathlib import Path
import base64
from typing import Optional
import logging

# pypdf's warnings are very verbose. We can silence them by raising the log level.
logging.getLogger("pypdf._reader").setLevel(logging.ERROR)

# Disable ChromaDB telemetry at app startup to prevent telemetry errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["CHROMA_SERVER_NOFILE"] = "1"

# Import our modules
from src.config import *
from src.utils.embeddings import EmbeddingManager
from src.utils.vector_store import VectorStoreManager
from src.utils.llm_manager import LLMManager
from src.utils.font_utils import inject_custom_font
# from src.rag_systems.naive_rag import NaiveRAG # No longer needed
from src.ui.comparison_ui import ComparisonUI
from src.ui.about_ui import AboutUI
from src.ui.document_loading_ui import DocumentLoadingUI
from src.ui.vector_store_ui import VectorStoreUI
from src.ui.rag_experiment_ui import RAGExperimentUI
from src.ui.translation_ui import TranslationUI
from src.ui.json_services_ui import JSONServicesUI
from src.ui.report_generation_ui import ReportGenerationUI
from src.ui.document_discovery_ui import DocumentDiscoveryUI
from src.ui.web_search_ui import WebSearchUI


def get_or_create_vector_store_manager() -> Optional[VectorStoreManager]:
    """Get or create vector store manager with lazy loading.
    
    Returns:
        VectorStoreManager instance or None if creation fails
    """
    if "vector_store_manager" not in st.session_state:
        try:
            # Initialize embeddings lazily
            embedding_manager = EmbeddingManager(EMBEDDING_MODEL, MODELS_FOLDER)
            embeddings = embedding_manager.get_embeddings()
            
            # Initialize vector store manager
            st.session_state.vector_store_manager = VectorStoreManager(
                embeddings=embeddings,
                vector_store_type=st.session_state.get("vector_store_type", "faiss"),
                collection_name="rag_documents"
            )
            st.session_state.embedding_manager = embedding_manager
            
            # Initialize empty metadata if not exists
            if "vector_store_metadata" not in st.session_state:
                st.session_state.vector_store_metadata = {}
            if "vector_store_source" not in st.session_state:
                st.session_state.vector_store_source = "auto_created"
            if "vector_store_id" not in st.session_state:
                import time
                st.session_state.vector_store_id = f"auto_{int(time.time())}"
            
        except Exception as e:
            st.error(f"❌ 벡터 스토어 매니저 초기화 실패: {str(e)}")
            return None
    
    return st.session_state.get("vector_store_manager")


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
    # vector_store_manager는 지연 로딩으로 처리 (필요할 때만 생성)


def setup_page():
    """Setup page configuration."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom font globally first
    inject_custom_font("fonts/Paperlogy.ttf")
    
    # Display title with custom styling
    st.markdown('<h1 class="main-title">🤖 RAG Systems Comparison Tool</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">단계별 Naive RAG, Advanced RAG, Modular RAG 비교 실험 애플리케이션</p>', unsafe_allow_html=True)


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
        options=["faiss", "chroma", "milvus"],
        index=["faiss", "chroma", "milvus"].index(st.session_state.get("vector_store_type", "faiss"))
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


def main():
    """Main function to run the RAG application."""
    setup_page()
    initialize_session_state()
    setup_sidebar()
    
    # Main tabs
    tabs = st.tabs([
        "📚 문서 로딩", "🔍 벡터 스토어", "🧪 RAG 실험", "📊 결과 비교", "📋 보고서 생성", 
        "🌐 문서 번역", "🏢 정보 서비스", "🔍 문서 발견", "🌍 웹 검색 RAG", "ℹ️ 소개"
    ])
    
    tab_map = {
        tabs[0]: DocumentLoadingUI.display_document_loading_tab,
        tabs[1]: VectorStoreUI.display_vector_store_tab,
        tabs[2]: RAGExperimentUI.display_rag_experiment_tab,
        tabs[3]: ComparisonUI.display_comparison_tab,
        tabs[4]: ReportGenerationUI.display_report_generation_tab,
        tabs[5]: TranslationUI.display_translation_tab,
        tabs[6]: JSONServicesUI.display_json_services_tab,
        tabs[7]: DocumentDiscoveryUI.display_document_discovery_tab,
        tabs[8]: WebSearchUI.display_web_search_tab,
        tabs[9]: AboutUI.display_about_tab,
    }

    for tab, display_func in tab_map.items():
        with tab:
            display_func()


if __name__ == "__main__":
    main()