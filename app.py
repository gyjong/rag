"""Main Streamlit application for RAG systems comparison."""

import streamlit as st
import os
from pathlib import Path
import base64
import pandas as pd
from datetime import datetime
import json
from typing import Optional

# Disable ChromaDB telemetry at app startup to prevent telemetry errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["CHROMA_SERVER_NOFILE"] = "1"

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
from src.ui.about_ui import AboutUI


def load_custom_font():
    """Load custom font if available."""
    return apply_custom_css()


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
    """Enhanced document loading tab with JSON functionality."""
    st.header("📚 문서 로딩 및 전처리")
    
    # Create JSON output folder if it doesn't exist
    JSON_OUTPUT_FOLDER.mkdir(exist_ok=True)
    
    # Data source selection
    st.subheader("🎯 데이터 소스 선택")
    
    tab1, tab2, tab3 = st.tabs(["📖 새 PDF 로딩", "📄 JSON 로딩", "🗂️ JSON 관리"])
    
    with tab1:
        st.write("### 📁 PDF 파일에서 새로 로딩")
        
        # Document folder info
        st.info(f"📂 문서 폴더: {DOCS_FOLDER}")
        
        if not DOCS_FOLDER.exists():
            st.error(f"문서 폴더가 존재하지 않습니다: {DOCS_FOLDER}")
            return
        
        # List available documents
        pdf_files = list(DOCS_FOLDER.glob("*.pdf"))
        st.write(f"📚 **사용 가능한 PDF 파일 ({len(pdf_files)}개):**")
        
        if pdf_files:
            # Display PDF files with details and selection
            pdf_data = []
            for pdf_file in pdf_files:
                file_size = pdf_file.stat().st_size / (1024 * 1024)  # MB
                pdf_data.append({
                    "파일명": pdf_file.name,
                    "크기 (MB)": f"{file_size:.1f}",
                    "수정일": datetime.fromtimestamp(pdf_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                })
            
            df_pdfs = pd.DataFrame(pdf_data)
            st.dataframe(df_pdfs, use_container_width=True)
            
            # File selection
            st.write("### 🎯 로딩할 파일 선택")
            
            # Select all option
            select_all = st.checkbox("📂 모든 파일 선택", value=True)
            
            # Individual file selection
            if select_all:
                selected_files = [f.name for f in pdf_files]
            else:
                selected_files = st.multiselect(
                    "로딩할 PDF 파일 선택:",
                    options=[f.name for f in pdf_files],
                    default=[]
                )
            
            # Display selected files info
            if selected_files:
                st.write(f"**선택된 파일:** {len(selected_files)}개")
                selected_size = 0
                for file_name in selected_files:
                    file_path = DOCS_FOLDER / file_name
                    selected_size += file_path.stat().st_size / (1024 * 1024)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📄 선택된 파일 수", len(selected_files))
                with col2:
                    st.metric("📊 총 크기", f"{selected_size:.1f} MB")
                with col3:
                    st.metric("📈 전체 대비", f"{len(selected_files)/len(pdf_files)*100:.0f}%")
                
                # Show selected files list
                with st.expander("📋 선택된 파일 목록"):
                    for file_name in selected_files:
                        st.write(f"• {file_name}")
            else:
                st.warning("⚠️ 로딩할 파일을 선택해주세요.")
            
            # JSON save options (only show if files are selected)
            if selected_files:
                st.write("### ⚙️ 처리 옵션")
                
                # Generate default filenames based on selected files
                if len(selected_files) == 1:
                    # Single file
                    file_base = selected_files[0].replace('.pdf', '')
                    default_docs_name = f"documents_{file_base}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                    default_chunks_name = f"chunks_{file_base}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                else:
                    # Multiple files
                    default_docs_name = f"documents_{len(selected_files)}files_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                    default_chunks_name = f"chunks_{len(selected_files)}files_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                
                col1, col2 = st.columns(2)
                
                with col1:
                    save_docs_json = st.checkbox("📄 원본 문서를 JSON으로 저장", value=True)
                    if save_docs_json:
                        docs_json_name = st.text_input(
                            "원본 JSON 파일명:", 
                            value=default_docs_name
                        )
                
                with col2:
                    save_chunks_json = st.checkbox("🧩 청크를 JSON으로 저장", value=True)
                    if save_chunks_json:
                        chunks_json_name = st.text_input(
                            "청크 JSON 파일명:", 
                            value=default_chunks_name
                        )
            
            # Load documents button
            load_disabled = not selected_files if 'selected_files' in locals() else True
            
            if st.button("🚀 PDF 문서 로딩 시작", type="primary", disabled=load_disabled):
                if not selected_files:
                    st.warning("⚠️ 로딩할 파일을 선택해주세요.")
                    return
                
                # Initialize document processor
                doc_processor = DocumentProcessor(st.session_state.chunk_size, st.session_state.chunk_overlap)
                
                # Create temporary folder with selected files only
                selected_file_paths = [DOCS_FOLDER / file_name for file_name in selected_files]
                
                # Load only selected documents
                st.write(f"📖 **{len(selected_files)}개 파일 로딩 시작**")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                documents = []
                loaded_count = 0
                
                for i, file_path in enumerate(selected_file_paths):
                    try:
                        status_text.text(f"📄 로딩 중: {file_path.name} ({i+1}/{len(selected_files)})")
                        file_docs = doc_processor.load_documents_from_file(file_path)
                        documents.extend(file_docs)
                        loaded_count += 1
                        
                        # Update progress
                        progress = (i + 1) / len(selected_files)
                        progress_bar.progress(progress)
                        
                        st.write(f"✅ {file_path.name} 로딩 완료 ({len(file_docs)}개 페이지)")
                        
                    except Exception as e:
                        st.error(f"❌ {file_path.name} 로딩 실패: {str(e)}")
                        continue
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.write(f"📊 **로딩 완료: {loaded_count}/{len(selected_files)}개 파일**")
                
                if documents:
                    st.session_state.documents = documents
                    
                    # Save documents to JSON if requested
                    if save_docs_json:
                        docs_json_path = JSON_OUTPUT_FOLDER / docs_json_name
                        doc_processor.save_documents_to_json(documents, docs_json_path)
                    
                    # Display document statistics
                    stats = doc_processor.get_document_stats(documents)
                    
                    st.write("### 📊 로딩 결과 통계")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📄 총 페이지 수", stats["total_documents"])
                    with col2:
                        st.metric("📝 총 문자 수", f"{stats['total_characters']:,}")
                    with col3:
                        st.metric("📏 평균 문자/페이지", f"{stats['average_chars_per_doc']:,.0f}")
                    with col4:
                        st.metric("📁 로딩된 파일", len(selected_files))
                    
                    # Display loaded files details
                    st.write("### 📋 로딩된 파일 상세")
                    loaded_files_data = []
                    for file_name in selected_files:
                        file_docs = [doc for doc in documents if doc.metadata.get('source') == file_name]
                        if file_docs:
                            file_chars = sum(len(doc.page_content) for doc in file_docs)
                            loaded_files_data.append({
                                "파일명": file_name,
                                "페이지 수": len(file_docs),
                                "문자 수": f"{file_chars:,}",
                                "평균 문자/페이지": f"{file_chars/len(file_docs):,.0f}" if file_docs else "0"
                            })
                    
                    if loaded_files_data:
                        df_loaded = pd.DataFrame(loaded_files_data)
                        st.dataframe(df_loaded, use_container_width=True)
                    
                    # Split documents
                    with st.spinner("🧩 문서 청크 분할 중..."):
                        chunks = doc_processor.split_documents(documents)
                    
                    st.session_state.document_chunks = chunks
                    st.session_state.documents_loaded = True
                    
                    # Save chunks to JSON if requested
                    if save_chunks_json:
                        chunks_json_path = JSON_OUTPUT_FOLDER / chunks_json_name
                        doc_processor.save_chunks_to_json(chunks, chunks_json_path)
                    
                    # Display chunk statistics
                    st.write("### 🧩 청크 분할 결과")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🧩 총 청크 수", len(chunks))
                    with col2:
                        avg_chunk_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
                        st.metric("📏 평균 청크 크기", f"{avg_chunk_size:.0f}")
                    with col3:
                        chunks_per_file = len(chunks) / len(selected_files)
                        st.metric("📊 파일당 청크", f"{chunks_per_file:.1f}")
                    with col4:
                        st.metric("⚙️ 청크 설정", f"{st.session_state.chunk_size}/{st.session_state.chunk_overlap}")
                    
                    st.success(f"✅ {len(selected_files)}개 파일의 문서 로딩 및 전처리 완료! ({len(chunks)}개 청크 생성)")
                else:
                    st.error("❌ 문서 로딩에 실패했습니다.")
        else:
            st.warning("📭 PDF 파일이 없습니다.")
    
    with tab2:
        st.write("### 📄 기존 JSON 파일에서 로딩")
        
        # List available JSON files
        json_files = list(JSON_OUTPUT_FOLDER.glob("*.json"))
        
        if json_files:
            st.write(f"📁 **사용 가능한 JSON 파일 ({len(json_files)}개):**")
            
            # JSON file selection
            selected_json = st.selectbox(
                "로딩할 JSON 파일 선택:",
                options=[f.name for f in json_files],
                key="json_file_selector"
            )
            
            if selected_json:
                json_path = JSON_OUTPUT_FOLDER / selected_json
                doc_processor = DocumentProcessor(st.session_state.chunk_size, st.session_state.chunk_overlap)
                
                # Get JSON file info
                json_info = doc_processor.get_json_info(json_path)
                if json_info:
                    st.write("### 📋 JSON 파일 정보")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📁 파일 크기", f"{json_info['file_size_mb']:.1f} MB")
                    with col2:
                        st.metric("📄 데이터 유형", json_info['data_type'])
                    with col3:
                        st.metric("📊 항목 수", json_info['total_items'])
                    
                    # Additional info
                    with st.expander("🔍 상세 정보"):
                        st.write(f"**생성 시간:** {json_info['created_at']}")
                        st.write(f"**총 문자 수:** {json_info['total_characters']:,}")
                        if json_info['chunk_config']:
                            st.write(f"**청크 설정:** {json_info['chunk_config']}")
                        st.write("**샘플 내용:**")
                        st.text(json_info['sample_content'])
                
                # Load JSON button
                if st.button("📥 JSON 파일 로딩", type="primary"):
                    try:
                        if json_info['data_type'] == 'documents':
                            # Load documents and split into chunks
                            documents = doc_processor.load_documents_from_json(json_path)
                            if documents:
                                st.session_state.documents = documents
                                
                                # Split documents
                                chunks = doc_processor.split_documents(documents)
                                st.session_state.document_chunks = chunks
                                st.session_state.documents_loaded = True
                                
                                st.success(f"✅ {len(documents)}개 문서를 {len(chunks)}개 청크로 로딩했습니다!")
                        
                        elif json_info['data_type'] == 'chunks':
                            # Load chunks directly
                            chunks = doc_processor.load_chunks_from_json(json_path)
                            if chunks:
                                st.session_state.document_chunks = chunks
                                st.session_state.documents_loaded = True
                                
                                st.success(f"✅ {len(chunks)}개 청크를 직접 로딩했습니다!")
                    
                    except Exception as e:
                        st.error(f"❌ JSON 로딩 실패: {str(e)}")
        else:
            st.info("📭 저장된 JSON 파일이 없습니다. 먼저 PDF를 로딩하여 JSON으로 저장해주세요.")
    
    with tab3:
        st.write("### 🗂️ JSON 파일 관리")
        
        json_files = list(JSON_OUTPUT_FOLDER.glob("*.json"))
        
        if json_files:
            st.write(f"📁 **JSON 파일 목록 ({len(json_files)}개):**")
            
            # Create detailed JSON file list
            json_data = []
            doc_processor = DocumentProcessor(st.session_state.chunk_size, st.session_state.chunk_overlap)
            
            for json_file in json_files:
                json_info = doc_processor.get_json_info(json_file)
                if json_info:
                    json_data.append({
                        "파일명": json_file.name,
                        "유형": json_info['data_type'],
                        "항목 수": json_info['total_items'],
                        "크기 (MB)": json_info['file_size_mb'],
                        "생성일": json_info['created_at'][:10]  # Date only
                    })
            
            if json_data:
                df_json = pd.DataFrame(json_data)
                st.dataframe(df_json, use_container_width=True)
                
                # JSON file operations
                st.write("### 🔧 파일 작업")
                selected_files = st.multiselect(
                    "작업할 파일 선택:",
                    options=[f.name for f in json_files]
                )
                
                if selected_files:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("👁️ 선택 파일 미리보기"):
                            for file_name in selected_files:
                                file_path = JSON_OUTPUT_FOLDER / file_name
                                json_info = doc_processor.get_json_info(file_path)
                                if json_info:
                                    with st.expander(f"📄 {file_name}"):
                                        st.write(f"**유형:** {json_info['data_type']}")
                                        st.write(f"**항목 수:** {json_info['total_items']}")
                                        st.write(f"**샘플 내용:**")
                                        st.text(json_info['sample_content'])
                    
                    with col2:
                        if st.button("🗑️ 선택 파일 삭제", type="secondary"):
                            for file_name in selected_files:
                                file_path = JSON_OUTPUT_FOLDER / file_name
                                try:
                                    file_path.unlink()
                                    st.success(f"✅ {file_name} 삭제 완료")
                                except Exception as e:
                                    st.error(f"❌ {file_name} 삭제 실패: {str(e)}")
                            st.rerun()
        else:
            st.info("📭 관리할 JSON 파일이 없습니다.")
    
    # Display loaded documents info
    if st.session_state.documents_loaded:
        st.markdown("---")
        st.subheader("📋 현재 로딩된 데이터")
        
        if st.session_state.document_chunks:
            chunks = st.session_state.document_chunks
            
            # Display summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🧩 총 청크", len(chunks))
            with col2:
                avg_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
                st.metric("📏 평균 크기", f"{avg_size:.0f}")
            with col3:
                sources = set(chunk.metadata.get('source', 'Unknown') for chunk in chunks)
                st.metric("📚 소스 수", len(sources))
            with col4:
                total_chars = sum(len(chunk.page_content) for chunk in chunks)
                st.metric("📝 총 문자", f"{total_chars:,}")
            
            # Document chunks preview
            with st.expander("🔍 청크 미리보기 및 탐색"):
                # Chunk navigation
                col1, col2 = st.columns([3, 1])
                with col1:
                    chunk_index = st.slider("청크 선택:", 0, len(chunks)-1, 0)
                with col2:
                    show_metadata = st.checkbox("메타데이터 표시", value=False)
                
                chunk = chunks[chunk_index]
                
                # Display chunk info
                st.write(f"### 📄 청크 {chunk_index + 1}/{len(chunks)}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**📂 출처:** {chunk.metadata.get('source', 'Unknown')}")
                with col2:
                    st.write(f"**📏 크기:** {len(chunk.page_content)} 문자")
                with col3:
                    st.write(f"**📄 페이지:** {chunk.metadata.get('page_number', 'N/A')}")
                
                if show_metadata:
                    with st.expander("🏷️ 전체 메타데이터"):
                        st.json(chunk.metadata)
                
                # Display content
                st.write("**📖 내용:**")
                st.text_area(
                    label="청크 내용",
                    value=chunk.page_content,
                    height=300,
                    disabled=True,
                    label_visibility="hidden",
                    key=f"chunk_content_{chunk_index}"
                )


def create_vector_store_tab():
    """Enhanced vector store tab with creation/loading/management functionality."""
    st.header("🔍 벡터 스토어 생성 및 관리")
    
    # Create vector stores output folder if it doesn't exist
    from src.config import VECTOR_STORES_FOLDER
    VECTOR_STORES_FOLDER.mkdir(exist_ok=True)
    
    # Vector store source selection
    st.subheader("🎯 벡터 스토어 소스 선택")
    
    tab1, tab2, tab3 = st.tabs(["🚀 새 벡터 스토어 생성", "📥 기존 벡터 스토어 로딩", "🗂️ 벡터 스토어 관리"])
    
    with tab1:
        st.write("### 🔍 현재 로딩된 문서에서 새 벡터 스토어 생성")
        
        if not st.session_state.documents_loaded:
            st.warning("⚠️ 먼저 문서를 로딩해주세요.")
            st.info("**📚 문서 로딩** 탭에서 PDF를 로딩하거나 JSON에서 문서를 불러와주세요.")
        else:
            # Display current document info
            chunks = st.session_state.document_chunks
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🧩 총 청크", len(chunks))
            with col2:
                avg_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
                st.metric("📏 평균 크기", f"{avg_size:.0f}")
            with col3:
                sources = set(chunk.metadata.get('source', 'Unknown') for chunk in chunks)
                st.metric("📚 소스 수", len(sources))
            with col4:
                total_chars = sum(len(chunk.page_content) for chunk in chunks)
                st.metric("📝 총 문자", f"{total_chars:,}")
            
            st.info(f"🤖 임베딩 모델: {EMBEDDING_MODEL}")
            
            # Vector store creation options
            st.write("### ⚙️ 벡터 스토어 설정")
            
            col1, col2 = st.columns(2)
            with col1:
                # Vector store type (from sidebar setting)
                vector_store_type = st.session_state.get("vector_store_type", "chroma")
                st.write(f"**벡터 스토어 타입:** {vector_store_type.upper()}")
                st.write(f"**컬렉션 이름:** {COLLECTION_NAME}")
                
            with col2:
                # Save options
                save_vector_store = st.checkbox("💾 벡터 스토어 저장", value=True)
                if save_vector_store:
                    store_name = st.text_input(
                        "벡터 스토어 이름:", 
                        value=f"vectorstore_{vector_store_type}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                    )
            
            # Create vector store button
            if st.button("🚀 벡터 스토어 생성 시작", type="primary"):
                try:
                    # Initialize embedding manager with models folder
                    embedding_manager = EmbeddingManager(EMBEDDING_MODEL, MODELS_FOLDER)
                    embeddings = embedding_manager.get_embeddings()
                    
                    # Display embedding model info
                    embed_info = embedding_manager.get_model_info()
                    with st.expander("🤖 임베딩 모델 정보"):
                        for key, value in embed_info.items():
                            st.write(f"• **{key}**: {value}")
                    
                    # Create vector store manager
                    vector_store_manager = VectorStoreManager(
                        embeddings, 
                        vector_store_type=vector_store_type,
                        collection_name=COLLECTION_NAME
                    )
                    
                    # Create vector store
                    vector_store = vector_store_manager.create_vector_store(chunks)
                    
                    # Store in session state
                    st.session_state.vector_store_manager = vector_store_manager
                    st.session_state.embedding_manager = embedding_manager
                    st.session_state.vector_store_created = True
                    
                    # Display collection stats
                    stats = vector_store_manager.get_collection_stats()
                    st.write("### 📊 벡터 스토어 통계")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📄 문서 수", stats.get("document_count", "N/A"))
                    with col2:
                        st.metric("🔧 상태", stats.get("status", "N/A"))
                    with col3:
                        st.metric("🚫 텔레메트리", stats.get("telemetry_status", "N/A"))
                    
                    # Save vector store if requested
                    if save_vector_store and store_name:
                        store_path = VECTOR_STORES_FOLDER / store_name
                        
                        # Prepare metadata
                        metadata = {
                            "document_count": len(chunks),
                            "total_characters": total_chars,
                            "source_count": len(sources),
                            "avg_chunk_size": avg_size,
                            "embedding_model": EMBEDDING_MODEL,
                            "chunk_size": st.session_state.get("chunk_size", CHUNK_SIZE),
                            "chunk_overlap": st.session_state.get("chunk_overlap", CHUNK_OVERLAP)
                        }
                        
                        success = vector_store_manager.save_vector_store(store_path, metadata)
                        if success:
                            st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ 벡터 스토어 생성 실패: {str(e)}")
    
    with tab2:
        st.write("### 📥 기존 벡터 스토어에서 로딩")
        
        # Debug: Show folder path and check existence
        st.write(f"**🔍 디버그 정보:**")
        st.write(f"- 벡터 스토어 폴더: `{VECTOR_STORES_FOLDER}`")
        st.write(f"- 폴더 존재 여부: {VECTOR_STORES_FOLDER.exists()}")
        
        if VECTOR_STORES_FOLDER.exists():
            all_items = list(VECTOR_STORES_FOLDER.iterdir())
            st.write(f"- 폴더 내 항목 수: {len(all_items)}")
            for item in all_items:
                st.write(f"  - {'[DIR]' if item.is_dir() else '[FILE]'} {item.name}")
        
        # List available vector stores
        saved_stores = VectorStoreManager.list_saved_vector_stores(VECTOR_STORES_FOLDER)
        st.write(f"- **감지된 벡터 스토어 수: {len(saved_stores)}**")
        
        # Debug: Show individual store info
        if VECTOR_STORES_FOLDER.exists():
            st.write("**📋 개별 스토어 분석:**")
            for item in VECTOR_STORES_FOLDER.iterdir():
                if item.is_dir():
                    metadata_path = item / "metadata.json"
                    st.write(f"  - **{item.name}**:")
                    st.write(f"    - metadata.json 존재: {metadata_path.exists()}")
                    if metadata_path.exists():
                        try:
                            import json
                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            st.write(f"    - JSON 파싱: ✅ 성공")
                            st.write(f"    - 문서 수: {metadata.get('document_count', 'N/A')}")
                            st.write(f"    - 타입: {metadata.get('vector_store_type', 'N/A')}")
                        except Exception as e:
                            st.write(f"    - JSON 파싱: ❌ 실패 - {str(e)}")
        
        if saved_stores:
            st.write(f"📁 **사용 가능한 벡터 스토어 ({len(saved_stores)}개):**")
            
            # Vector store selection
            store_options = [f"{store['store_name']} ({store.get('vector_store_type', 'unknown').upper()})" for store in saved_stores]
            selected_store_idx = st.selectbox(
                "로딩할 벡터 스토어 선택:",
                options=range(len(store_options)),
                format_func=lambda x: store_options[x],
                key="vector_store_selector"
            )
            
            if selected_store_idx is not None:
                selected_store = saved_stores[selected_store_idx]
                store_path = Path(selected_store["store_path"])
                
                # Display vector store info
                st.write("### 📋 벡터 스토어 정보")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📁 파일 크기", f"{selected_store.get('file_size_mb', 0):.1f} MB")
                with col2:
                    st.metric("🔧 타입", selected_store.get('vector_store_type', 'unknown').upper())
                with col3:
                    st.metric("📄 문서 수", selected_store.get('document_count', 'N/A'))
                
                # Additional info
                with st.expander("🔍 상세 정보"):
                    st.write(f"**생성 시간:** {selected_store.get('created_at', 'N/A')[:19]}")
                    st.write(f"**컬렉션 이름:** {selected_store.get('collection_name', 'N/A')}")
                    st.write(f"**임베딩 모델:** {selected_store.get('embedding_model', 'N/A')}")
                    if selected_store.get('total_characters'):
                        st.write(f"**총 문자 수:** {selected_store['total_characters']:,}")
                    if selected_store.get('avg_chunk_size'):
                        st.write(f"**평균 청크 크기:** {selected_store['avg_chunk_size']:.0f}")
                    if selected_store.get('chunk_size'):
                        st.write(f"**청크 설정:** {selected_store['chunk_size']}/{selected_store.get('chunk_overlap', 0)}")
            
                # Load vector store button
                if st.button("📥 벡터 스토어 로딩", type="primary"):
                    try:
                        # Initialize embedding manager
                        embedding_manager = EmbeddingManager(EMBEDDING_MODEL, MODELS_FOLDER)
                        embeddings = embedding_manager.get_embeddings()
                        
                        # Create vector store manager and load
                        vector_store_manager = VectorStoreManager(
                            embeddings,
                            vector_store_type=selected_store.get('vector_store_type', 'chroma'),
                            collection_name=selected_store.get('collection_name', COLLECTION_NAME)
                        )
                        
                        success = vector_store_manager.load_vector_store(store_path)
                        if success:
                            # Store in session state
                            st.session_state.vector_store_manager = vector_store_manager
                            st.session_state.embedding_manager = embedding_manager
                            st.session_state.vector_store_created = True
                            
                            # Show loaded stats
                            stats = vector_store_manager.get_collection_stats()
                            st.write("### ✅ 로딩 완료")
                            for key, value in stats.items():
                                st.write(f"• **{key}**: {value}")
                            
                            st.balloons()
                    
                    except Exception as e:
                        st.error(f"❌ 벡터 스토어 로딩 실패: {str(e)}")
        else:
            st.info("📭 저장된 벡터 스토어가 없습니다. 먼저 새 벡터 스토어를 생성해주세요.")
            
            # Manual path option as fallback
            st.write("**🔧 수동 로딩 (고급 사용자용):**")
            manual_path = st.text_input(
                "벡터 스토어 폴더 경로를 직접 입력:",
                placeholder="예: /Users/kenny/GitHub/rag/vector_stores/vectorstore_chroma_20250628_1832"
            )
            
            if manual_path and st.button("🔍 수동 경로에서 로딩"):
                manual_store_path = Path(manual_path)
                if manual_store_path.exists() and manual_store_path.is_dir():
                    metadata_path = manual_store_path / "metadata.json"
                    if metadata_path.exists():
                        try:
                            # Load metadata
                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            
                            st.success(f"✅ 메타데이터 발견: {metadata.get('vector_store_type', 'unknown')} 타입")
                            
                            # Initialize and load
                            embedding_manager = EmbeddingManager(EMBEDDING_MODEL, MODELS_FOLDER)
                            embeddings = embedding_manager.get_embeddings()
                            
                            vector_store_manager = VectorStoreManager(
                                embeddings,
                                vector_store_type=metadata.get('vector_store_type', 'chroma'),
                                collection_name=metadata.get('collection_name', COLLECTION_NAME)
                            )
                            
                            success = vector_store_manager.load_vector_store(manual_store_path)
                            if success:
                                st.session_state.vector_store_manager = vector_store_manager
                                st.session_state.embedding_manager = embedding_manager
                                st.session_state.vector_store_created = True
                                st.balloons()
                                
                        except Exception as e:
                            st.error(f"❌ 수동 로딩 실패: {str(e)}")
                    else:
                        st.error("❌ metadata.json 파일을 찾을 수 없습니다.")
                else:
                    st.error("❌ 지정된 경로가 존재하지 않습니다.")
    
    with tab3:
        st.write("### 🗂️ 벡터 스토어 파일 관리")
        
        saved_stores = VectorStoreManager.list_saved_vector_stores(VECTOR_STORES_FOLDER)
        
        if saved_stores:
            st.write(f"📁 **벡터 스토어 목록 ({len(saved_stores)}개):**")
            
            # Create detailed vector store list
            store_data = []
            for store in saved_stores:
                store_data.append({
                    "이름": store['store_name'],
                    "타입": store.get('vector_store_type', 'unknown').upper(),
                    "문서 수": store.get('document_count', 'N/A'),
                    "크기 (MB)": store.get('file_size_mb', 0),
                    "생성일": store.get('created_at', 'N/A')[:10]  # Date only
                })
            
            if store_data:
                df_stores = pd.DataFrame(store_data)
                st.dataframe(df_stores, use_container_width=True)
                
                # Vector store operations
                st.write("### 🔧 파일 작업")
                selected_stores = st.multiselect(
                    "작업할 벡터 스토어 선택:",
                    options=[store['store_name'] for store in saved_stores]
                )
                
                if selected_stores:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("👁️ 선택 스토어 미리보기"):
                            for store_name in selected_stores:
                                # Find store info
                                store_info = next((s for s in saved_stores if s['store_name'] == store_name), None)
                                if store_info:
                                    with st.expander(f"🔍 {store_name}"):
                                        st.write(f"**타입:** {store_info.get('vector_store_type', 'unknown').upper()}")
                                        st.write(f"**문서 수:** {store_info.get('document_count', 'N/A')}")
                                        st.write(f"**컬렉션:** {store_info.get('collection_name', 'N/A')}")
                                        st.write(f"**임베딩 모델:** {store_info.get('embedding_model', 'N/A')}")
                                        st.write(f"**크기:** {store_info.get('file_size_mb', 0):.1f} MB")
                    
                    with col2:
                        if st.button("🗑️ 선택 스토어 삭제", type="secondary"):
                            for store_name in selected_stores:
                                store_path = VECTOR_STORES_FOLDER / store_name
                                success = VectorStoreManager.delete_vector_store(store_path)
                                if success:
                                    st.success(f"✅ {store_name} 삭제 완료")
                                else:
                                    st.error(f"❌ {store_name} 삭제 실패")
                            st.rerun()
        else:
            st.info("📭 관리할 벡터 스토어가 없습니다.")
    
    # Test search functionality (if vector store is loaded)
    if st.session_state.vector_store_created:
        st.markdown("---")
        st.subheader("🔍 벡터 스토어 검색 테스트")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            test_query = st.text_input("테스트 검색어를 입력하세요:", placeholder="예: AI 트렌드")
        with col2:
            test_k = st.slider("검색 문서 수:", 1, 10, st.session_state.get("top_k", DEFAULT_K))
        
        if test_query and st.button("🔍 검색 테스트"):
            vector_store_manager = st.session_state.vector_store_manager
            try:
                docs_with_score = vector_store_manager.similarity_search_with_score(test_query, k=test_k)
                
                st.write(f"### 📊 검색 결과 ({len(docs_with_score)}개)")
                
                for i, (doc, score) in enumerate(docs_with_score):
                    with st.expander(f"📄 문서 {i+1}: {doc.metadata.get('source', 'Unknown')} (점수: {score:.3f})"):
                        st.write(f"**유사도 점수:** {score:.4f}")
                        st.write(f"**출처:** {doc.metadata.get('source', 'Unknown')}")
                        st.write(f"**페이지:** {doc.metadata.get('page_number', 'N/A')}")
                        st.write("**내용:**")
                        content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                        st.text(content_preview)
                        
            except Exception as e:
                st.error(f"❌ 검색 테스트 실패: {str(e)}")
        
        # Current vector store status
        with st.expander("📊 현재 벡터 스토어 상태"):
            vector_store_manager = st.session_state.vector_store_manager
            stats = vector_store_manager.get_collection_stats()
            for key, value in stats.items():
                st.write(f"• **{key}**: {value}")


def rag_experiment_tab():
    """RAG experiment tab with various systems."""
    st.header("🧪 RAG 시스템 실험")
    
    # First check if we have a vector store manager with actual vector store
    vector_store_manager = st.session_state.get("vector_store_manager")
    vector_store = None
    
    if vector_store_manager:
        try:
            vector_store = vector_store_manager.get_vector_store()
        except Exception as e:
            st.warning(f"⚠️ 기존 벡터 스토어 확인 실패: {str(e)}")
            vector_store = None
    
    # If no vector store exists, try to create manager (but don't force it)
    if vector_store is None:
        st.warning("📋 벡터 스토어가 필요합니다.")
        st.info("**다음 중 하나를 수행하세요:**")
        st.markdown("""
        1. **📚 문서 로딩** 탭에서 문서를 로드한 후 **🔍 벡터 스토어** 탭에서 새 벡터 스토어 생성
        2. **🔍 벡터 스토어** 탭에서 기존에 저장된 벡터 스토어 로딩
        """)
        return
    
    # Display vector store info
    st.success("✅ 벡터 스토어 준비 완료!")
    
    # Show current vector store status
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
    
    # Initialize RAG systems if not already done
    if not st.session_state.rag_systems:
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
                import traceback
                st.error(f"상세 오류 정보: {traceback.format_exc()}")
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
    AboutUI.display_about_page()


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