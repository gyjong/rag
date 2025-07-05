"""Document loading UI module for the RAG application."""

import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path

from ..config import *
from ..utils.document_processor import DocumentProcessor


class DocumentLoadingUI:
    """UI components for document loading and management."""
    
    @staticmethod
    def display_document_loading_tab():
        """Enhanced document loading tab with JSON functionality."""
        st.header("📚 문서 로딩 및 전처리")
        
        # Create JSON output folder if it doesn't exist
        JSON_OUTPUT_FOLDER.mkdir(exist_ok=True)
        
        # Data source selection
        st.subheader("🎯 데이터 소스 선택")
        
        tab1, tab2, tab3 = st.tabs(["📖 새 PDF 로딩", "📄 JSON 로딩", "🗂️ JSON 관리"])
        
        with tab1:
            DocumentLoadingUI._display_pdf_loading_tab()
        
        with tab2:
            DocumentLoadingUI._display_json_loading_tab()
        
        with tab3:
            DocumentLoadingUI._display_json_management_tab()
        
        # Display loaded documents info
        DocumentLoadingUI._display_current_documents_info()
    
    @staticmethod
    def _display_pdf_loading_tab():
        """Display PDF loading tab."""
        st.write("### 📁 PDF 파일에서 새로 로딩")
        
        # Document folder info
        st.info(f"📂 문서 폴더: {DOCS_FOLDER}")
        
        if not DOCS_FOLDER.exists():
            st.error(f"문서 폴더가 존재하지 않습니다: {DOCS_FOLDER}")
            return
        
        # List available documents
        pdf_files = sorted(list(DOCS_FOLDER.glob("*.pdf")), key=lambda x: x.name.lower())
        
        
        # Display sorting options
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"📚 **사용 가능한 PDF 파일 ({len(pdf_files)}개):** (파일명 순)")
        with col2:
            sort_option = st.selectbox(
                "정렬 기준:",
                ["파일명 순", "크기 순", "수정일 순"],
                key="pdf_sort_option"
            )
        
        # Apply sorting based on selection
        if sort_option == "파일명 순":
            pdf_files = sorted(pdf_files, key=lambda x: x.name.lower())
        elif sort_option == "크기 순":
            pdf_files = sorted(pdf_files, key=lambda x: x.stat().st_size, reverse=True)
        elif sort_option == "수정일 순":
            pdf_files = sorted(pdf_files, key=lambda x: x.stat().st_mtime, reverse=True)
        
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
            selected_files = DocumentLoadingUI._display_file_selection(pdf_files)
            
            # JSON save options and loading
            if selected_files:
                DocumentLoadingUI._display_pdf_processing_options(selected_files, pdf_files)
        else:
            st.warning("📭 PDF 파일이 없습니다.")
    
    @staticmethod
    def _display_file_selection(pdf_files):
        """Display file selection interface."""
        st.write("### 🎯 로딩할 파일 선택")
        
        # Select all option
        select_all = st.checkbox("📂 모든 파일 선택", value=True)
        
        # Individual file selection
        if select_all:
            selected_files = [f.name for f in pdf_files]
        else:
            selected_files = st.multiselect(
                "로딩할 PDF 파일 선택: (위에서 선택한 정렬 기준으로 표시)",
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
        
        return selected_files
    
    @staticmethod
    def _display_pdf_processing_options(selected_files, pdf_files):
        """Display PDF processing options and handle loading."""
        st.write("### ⚙️ 처리 옵션")
        
        # Add chunking strategy option
        merge_pages = st.checkbox(
            "📄 파일 내 페이지 병합 후 청킹", 
            value=True, 
            help="**권장**: PDF 같은 다중 페이지 파일의 모든 페이지를 하나로 합친 후 청킹합니다. 페이지 경계 없이 일관된 크기로 분할되어 문서의 논리적 흐름을 유지하는 데 유리합니다. (개별 청크의 페이지 번호는 추적되지 않음)"
        )
        
        # Generate default filenames based on selected files (only once per session)
        selected_files_key = "_".join(sorted(selected_files))  # Create a unique key
        
        if f"default_docs_name_{selected_files_key}" not in st.session_state:
            if len(selected_files) == 1:
                # Single file
                file_base = selected_files[0].replace('.pdf', '')
                default_docs_name = f"documents_{file_base}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                default_chunks_name = f"chunks_{file_base}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            else:
                # Multiple files
                default_docs_name = f"documents_{len(selected_files)}files_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                default_chunks_name = f"chunks_{len(selected_files)}files_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            
            # Store in session state
            st.session_state[f"default_docs_name_{selected_files_key}"] = default_docs_name
            st.session_state[f"default_chunks_name_{selected_files_key}"] = default_chunks_name
        
        # Get default names from session state
        default_docs_name = st.session_state[f"default_docs_name_{selected_files_key}"]
        default_chunks_name = st.session_state[f"default_chunks_name_{selected_files_key}"]
        
        col1, col2 = st.columns(2)
        
        # Initialize variables to avoid UnboundLocalError
        docs_json_name = None
        chunks_json_name = None
        
        with col1:
            save_docs_json = st.checkbox("📄 원본 문서를 JSON으로 저장", value=True)
            if save_docs_json:
                docs_json_name = st.text_input(
                    "원본 JSON 파일명:", 
                    value=default_docs_name,
                    key=f"docs_json_name_{selected_files_key}",
                    help="JSON 파일명을 수정하면 변경 사항이 자동으로 저장됩니다"
                )
        
        with col2:
            save_chunks_json = st.checkbox("🧩 청크를 JSON으로 저장", value=True)
            if save_chunks_json:
                chunks_json_name = st.text_input(
                    "청크 JSON 파일명:", 
                    value=default_chunks_name,
                    key=f"chunks_json_name_{selected_files_key}",
                    help="JSON 파일명을 수정하면 변경 사항이 자동으로 저장됩니다"
                )
        
        # Load documents button
        load_disabled = not selected_files
        
        if st.button("🚀 PDF 문서 로딩 시작", type="primary", disabled=load_disabled):
            if not selected_files:
                st.warning("⚠️ 로딩할 파일을 선택해주세요.")
                return
            
            DocumentLoadingUI._process_pdf_files(
                selected_files, merge_pages, save_docs_json, save_chunks_json,
                docs_json_name if save_docs_json else None,
                chunks_json_name if save_chunks_json else None
            )
    
    @staticmethod
    def _process_pdf_files(selected_files, merge_pages, save_docs_json, save_chunks_json, docs_json_name, chunks_json_name):
        """Process selected PDF files."""
        # Initialize document processor
        doc_processor = DocumentProcessor(st.session_state.chunk_size, st.session_state.chunk_overlap)
        
        # Create file paths
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
            if save_docs_json and docs_json_name:
                docs_json_path = JSON_OUTPUT_FOLDER / docs_json_name
                doc_processor.save_documents_to_json(documents, docs_json_path)
            
            # Display document statistics
            DocumentLoadingUI._display_loading_statistics(documents, selected_files, doc_processor)
            
            # Split documents
            with st.spinner("🧩 문서 청크 분할 중..."):
                chunks = doc_processor.split_documents(documents, merge_pages=merge_pages)
            
            st.session_state.document_chunks = chunks
            st.session_state.documents_loaded = True
            
            # Save chunks to JSON if requested
            if save_chunks_json and chunks_json_name:
                chunks_json_path = JSON_OUTPUT_FOLDER / chunks_json_name
                doc_processor.save_chunks_to_json(chunks, chunks_json_path)
            
            # Display chunk statistics
            DocumentLoadingUI._display_chunk_statistics(chunks, selected_files)
            
            st.success(f"✅ {len(selected_files)}개 파일의 문서 로딩 및 전처리 완료! ({len(chunks)}개 청크 생성)")
        else:
            st.error("❌ 문서 로딩에 실패했습니다.")
    
    @staticmethod
    def _display_loading_statistics(documents, selected_files, doc_processor):
        """Display document loading statistics."""
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
    
    @staticmethod
    def _display_chunk_statistics(chunks, selected_files):
        """Display chunk statistics."""
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
    
    @staticmethod
    def _display_json_loading_tab():
        """Display JSON loading tab."""
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
                    DocumentLoadingUI._process_json_file(json_path, json_info, doc_processor)
        else:
            st.info("📭 저장된 JSON 파일이 없습니다. 먼저 PDF를 로딩하여 JSON으로 저장해주세요.")
    
    @staticmethod
    def _process_json_file(json_path, json_info, doc_processor):
        """Process JSON file loading."""
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
    
    @staticmethod
    def _display_json_management_tab():
        """Display JSON management tab."""
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
                DocumentLoadingUI._display_json_file_operations(json_files, doc_processor)
        else:
            st.info("📭 관리할 JSON 파일이 없습니다.")
    
    @staticmethod
    def _display_json_file_operations(json_files, doc_processor):
        """Display JSON file operations."""
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
    
    @staticmethod
    def _display_current_documents_info():
        """Display current loaded documents information."""
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
                DocumentLoadingUI._display_chunk_preview(chunks)
    
    @staticmethod
    def _display_chunk_preview(chunks):
        """Display chunk preview and navigation."""
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