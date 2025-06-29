"""
Document Discovery UI
문서 발견 및 상세 검색을 위한 사용자 인터페이스
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import time

from src.rag_systems.document_discovery_rag import DocumentDiscoveryRAG
from src.utils.llm_manager import LLMManager
from src.utils.embeddings import EmbeddingManager
from src.utils.vector_store import VectorStoreManager
from src.config import MODELS_FOLDER, EMBEDDING_MODEL


class DocumentDiscoveryUI:
    """문서 발견 RAG를 위한 UI 클래스"""
    
    @staticmethod
    def display_document_discovery_tab():
        """Document Discovery 탭 표시"""
        st.header("🔍 문서 발견 및 상세 검색")
        st.markdown("""
        이 기능은 2단계로 작동합니다:
        1. **문서 발견**: 사용자 질문과 관련성이 높은 문서들을 찾습니다
        2. **상세 검색**: 선택된 문서에서 구체적인 답변을 검색합니다
        """)
        
        # RAG 시스템 초기화
        rag_system = DocumentDiscoveryUI._get_rag_system()
        if not rag_system:
            st.error("❌ RAG 시스템 초기화 실패")
            return
        
        # 메인 탭 구성
        tab1, tab2, tab3 = st.tabs(["📊 문서 요약 생성", "🔍 문서 발견", "📖 상세 검색"])
        
        with tab1:
            DocumentDiscoveryUI._display_summary_generation_tab(rag_system)
        
        with tab2:
            DocumentDiscoveryUI._display_document_discovery_tab(rag_system)
        
        with tab3:
            DocumentDiscoveryUI._display_detailed_search_tab(rag_system)
    
    @staticmethod
    def _get_rag_system() -> Optional[DocumentDiscoveryRAG]:
        """RAG 시스템 인스턴스 반환"""
        try:
            # LLM Manager
            llm_manager = LLMManager(
                st.session_state.get("selected_llm_model", "llama3.2:3b"),
                "http://localhost:11434",
                temperature=st.session_state.get("llm_temperature", 0.1)
            )
            
            # Embedding Manager
            embedding_manager = EmbeddingManager(EMBEDDING_MODEL, MODELS_FOLDER)
            embeddings = embedding_manager.get_embeddings()
            
            # Vector Store Manager
            vector_store_manager = VectorStoreManager(
                embeddings=embeddings,
                vector_store_type=st.session_state.get("vector_store_type", "faiss"),
                collection_name="document_discovery"
            )
            
            return DocumentDiscoveryRAG(llm_manager, embedding_manager, vector_store_manager)
            
        except Exception as e:
            st.error(f"RAG 시스템 초기화 실패: {str(e)}")
            return None
    
    @staticmethod
    def _display_summary_generation_tab(rag_system: DocumentDiscoveryRAG):
        """문서 요약 생성 탭"""
        st.subheader("📊 문서 요약 생성")
        st.markdown("모든 문서의 요약을 생성하여 효율적인 검색을 준비합니다.")
        
        # 사용 가능한 문서 목록 표시
        available_docs = rag_system.get_available_documents()
        existing_summaries = rag_system.load_document_summaries()
        
        if not available_docs:
            st.warning("⚠️ docs 폴더에 PDF 문서가 없습니다.")
            return
        
        # 문서 상태 표시
        st.write(f"**사용 가능한 문서**: {len(available_docs)}개")
        st.write(f"**요약 완료된 문서**: {len(existing_summaries)}개")
        
        # 문서 목록 표시
        with st.expander("📄 문서 목록 보기"):
            for doc_info in available_docs:
                filename = doc_info["filename"]
                has_summary = filename in existing_summaries
                status = "✅ 요약 완료" if has_summary else "⏳ 요약 필요"
                
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.write(f"**{filename}**")
                    st.caption(f"크기: {doc_info.get('size_mb', 0):.1f}MB, 페이지: {doc_info.get('pages', 0)}개")
                with col2:
                    st.write(status)
                with col3:
                    if has_summary:
                        if st.button("🔍 보기", key=f"view_{filename}"):
                            # 세션 상태에 선택된 요약 저장
                            st.session_state.selected_summary_generation = existing_summaries[filename]
                            st.session_state.show_summary_generation = True
                with col4:
                    if has_summary:
                        if st.button("🗑️ 삭제", key=f"delete_{filename}"):
                            del existing_summaries[filename]
                            rag_system.save_document_summaries(existing_summaries)
                            # 삭제된 문서의 요약이 현재 표시 중이면 닫기
                            if (st.session_state.get("selected_summary_generation", {}).get("filename") == filename):
                                st.session_state.show_summary_generation = False
                                st.session_state.selected_summary_generation = None
                            st.rerun()
        
        # 요약 생성 버튼
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 모든 문서 요약 생성", type="primary"):
                DocumentDiscoveryUI._generate_all_summaries(rag_system)
        
        with col2:
            if st.button("🗑️ 모든 요약 삭제"):
                if st.session_state.get("confirm_delete_summaries", False):
                    rag_system.save_document_summaries({})
                    st.success("모든 요약이 삭제되었습니다.")
                    st.session_state.confirm_delete_summaries = False
                    # 요약 표시 상태 초기화
                    st.session_state.show_summary_generation = False
                    st.session_state.selected_summary_generation = None
                    st.session_state.show_summary_detail = False
                    st.session_state.selected_summary = None
                    st.rerun()
                else:
                    st.session_state.confirm_delete_summaries = True
                    st.warning("다시 클릭하면 모든 요약이 삭제됩니다.")
        
        # 선택된 요약 표시 (세션 상태 기반)
        if st.session_state.get("show_summary_generation", False) and st.session_state.get("selected_summary_generation"):
            st.markdown("---")
            DocumentDiscoveryUI._show_summary_detail(st.session_state.selected_summary_generation)
            
            # 요약 닫기 버튼
            if st.button("❌ 요약 닫기", key="close_summary_generation"):
                st.session_state.show_summary_generation = False
                st.session_state.selected_summary_generation = None
                st.rerun()
    
    @staticmethod
    def _show_summary_detail(summary_data: Dict[str, Any]):
        """요약 상세 정보 표시"""
        st.markdown(f"### 📄 {summary_data['filename']} 요약 상세")
        
        # 문서 기본 정보를 컬럼으로 표시
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 페이지", f"{summary_data.get('pages', 0)}개")
        with col2:
            st.metric("💾 크기", f"{summary_data.get('size_mb', 0):.1f}MB")
        with col3:
            st.metric("📅 생성일", 
                     summary_data.get('generated_at', 'Unknown')[:10] if summary_data.get('generated_at') else 'Unknown')
        with col4:
            st.metric("📝 제목", summary_data.get('title', 'Unknown')[:20] + "..." if len(summary_data.get('title', '')) > 20 else summary_data.get('title', 'Unknown'))
        
        # 요약 내용
        st.markdown("#### 📋 요약 내용")
        st.markdown(summary_data['summary'])
        
        # 문서 미리보기 (접기 가능)
        if summary_data.get('preview'):
            with st.expander("👀 문서 미리보기"):
                st.text(summary_data['preview'])
    
    @staticmethod
    def _generate_all_summaries(rag_system: DocumentDiscoveryRAG):
        """모든 문서 요약 생성"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(processed, total, message):
            progress = processed / total if total > 0 else 0
            progress_bar.progress(progress)
            status_text.text(f"진행률: {processed}/{total} - {message}")
        
        with st.spinner("문서 요약 생성 중..."):
            summaries = rag_system.generate_all_summaries(progress_callback)
            
        st.success(f"✅ 요약 생성 완료! {len(summaries)}개 문서 처리됨")
        progress_bar.empty()
        status_text.empty()
        time.sleep(1)
        st.rerun()
    
    @staticmethod
    def _display_document_discovery_tab(rag_system: DocumentDiscoveryRAG):
        """문서 발견 탭"""
        st.subheader("🔍 문서 발견")
        st.markdown("질문을 입력하여 관련성이 높은 문서들을 찾아보세요.")
        
        # 요약 상태 확인
        summaries = rag_system.load_document_summaries()
        if not summaries:
            st.warning("⚠️ 먼저 '문서 요약 생성' 탭에서 문서 요약을 생성해주세요.")
            return
        
        st.success(f"✅ {len(summaries)}개 문서의 요약이 준비되었습니다.")
        
        # 질문 입력
        query = st.text_area(
            "🤔 질문을 입력하세요:",
            placeholder="예: 2024년 AI 정책의 주요 변화점은 무엇인가요?",
            height=100,
            key="discovery_query_input"
        )
        
        # 검색 옵션
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("검색할 문서 수", min_value=1, max_value=10, value=5)
        with col2:
            show_scores = st.checkbox("관련성 점수 표시", value=True)
        
        # 검색 실행
        if st.button("🔍 관련 문서 찾기", type="primary") and query:
            # 새 검색 시 이전 요약 표시 닫기
            st.session_state.show_summary_detail = False
            st.session_state.selected_summary = None
            
            with st.spinner("관련 문서를 찾는 중..."):
                relevant_docs = rag_system.find_relevant_documents(query, top_k)
            
            if relevant_docs:
                # 세션 상태에 결과 저장
                st.session_state.discovery_results = relevant_docs
                st.session_state.discovery_query = query
                st.session_state.discovery_show_scores = show_scores
            else:
                st.session_state.discovery_results = []
                st.warning("관련된 문서를 찾을 수 없습니다.")
        
        # 저장된 검색 결과 표시 (세션 상태 기반)
        if st.session_state.get("discovery_results") and len(st.session_state.discovery_results) > 0:
            relevant_docs = st.session_state.discovery_results
            saved_show_scores = st.session_state.get("discovery_show_scores", True)
            
            st.markdown("### 📋 관련 문서 목록")
            st.markdown(f"*검색어: \"{st.session_state.get('discovery_query', '')}\"*")
            
            for i, (filename, score, explanation) in enumerate(relevant_docs, 1):
                with st.container():
                    col1, col2, col3 = st.columns([6, 2, 2])
                    
                    with col1:
                        st.markdown(f"**{i}. {filename}**")
                        if saved_show_scores:
                            st.markdown(f"*관련성 점수: {score}점*")
                        st.markdown(f"📝 {explanation}")
                    
                    with col2:
                        if st.button("📖 요약 보기", key=f"summary_{i}"):
                            if filename in summaries:
                                # 세션 상태에 선택된 요약 저장
                                st.session_state.selected_summary = summaries[filename]
                                st.session_state.show_summary_detail = True
                    
                    with col3:
                        if st.button("🔍 상세 검색", key=f"detail_{i}"):
                            st.session_state.selected_document = filename
                            st.session_state.search_query = st.session_state.get("discovery_query", "")
                            st.info("📖 '상세 검색' 탭으로 이동하여 검색을 계속하세요.")
                    
                    st.divider()
        elif st.session_state.get("discovery_results") is not None and len(st.session_state.discovery_results) == 0:
            st.warning("관련된 문서를 찾을 수 없습니다.")
        
        # 선택된 요약 표시 (세션 상태 기반)
        if st.session_state.get("show_summary_detail", False) and st.session_state.get("selected_summary"):
            st.markdown("---")
            DocumentDiscoveryUI._show_summary_detail(st.session_state.selected_summary)
            
            # 요약 닫기 버튼
            if st.button("❌ 요약 닫기"):
                st.session_state.show_summary_detail = False
                st.session_state.selected_summary = None
                st.rerun()
    
    @staticmethod
    def _display_detailed_search_tab(rag_system: DocumentDiscoveryRAG):
        """상세 검색 탭"""
        st.subheader("📖 상세 검색")
        st.markdown("특정 문서에서 구체적인 답변을 찾아보세요.")
        
        # 문서 선택 방법
        selection_method = st.radio(
            "문서 선택 방법:",
            ["📋 직접 선택", "🔍 발견된 문서에서 선택"],
            horizontal=True
        )
        
        selected_filename = None
        search_query = ""
        
        if selection_method == "📋 직접 선택":
            # 직접 문서 선택
            available_docs = rag_system.get_available_documents()
            if available_docs:
                selected_filename = st.selectbox(
                    "문서 선택:",
                    options=[doc["filename"] for doc in available_docs],
                    format_func=lambda x: f"{x} ({next((d['size_mb'] for d in available_docs if d['filename'] == x), 0):.1f}MB)"
                )
        
        else:
            # 발견된 문서에서 선택
            if "discovery_results" in st.session_state and st.session_state.discovery_results:
                discovery_results = st.session_state.discovery_results
                
                # 이전에 선택된 문서가 있는지 확인
                default_index = 0
                if "selected_document" in st.session_state:
                    try:
                        default_index = [doc[0] for doc in discovery_results].index(st.session_state.selected_document)
                    except ValueError:
                        default_index = 0
                
                selected_filename = st.selectbox(
                    "발견된 문서에서 선택:",
                    options=[doc[0] for doc in discovery_results],
                    index=default_index,
                    format_func=lambda x: f"{x} (관련성: {next((doc[1] for doc in discovery_results if doc[0] == x), 0)}점)"
                )
                search_query = st.session_state.get("discovery_query", "")
            else:
                st.info("먼저 '문서 발견' 탭에서 관련 문서를 찾아보세요.")
                return
        
        if selected_filename:
            # 문서 개요 표시
            overview = rag_system.get_document_overview(selected_filename)
            if overview:
                with st.expander(f"📄 {selected_filename} 개요"):
                    st.write(f"**제목**: {overview.get('title', 'Unknown')}")
                    st.write(f"**페이지 수**: {overview.get('pages', 0)}개")
                    st.write(f"**파일 크기**: {overview.get('size_mb', 0):.1f}MB")
                    if overview.get('summary'):
                        st.markdown("**요약**:")
                        st.markdown(overview['summary'])
            
            # 질문 입력
            if search_query or st.session_state.get("search_query"):
                initial_query = search_query or st.session_state.get("search_query", "")
                query = st.text_area("🤔 질문 (수정 가능):", value=initial_query, height=100)
            else:
                query = st.text_area(
                    "🤔 이 문서에 대한 질문을 입력하세요:",
                    placeholder="예: 이 문서의 주요 결론은 무엇인가요?",
                    height=100
                )
            
            # 검색 옵션
            col1, col2 = st.columns(2)
            with col1:
                chunk_k = st.slider("검색할 청크 수", min_value=3, max_value=15, value=5)
            with col2:
                show_chunks = st.checkbox("관련 청크 표시", value=False)
            
            if st.button("🔍 상세 검색 실행", type="primary") and query:
                with st.spinner(f"{selected_filename}에서 답변을 찾는 중..."):
                    result = rag_system.detailed_search(selected_filename, query, chunk_k)
                
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    st.markdown("### 💡 답변")
                    st.markdown(result["answer"])
                    
                    st.markdown(f"### 📊 검색 정보")
                    st.write(f"**대상 문서**: {result['filename']}")
                    st.write(f"**관련 청크 수**: {result['total_chunks_found']}개")
                    
                    if show_chunks and result.get("relevant_chunks"):
                        st.markdown("### 📝 관련 청크")
                        for i, chunk_info in enumerate(result["relevant_chunks"], 1):
                            with st.expander(f"청크 {i} (페이지: {chunk_info['metadata'].get('page', 'Unknown')})"):
                                st.text(chunk_info["content"])
        
        else:
            st.info("문서를 선택해주세요.") 