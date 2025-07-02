"""
Document Discovery UI
문서 발견 및 상세 검색을 위한 사용자 인터페이스
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import time

from src.rag_systems import document_discovery_rag as dd_rag
from src.utils.llm_manager import LLMManager
from src.utils.embeddings import EmbeddingManager
from src.graphs.document_discovery_graph import (
    create_summary_generation_graph,
    create_document_discovery_graph,
    create_detailed_search_graph
)
from src.config import MODELS_FOLDER, EMBEDDING_MODEL

class DocumentDiscoveryUI:
    """문서 발견 RAG를 위한 UI 클래스"""

    def __init__(self):
        # LLM, Embedding Manager, Graphs를 초기화하고 캐시합니다.
        self.llm_manager = self._get_llm_manager()
        self.embedding_manager = self._get_embedding_manager()
        self.graphs = self._get_graphs()

    @st.cache_resource
    def _get_llm_manager(_self) -> LLMManager:
        return LLMManager(
            st.session_state.get("selected_llm_model", "llama3.2:latest"),
            "http://localhost:11434",
            temperature=st.session_state.get("llm_temperature", 0.1)
        )

    @st.cache_resource
    def _get_embedding_manager(_self) -> EmbeddingManager:
        return EmbeddingManager(EMBEDDING_MODEL, MODELS_FOLDER)

    @st.cache_resource
    def _get_graphs(_self) -> Dict[str, Any]:
        """세 가지 주요 그래프를 생성하고 캐시합니다."""
        llm_manager = _self._get_llm_manager()
        embedding_manager = _self._get_embedding_manager()
        return {
            "summary_generation": create_summary_generation_graph(llm_manager),
            "document_discovery": create_document_discovery_graph(llm_manager),
            "detailed_search": create_detailed_search_graph(llm_manager, embedding_manager),
        }

    def render(self):
        """Document Discovery 탭의 전체 UI를 렌더링합니다."""
        st.header("🔍 문서 발견 및 상세 검색")
        st.markdown("""
        이 기능은 2단계로 작동합니다:
        1. **문서 발견**: 사용자 질문과 관련성이 높은 문서들을 찾습니다. (사전 요약 필요)
        2. **상세 검색**: 선택된 문서에서 구체적인 답변을 검색합니다.
        """)

        tab1, tab2, tab3 = st.tabs(["📊 문서 요약 관리", "🔍 문서 발견", "📖 상세 검색"])
        
        with tab1:
            self._display_summary_management_tab()
        with tab2:
            self._display_document_discovery_tab()
        with tab3:
            self._display_detailed_search_tab()
    
    def _display_summary_management_tab(self):
        st.subheader("📊 문서 요약 관리")
        st.markdown("모든 문서의 요약을 생성하고 관리하여 효율적인 검색을 준비합니다.")
        
        available_docs = dd_rag.get_available_documents()
        existing_summaries = dd_rag.load_document_summaries()

        if not available_docs:
            st.warning("⚠️ `docs` 폴더에 PDF 문서가 없습니다.")
            return

        st.write(f"**사용 가능한 문서**: {len(available_docs)}개 | **요약 완료된 문서**: {len(existing_summaries)}개")
        
        with st.expander("📄 문서 목록 보기"):
            for doc_info in available_docs:
                filename = doc_info["filename"]
                has_summary = filename in existing_summaries
                status = "✅ 요약 완료" if has_summary else "⏳ 요약 필요"
                
                col1, col2, col3 = st.columns([4, 2, 2])
                with col1:
                    st.write(f"**{filename}** ({doc_info.get('size_mb', 0):.1f}MB, {doc_info.get('pages', 0)}p)")
                with col2:
                    if st.button("👁️ 보기", key=f"view_{filename}", disabled=not has_summary):
                        st.session_state.selected_summary_for_view = existing_summaries[filename]
                with col3:
                    if st.button("🗑️ 삭제", key=f"delete_{filename}", disabled=not has_summary):
                        del existing_summaries[filename]
                        dd_rag.save_document_summaries(existing_summaries)
                        st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 모든 문서 요약 생성", type="primary"):
                self._run_summary_generation()
        with col2:
            if st.button("🗑️ 모든 요약 삭제"):
                dd_rag.save_document_summaries({})
                st.success("모든 요약이 삭제되었습니다.")
                time.sleep(1)
                st.rerun()
        
        if "selected_summary_for_view" in st.session_state:
            self._show_summary_detail(st.session_state.selected_summary_for_view)
            if st.button("❌ 닫기", key="close_summary_view"):
                del st.session_state.selected_summary_for_view
                st.rerun()

    def _run_summary_generation(self):
        graph = self.graphs["summary_generation"]
        if not graph:
            st.error("요약 생성 그래프를 초기화할 수 없습니다.")
            return
            
        with st.spinner("문서 요약 생성 중..."):
            final_state = graph.invoke({})
        
        st.success(f"✅ 요약 생성 완료! {final_state.get('processed_docs', 0)}개 신규 문서 처리됨")
        time.sleep(1)
        st.rerun()

    def _display_document_discovery_tab(self):
        st.subheader("🔍 문서 발견")
        st.markdown("질문을 입력하여 관련성이 높은 문서들을 찾아보세요.")
        
        summaries = dd_rag.load_document_summaries()
        if not summaries:
            st.warning("⚠️ 먼저 '문서 요약 관리' 탭에서 문서 요약을 생성해주세요.")
            return
        
        st.success(f"✅ {len(summaries)}개 문서의 요약이 준비되었습니다.")
        
        query = st.text_area("🤔 질문:", placeholder="예: 2024년 AI 정책의 주요 변화점은?", height=100, key="discovery_query")
        top_k = st.slider("최대 검색 문서 수", 1, 10, 5, key="discovery_top_k")

        if st.button("🔍 관련 문서 찾기", type="primary"):
            if query.strip():
                self._run_document_discovery(query, top_k)
            else:
                st.warning("질문을 입력해주세요.")
        
        if "discovery_results" in st.session_state:
            self._display_discovery_results(summaries)
    
    def _run_document_discovery(self, query: str, top_k: int):
        graph = self.graphs["document_discovery"]
        if not graph:
            st.error("문서 발견 그래프를 초기화할 수 없습니다.")
            return

        with st.spinner("관련 문서를 찾는 중..."):
            inputs = {"query": query, "top_k": top_k}
            result = graph.invoke(inputs)
            st.session_state.discovery_results = result.get("relevant_docs", [])

    def _display_discovery_results(self, summaries: Dict[str, Any]):
        results = st.session_state.discovery_results
        st.markdown(f"### 📋 관련 문서 목록 ({len(results)}개)")
        st.markdown(f"*검색어: \"{st.session_state.discovery_query}\"*")

        if not results:
            st.warning("관련된 문서를 찾을 수 없습니다.")
            return
            
        for i, (filename, score, explanation) in enumerate(results, 1):
            with st.container(border=True):
                col1, col2, col3 = st.columns([5, 2, 2])
                with col1:
                    st.markdown(f"**{i}. {filename}** (점수: {score})")
                    st.caption(explanation)
                with col2:
                    if st.button("👁️ 요약 보기", key=f"summary_{i}"):
                        if filename in summaries:
                            st.session_state.selected_summary_for_view = summaries[filename]
                with col3:
                    if st.button("👉 상세 검색으로", key=f"detail_{i}"):
                        st.session_state.selected_document_for_detail = filename
                        st.session_state.query_for_detail = st.session_state.discovery_query
                        st.info("📖 '상세 검색' 탭으로 이동하여 검색을 계속하세요.")

        if "selected_summary_for_view" in st.session_state:
            self._show_summary_detail(st.session_state.selected_summary_for_view)
            if st.button("❌ 닫기", key="close_summary_detail_view"):
                del st.session_state.selected_summary_for_view
                st.rerun()

    def _display_detailed_search_tab(self):
        st.subheader("📖 상세 검색")
        st.markdown("특정 문서에서 구체적인 답변을 찾아보세요.")

        available_docs = dd_rag.get_available_documents()
        doc_list = [doc["filename"] for doc in available_docs]
        
        selected_filename = None
        if "selected_document_for_detail" in st.session_state:
            default_index = doc_list.index(st.session_state.selected_document_for_detail) if st.session_state.selected_document_for_detail in doc_list else 0
            selected_filename = st.selectbox("문서 선택", doc_list, index=default_index, key="detail_filename_select")
        else:
            selected_filename = st.selectbox("문서 선택", doc_list, key="detail_filename_select")

        query = st.text_area("🤔 질문", value=st.session_state.get("query_for_detail", ""), height=100, key="detail_query")
        chunk_k = st.slider("참고할 청크 수", 3, 15, 5, key="detail_chunk_k")
        
        if st.button("🔍 상세 검색 실행", type="primary"):
            if selected_filename and query.strip():
                self._run_detailed_search(selected_filename, query, chunk_k)
            else:
                st.warning("문서와 질문을 모두 입력해주세요.")

        if "detailed_search_result" in st.session_state:
            self._display_detailed_search_result()

    def _run_detailed_search(self, filename: str, query: str, top_k: int):
        graph = self.graphs["detailed_search"]
        if not graph:
            st.error("상세 검색 그래프를 초기화할 수 없습니다.")
            return

        with st.spinner(f"'{filename}'에서 답변을 찾는 중..."):
            inputs = {
                "filename": filename,
                "query": query,
                "top_k": top_k,
                "vector_store_config": {"vector_store_type": st.session_state.get("vector_store_type", "faiss")}
            }
            result = graph.invoke(inputs)
            st.session_state.detailed_search_result = result
            st.session_state.detailed_search_filename = filename

    def _display_detailed_search_result(self):
        result = st.session_state.detailed_search_result
        if result.get("error"):
            st.error(f"❌ {result['error']}")
            return

        st.markdown(f"### 💡 답변 (출처: {st.session_state.detailed_search_filename})")
        st.markdown(result["answer"])
        
        with st.expander("📝 참고한 문서 내용 보기"):
            for chunk in result.get("relevant_chunks", []):
                st.text_area(
                    f"페이지 {chunk['metadata'].get('page', 'N/A')}",
                    chunk["content"],
                    height=150,
                    disabled=True
                )

    def _show_summary_detail(self, summary_data: Dict[str, Any]):
        st.markdown(f"### 📄 {summary_data['filename']} 요약 상세")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 페이지", f"{summary_data.get('pages', 0)}개")
            st.metric("💾 크기", f"{summary_data.get('size_mb', 0):.1f}MB")
        with col2:
            st.metric("📅 생성일", summary_data.get('generated_at', 'Unknown')[:10])
            st.metric("📝 제목", (summary_data.get('title') or "제목 없음")[:20] + "...")
        st.markdown("#### 📋 요약 내용")
        st.markdown(summary_data['summary'])
        if summary_data.get('preview'):
            with st.expander("👀 문서 미리보기 (첫 페이지)"):
                st.text(summary_data['preview']) 