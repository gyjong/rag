"""Web Search RAG UI module."""

import streamlit as st
import time
from typing import Dict, Any, List
import json

from ..graphs.web_search_graph import run_web_search_graph
from ..config import OLLAMA_BASE_URL


class WebSearchUI:
    """Web Search RAG UI class."""
    
    @staticmethod
    def display_web_search_tab():
        """Display the web search RAG tab."""
        st.header("🌐 웹 검색 RAG")
        st.markdown("외부 검색 도구를 활용하여 최신 정보를 기반으로 전문가적인 답변을 생성합니다.")
        
        # Information section
        with st.expander("💡 웹 검색 RAG 기능", expanded=False):
            st.markdown("""
            ### 주요 기능
            
            1. **질문 의도 정제**: 사용자 질문을 분석하여 검색에 최적화된 쿼리 생성
            2. **웹 검색 실행**: Google 검색을 통한 관련 웹사이트 수집
            3. **답변 생성**: 수집된 정보를 종합하여 포괄적인 답변 생성 및 출처 제공
            4. **진행 과정 표시**: 실시간으로 각 단계별 처리 과정 확인
            """)
        
        # Settings section
        st.subheader("⚙️ 검색 설정")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_results = st.slider("검색 결과 수", min_value=3, max_value=10, value=5)
        
        with col2:
            temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        
        # Query input section
        st.subheader("❓ 질문 입력")
        
        example_queries = [
            "2025년 인공지능 산업 전망은?", "최신 GPT 모델의 특징과 성능은?", "한국의 디지털 전환 정책 현황은?"
        ]
        selected_example = st.selectbox("예시 질문 선택", ["직접 입력"] + example_queries)
        
        user_query = st.text_area(
            "질문을 입력하세요:",
            value=selected_example if selected_example != "직접 입력" else "",
            height=100
        )
        
        if st.button("🔍 웹 검색 및 답변 생성", type="primary", use_container_width=True):
            if not user_query.strip():
                st.error("질문을 입력해주세요.")
            else:
                WebSearchUI._perform_web_search(user_query, num_results, temperature)
        
        if "web_search_results" in st.session_state and st.session_state.web_search_results:
            WebSearchUI._display_search_results(st.session_state.web_search_results)
    
    @staticmethod
    def _perform_web_search(query: str, num_results: int, temperature: float):
        """Perform web search and display results using LangGraph."""
        try:
            llm_model = st.session_state.get("selected_llm_model", "llama3.2:8b")
            st.session_state.web_search_results = None
            
            with st.spinner("웹 검색 및 답변 생성 중..."):
                results = run_web_search_graph(
                    query=query, num_results=num_results, llm_model=llm_model, temperature=temperature
                )
            
            st.session_state.web_search_results = results
            
            search_count = len(results.get('search_results', []))
            if search_count > 0:
                st.success(f"검색 완료! {search_count}개의 결과를 찾았습니다.")
            else:
                st.warning("웹 검색 결과를 찾지 못했거나, 답변 생성에 실패했습니다.")
            
            st.rerun()

        except Exception as e:
            st.error(f"웹 검색 중 오류가 발생했습니다: {str(e)}")
            with st.expander("🔧 오류 상세 정보", expanded=False):
                st.exception(e)

    @staticmethod
    def _display_search_results(results: Dict[str, Any]):
        """Display the search results in a structured format."""
        st.subheader("📊 검색 결과")
        
        tab_titles = ["🎯 최종 답변", "🔍 질문 분석", "🌐 검색 결과", "🛤️ 처리 과정", "📝 전체 과정"]
        tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_titles)
        
        with tab1:
            st.markdown("### 💡 전문가 답변")
            st.markdown(results.get("final_answer", "답변이 생성되지 않았습니다."))
        
        with tab2:
            st.markdown("### 🔍 질문 의도 분석")
            refined_queries = results.get("refined_queries", {})
            st.info(f"**원본 질문:** {results.get('original_query', 'N/A')}")
            st.info(f"**검색 의도:** {refined_queries.get('intent', 'N/A')}")
            st.code(f"한국어 쿼리: {refined_queries.get('korean', 'N/A')}\n영어 쿼리: {refined_queries.get('english', 'N/A')}", language='text')
        
        with tab3:
            st.markdown("### 🌐 웹 검색 결과")
            search_results = results.get("search_results", [])
            if search_results:
                for i, result in enumerate(search_results, 1):
                    with st.expander(f"🔗 {i}: {result.get('title', '제목 없음')}", expanded=False):
                        st.markdown(f"**URL:** [{result.get('domain', '')}]({result.get('url', '')})")
                        st.text(result.get('content', '내용 없음')[:500] + "...")
            else:
                st.warning("검색 결과가 없습니다.")
        
        with tab4:
            st.markdown("### 🛤️ 처리 과정")
            process_steps = results.get("process_steps", [])
            if process_steps:
                st.text('\n'.join(process_steps))
            else:
                st.warning("처리 과정 정보가 없습니다.")
        
        with tab5:
            st.markdown("### 📝 전체 과정 (JSON)")
            st.json(results, expanded=False)
            
            st.markdown("### 📤 결과 내보내기")
            if st.button("📋 텍스트로 복사"):
                export_text = WebSearchUI._format_results_for_export(results)
                st.text_area("복사할 내용:", value=export_text, height=200)

    @staticmethod
    def _format_results_for_export(results: Dict[str, Any]) -> str:
        """Format results for text export."""
        lines = ["=== 웹 검색 RAG 결과 ==="]
        lines.append(f"\n질문: {results.get('original_query', '')}")
        
        refined = results.get('refined_queries', {})
        if refined:
            lines.append(f"\n=== 질문 분석 ===\n의도: {refined.get('intent', '')}\n한국어: {refined.get('korean', '')}\n영어: {refined.get('english', '')}")
            
        lines.append(f"\n=== 최종 답변 ===\n{results.get('final_answer', '')}")
        
        search = results.get('search_results', [])
        if search:
            lines.append("\n=== 검색 결과 ===")
            for i, res in enumerate(search, 1):
                lines.append(f"\n{i}. {res.get('title', '')}\n   URL: {res.get('url', '')}")
        
        return "\n".join(lines) 