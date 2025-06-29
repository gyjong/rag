"""Web Search RAG UI module."""

import streamlit as st
import time
from typing import Dict, Any, List
import json

from ..rag_systems.web_search_rag import WebSearchRAG
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
            
            1. **질문 의도 정제**
               - 사용자 질문을 분석하여 검색에 최적화된 쿼리 생성
               - 한국어/영어 검색 쿼리 분별 생성
               - 검색 의도 파악 및 분류
            
            2. **웹 검색 실행**
               - Google 검색을 통한 관련 웹사이트 수집
               - 다양한 소스에서 정보 수집
               - 웹페이지 내용 자동 추출
            
            3. **검색 결과 분석**
               - 검색 결과의 신뢰성 평가
               - 정보의 일관성 및 최신성 확인
               - 전문적인 관점에서 내용 분석
            
            4. **전문가 답변 생성**
               - 수집된 정보를 종합하여 포괄적인 답변 생성
               - 출처 정보 포함
               - 구조화된 답변 제공
            
            5. **진행 과정 표시**
               - 실시간 진행 상황 표시
               - 각 단계별 결과 확인
               - 투명한 처리 과정
            """)
        
        # Settings section
        st.subheader("⚙️ 검색 설정")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_results = st.slider(
                "검색 결과 수",
                min_value=3,
                max_value=10,
                value=5,
                help="웹 검색에서 가져올 결과의 개수를 설정합니다."
            )
        
        with col2:
            temperature = st.slider(
                "LLM Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                help="답변 생성의 창의성을 조절합니다. 낮을수록 일관된 답변, 높을수록 창의적인 답변을 생성합니다."
            )
        
        # Query input section
        st.subheader("❓ 질문 입력")
        
        # Pre-defined example queries
        example_queries = [
            "2025년 인공지능 산업 전망은?",
            "최신 GPT 모델의 특징과 성능은?",
            "한국의 디지털 전환 정책 현황은?",
            "메타버스 기술의 최근 동향은?",
            "양자컴퓨팅의 상용화 전망은?",
            "ChatGPT 최신 업데이트",  # Simple test query
            "AI 뉴스 2025"  # Another simple test query
        ]
        
        selected_example = st.selectbox(
            "예시 질문 선택 (선택사항)",
            ["직접 입력"] + example_queries,
            help="미리 준비된 예시 질문을 선택하거나 직접 질문을 입력할 수 있습니다."
        )
        
        if selected_example != "직접 입력":
            default_query = selected_example
        else:
            default_query = ""
        
        user_query = st.text_area(
            "질문을 입력하세요:",
            value=default_query,
            height=100,
            placeholder="예: 2025년 인공지능 기술 트렌드에 대해 알려주세요.",
            help="궁금한 내용을 자세히 입력해주세요. 구체적일수록 더 정확한 답변을 받을 수 있습니다."
        )
        
        # Search button
        search_col1, search_col2, search_col3 = st.columns([1, 2, 1])
        
        with search_col2:
            if st.button("🔍 웹 검색 및 답변 생성", type="primary", use_container_width=True):
                if not user_query.strip():
                    st.error("질문을 입력해주세요.")
                else:
                    WebSearchUI._perform_web_search(user_query, num_results, temperature)
        
        # Display results if available
        if "web_search_results" in st.session_state and st.session_state.web_search_results:
            WebSearchUI._display_search_results(st.session_state.web_search_results)
    
    @staticmethod
    def _perform_web_search(query: str, num_results: int, temperature: float):
        """Perform web search and display results."""
        try:
            # Initialize WebSearchRAG
            llm_model = st.session_state.get("selected_llm_model", "llama3.2:3b")
            web_search_rag = WebSearchRAG(llm_model, OLLAMA_BASE_URL, temperature)
            
            # Create progress tracking
            progress_container = st.container()
            status_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # Execute search and answer pipeline
            with status_container:
                status_text.text("🔍 검색 및 분석을 시작합니다...")
                progress_bar.progress(0.1)
                
                # Start the process
                results = web_search_rag.search_and_answer(query, num_results)
                
                # Update progress for each step
                total_steps = max(len(results.get("process_steps", [])), 4)
                for i, step in enumerate(results.get("process_steps", [])):
                    progress_bar.progress((i + 1) / total_steps)
                    status_text.text(f"{step}")
                    time.sleep(0.3)  # Reduced delay
                
                # Complete
                progress_bar.progress(1.0)
                status_text.text("✅ 검색 및 분석이 완료되었습니다!")
            
            # Store results in session state
            st.session_state.web_search_results = results
            
            # Show summary immediately
            search_count = len(results.get('search_results', []))
            if search_count > 0:
                st.success(f"검색 완료! {search_count}개의 결과를 찾았습니다.")
            else:
                st.warning("웹 검색 결과를 찾지 못했지만, LLM 지식을 기반으로 답변을 생성했습니다.")
            
            # Small delay before showing results
            time.sleep(1)
            
            # Clear progress indicators
            progress_container.empty()
            status_container.empty()
            
        except Exception as e:
            st.error(f"웹 검색 중 오류가 발생했습니다: {str(e)}")
            
            with st.expander("🔧 오류 상세 정보", expanded=False):
                st.exception(e)
                import traceback
                st.code(traceback.format_exc())
    
    @staticmethod
    def _display_search_results(results: Dict[str, Any]):
        """Display the search results in a structured format."""
        st.subheader("📊 검색 결과")
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 최종 답변",
            "🔍 질문 분석",
            "🌐 검색 결과",
            "📊 분석 과정",
            "📝 전체 과정"
        ])
        
        with tab1:
            st.markdown("### 💡 전문가 답변")
            if results.get("final_answer"):
                st.markdown(results["final_answer"])
            else:
                st.warning("답변이 생성되지 않았습니다.")
        
        with tab2:
            st.markdown("### 🔍 질문 의도 분석")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**원본 질문:**")
                original_query = results.get("original_query", "")
                if original_query:
                    st.info(original_query)
                else:
                    st.warning("원본 질문이 없습니다.")
            
            with col2:
                refined_queries = results.get("refined_queries", {})
                st.markdown("**검색 의도:**")
                intent = refined_queries.get("intent", "") if refined_queries else ""
                if intent:
                    st.info(intent)
                else:
                    st.warning("검색 의도가 분석되지 않았습니다.")
            
            if refined_queries and (refined_queries.get("korean") or refined_queries.get("english")):
                st.markdown("**정제된 검색 쿼리:**")
                col3, col4 = st.columns(2)
                
                with col3:
                    st.markdown("*한국어 쿼리:*")
                    korean_query = refined_queries.get("korean", "")
                    if korean_query:
                        st.code(korean_query, language="text")
                    else:
                        st.warning("한국어 쿼리가 생성되지 않았습니다.")
                
                with col4:
                    st.markdown("*영어 쿼리:*")
                    english_query = refined_queries.get("english", "")
                    if english_query:
                        st.code(english_query, language="text")
                    else:
                        st.warning("영어 쿼리가 생성되지 않았습니다.")
            else:
                st.warning("정제된 검색 쿼리가 생성되지 않았습니다.")
            
            # Show debugging info only if nothing was found
            if not refined_queries or not any([refined_queries.get("korean"), refined_queries.get("english"), refined_queries.get("intent")]):
                with st.expander("🔧 디버깅 정보", expanded=False):
                    st.write("Refined queries data:", refined_queries)
        
        with tab3:
            st.markdown("### 🌐 웹 검색 결과")
            
            search_results = results.get("search_results", [])
            
            if search_results and len(search_results) > 0:
                st.success(f"총 {len(search_results)}개의 검색 결과를 찾았습니다.")
                
                for i, result in enumerate(search_results, 1):
                    title = result.get('title', '제목 없음')
                    with st.expander(f"🔗 검색 결과 {i}: {title}", expanded=False):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            url = result.get('url', '')
                            domain = result.get('domain', '')
                            st.markdown(f"**URL:** {url}")
                            st.markdown(f"**도메인:** {domain}")
                        
                        with col2:
                            if url and url != "https://error":
                                if st.button(f"🔗 링크 열기", key=f"open_link_{i}"):
                                    st.markdown(f"[새 창에서 열기]({url})")
                        
                        st.markdown("**내용 미리보기:**")
                        content = result.get('content', '')
                        if content:
                            if len(content) > 500:
                                st.text(content[:500] + "...")
                                if st.button(f"전체 내용 보기", key=f"full_content_{i}"):
                                    st.text(content)
                            else:
                                st.text(content)
                        else:
                            st.warning("내용이 없습니다.")
            else:
                st.warning("검색 결과가 없습니다.")
                
                # Show debugging info only when no results
                with st.expander("🔧 디버깅 정보", expanded=False):
                    st.write("Search results type:", type(search_results))
                    st.write("Search results length:", len(search_results) if search_results else 0)
                    st.write("Search results data:", search_results)
        
        with tab4:
            st.markdown("### 📊 분석 과정")
            
            analysis = results.get("analysis", "")
            
            if analysis and analysis.strip():
                st.markdown("#### 분석 결과:")
                st.markdown(analysis)
            else:
                st.warning("분석 결과가 없습니다.")
                
                # Show debugging info only when no analysis
                with st.expander("🔧 디버깅 정보", expanded=False):
                    st.write("Analysis type:", type(analysis))
                    st.write("Analysis length:", len(analysis) if analysis else 0)
                    st.write("Analysis content:", repr(analysis))
        
        with tab5:
            st.markdown("### 📝 전체 처리 과정")
            
            process_steps = results.get("process_steps", [])
            if process_steps:
                for i, step in enumerate(process_steps, 1):
                    st.markdown(f"{i}. {step}")
            
            # Show technical details
            with st.expander("🔧 기술적 세부사항", expanded=False):
                st.markdown("**처리된 데이터 요약:**")
                st.write(f"- 원본 질문: {results.get('original_query', 'N/A')}")
                st.write(f"- 정제된 쿼리: {results.get('refined_queries', {})}")
                st.write(f"- 검색 결과 수: {len(results.get('search_results', []))}")
                st.write(f"- 분석 길이: {len(results.get('analysis', '')) if results.get('analysis') else 0} 문자")
                st.write(f"- 최종 답변 길이: {len(results.get('final_answer', '')) if results.get('final_answer') else 0} 문자")
            
            # Export functionality
            st.markdown("---")
            st.markdown("### 📤 결과 내보내기")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 텍스트로 복사"):
                    export_text = WebSearchUI._format_results_for_export(results)
                    st.text_area("복사할 내용:", value=export_text, height=200)
            
            with col2:
                if st.button("💾 JSON 다운로드"):
                    json_str = json.dumps(results, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="결과 다운로드",
                        data=json_str,
                        file_name=f"web_search_results_{int(time.time())}.json",
                        mime="application/json"
                    )
    
    @staticmethod
    def _format_results_for_export(results: Dict[str, Any]) -> str:
        """Format results for text export."""
        export_lines = []
        export_lines.append("=== 웹 검색 RAG 결과 ===\n")
        
        export_lines.append(f"질문: {results.get('original_query', '')}\n")
        
        refined_queries = results.get('refined_queries', {})
        if refined_queries:
            export_lines.append("=== 질문 분석 ===")
            export_lines.append(f"검색 의도: {refined_queries.get('intent', '')}")
            export_lines.append(f"한국어 쿼리: {refined_queries.get('korean', '')}")
            export_lines.append(f"영어 쿼리: {refined_queries.get('english', '')}\n")
        
        export_lines.append("=== 최종 답변 ===")
        export_lines.append(results.get('final_answer', ''))
        
        export_lines.append("\n=== 검색 결과 ===")
        search_results = results.get('search_results', [])
        for i, result in enumerate(search_results, 1):
            export_lines.append(f"\n{i}. {result.get('title', '')}")
            export_lines.append(f"   URL: {result.get('url', '')}")
            export_lines.append(f"   도메인: {result.get('domain', '')}")
        
        return "\n".join(export_lines) 