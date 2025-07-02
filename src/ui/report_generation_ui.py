"""Report Generation UI module for the RAG application."""

import streamlit as st
import json
from datetime import datetime
from typing import Dict, Any

from ..config import OLLAMA_BASE_URL
from ..graphs.report_generation_graph import run_report_generation_graph
from ..utils.vector_store import VectorStoreManager

class ReportGenerationUI:
    REPORT_TYPES = {
        "연구보고서": {
            "description": "학술적 연구 결과를 체계적으로 정리한 보고서",
            "default_outline": [
                {"title": "서론", "content_guide": "연구 배경, 목적, 필요성을 설명"},
                {"title": "이론적 배경", "content_guide": "관련 이론과 선행 연구를 검토"},
                {"title": "연구 방법론", "content_guide": "연구 설계, 데이터 수집 및 분석 방법"},
                {"title": "연구 결과", "content_guide": "연구를 통해 얻은 결과를 제시"},
                {"title": "논의", "content_guide": "결과 해석 및 이론적/실무적 함의"},
                {"title": "결론 및 제언", "content_guide": "연구 결과를 종합하고 향후 과제를 제시"}
            ]
        },
        "시장분석보고서": {
            "description": "특정 시장의 현황과 전망을 분석한 보고서",
            "default_outline": [
                {"title": "시장 개요", "content_guide": "시장의 기본 현황과 규모를 소개"},
                {"title": "시장 환경 분석", "content_guide": "PEST 분석 (정치, 경제, 사회, 기술)"},
                {"title": "시장 동향", "content_guide": "최근 시장 변화와 트렌드를 분석"},
                {"title": "경쟁 현황", "content_guide": "주요 경쟁업체와 경쟁 구조를 분석"},
                {"title": "소비자 분석", "content_guide": "고객 세그먼트, 니즈, 행동 패턴"},
                {"title": "시장 전망", "content_guide": "향후 시장 전망과 성장 가능성을 예측"},
                {"title": "전략적 제언", "content_guide": "시장 진입/확장 전략 및 리스크 관리"}
            ]
        },
        "기술동향보고서": {
            "description": "특정 기술 분야의 현황과 전망을 분석한 보고서",
            "default_outline": [
                {"title": "기술 개요", "content_guide": "기술의 정의와 특징을 설명"},
                {"title": "기술 현황", "content_guide": "현재 기술 개발 수준과 상용화 현황"},
                {"title": "주요 동향", "content_guide": "최신 기술 동향과 혁신 사례"},
                {"title": "핵심 기술 분석", "content_guide": "주요 기술 요소와 성능 지표"},
                {"title": "응용 분야", "content_guide": "기술의 다양한 응용 사례와 산업 활용"},
                {"title": "향후 전망", "content_guide": "기술 발전 방향과 미래 전망"},
                {"title": "정책 및 투자 동향", "content_guide": "관련 정책, 투자 현황, 표준화 동향"}
            ]
        },
        "정책분석보고서": {
            "description": "정책의 효과성과 영향을 종합적으로 분석한 보고서",
            "default_outline": [
                {"title": "정책 개요", "content_guide": "정책의 목적, 배경, 주요 내용"},
                {"title": "정책 환경 분석", "content_guide": "정책 수립 배경과 사회적 맥락"},
                {"title": "정책 효과 분석", "content_guide": "정책 시행 결과와 성과 지표"},
                {"title": "이해관계자 분석", "content_guide": "정책의 영향받는 주체들과 반응"},
                {"title": "비용-효과 분석", "content_guide": "정책 비용과 기대 효과의 비교"},
                {"title": "정책 개선 방안", "content_guide": "현재 정책의 문제점과 개선 제안"},
                {"title": "향후 정책 방향", "content_guide": "정책 발전 방향과 제언"}
            ]
        },
        "경영전략보고서": {
            "description": "기업의 경영 전략과 비즈니스 모델을 분석한 보고서",
            "default_outline": [
                {"title": "기업 개요", "content_guide": "기업의 비전, 미션, 핵심 가치"},
                {"title": "환경 분석", "content_guide": "SWOT 분석 및 산업 환경 분석"},
                {"title": "현재 전략 분석", "content_guide": "기존 전략의 성과와 한계점"},
                {"title": "핵심 역량", "content_guide": "기업의 핵심 경쟁력과 차별화 요소"},
                {"title": "전략적 제안", "content_guide": "새로운 전략 방향과 실행 계획"},
                {"title": "리스크 관리", "content_guide": "전략 실행 시 예상 리스크와 대응 방안"},
                {"title": "성과 측정", "content_guide": "전략 성과 측정 지표와 모니터링 체계"}
            ]
        },
        "투자분석보고서": {
            "description": "투자 대상의 가치와 투자 위험을 분석한 보고서",
            "default_outline": [
                {"title": "투자 대상 개요", "content_guide": "투자 대상의 사업 영역과 현황"},
                {"title": "재무 분석", "content_guide": "재무제표 분석 및 재무 건전성 평가"},
                {"title": "산업 분석", "content_guide": "해당 산업의 성장성과 경쟁 구조"},
                {"title": "가치 평가", "content_guide": "DCF, P/E 등 다양한 가치 평가 모델"},
                {"title": "리스크 분석", "content_guide": "투자 위험 요소와 리스크 관리 방안"},
                {"title": "투자 의견", "content_guide": "투자 권고 사항과 투자 전략"},
                {"title": "투자 후 모니터링", "content_guide": "투자 후 추적 관리 방안"}
            ]
        },
        "환경영향평가보고서": {
            "description": "사업이나 정책의 환경적 영향을 평가한 보고서",
            "default_outline": [
                {"title": "사업 개요", "content_guide": "평가 대상 사업의 내용과 규모"},
                {"title": "환경 현황", "content_guide": "사업 지역의 환경 현황과 특성"},
                {"title": "환경 영향 예측", "content_guide": "사업 시행 시 예상되는 환경 영향"},
                {"title": "대기환경 영향", "content_guide": "대기질 변화 및 영향 분석"},
                {"title": "수환경 영향", "content_guide": "수질 변화 및 수생태계 영향"},
                {"title": "생태계 영향", "content_guide": "육상생태계 및 생물다양성 영향"},
                {"title": "환경 보전 대책", "content_guide": "환경 영향 최소화 방안과 보전 대책"},
                {"title": "환경 관리 계획", "content_guide": "사업 시행 중 환경 관리 방안"}
            ]
        },
        "디지털전환보고서": {
            "description": "기업의 디지털 전환 현황과 전략을 분석한 보고서",
            "default_outline": [
                {"title": "디지털 전환 개요", "content_guide": "디지털 전환의 정의와 필요성"},
                {"title": "현재 상태 진단", "content_guide": "기업의 현재 디지털화 수준과 한계"},
                {"title": "기술 인프라 분석", "content_guide": "현재 IT 인프라와 기술 스택 현황"},
                {"title": "조직 문화 분석", "content_guide": "디지털 전환을 위한 조직 문화와 역량"},
                {"title": "디지털 전환 전략", "content_guide": "단계별 디지털 전환 로드맵과 전략"},
                {"title": "핵심 기술 도입", "content_guide": "AI, 클라우드, IoT 등 핵심 기술 도입 계획"},
                {"title": "변경 관리", "content_guide": "조직 변화 관리와 직원 교육 계획"},
                {"title": "성과 측정", "content_guide": "디지털 전환 성과 지표와 ROI 분석"}
            ]
        },
    }

    @staticmethod
    def display_report_generation_tab():
        st.header("📋 보고서 생성")
        if not ReportGenerationUI._check_vector_store():
            return

        # This is the main router for the UI state.
        if "generation_in_progress" in st.session_state and st.session_state.generation_in_progress:
            # If generation is running, call the generation method which contains the spinner and stream handling.
            ReportGenerationUI._generate_report(st.session_state.report_config)
        elif "generated_report" in st.session_state:
            # If a report is already generated, display it.
            ReportGenerationUI._display_generated_report()
        else:
            # Otherwise, show the configuration UI.
            ReportGenerationUI._display_report_configuration()
            ReportGenerationUI._display_outline_configuration()
            ReportGenerationUI._display_generation_interface()

    @staticmethod
    def _display_generated_report():
        report_data = st.session_state.generated_report
        st.success("✅ 보고서가 성공적으로 생성되었습니다.")
        
        with st.expander("📖 최종 보고서 보기", expanded=True):
            st.markdown(report_data["content"])

        ReportGenerationUI._display_download_options(report_data["content"], report_data["config"])

        if st.button("🔄 새 보고서 생성하기", use_container_width=True):
            # Clean up all related session state keys
            for key in ["generated_report", "report_config", "generation_in_progress"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    @staticmethod
    def _check_vector_store():
        if not st.session_state.get("vector_store_manager") or not st.session_state.get("vector_store_manager").get_vector_store():
            st.warning("보고서 생성을 위해 벡터 스토어를 먼저 로드하거나 생성해주세요.")
            return False
        st.success("✅ 벡터 스토어 준비 완료!")
        return True

    @staticmethod
    def _display_report_configuration():
        if "report_config" not in st.session_state:
            st.session_state.report_config = {
                "report_type": "연구보고서", "topic": "", "purpose": "정책 결정 지원", "audience": "전문가", "language": "한국어",
                "target_length": "medium", "include_visuals": False, "citation_style": "simple", "outline": []
            }
        
        with st.form("report_config_form"):
            st.subheader("📊 보고서 기본 설정")
            selected_type = st.selectbox("보고서 유형:", options=list(ReportGenerationUI.REPORT_TYPES.keys()),
                                         index=list(ReportGenerationUI.REPORT_TYPES.keys()).index(st.session_state.report_config.get("report_type", "연구보고서")))
            st.info(ReportGenerationUI.REPORT_TYPES[selected_type]['description'])
            
            topic = st.text_input("보고서 주제:", value=st.session_state.report_config.get("topic", ""), placeholder="예: 2025년 AI 산업 동향")
            purpose = st.text_input("보고서 목적:", value=st.session_state.report_config.get("purpose", ""), placeholder="예: AI 정책 수립을 위한 기초 자료")
            
            st.subheader("⚙️ 보고서 상세 옵션")
            audience = st.text_input("대상 독자:", value=st.session_state.report_config.get("audience", ""), placeholder="예: 정책 결정자, 투자자")
            include_visuals = st.checkbox("시각 요소 플레이스홀더 포함", value=st.session_state.report_config.get("include_visuals", False))

            if st.form_submit_button("설정 저장 및 목차 업데이트", use_container_width=True, type="primary"):
                st.session_state.report_config.update({
                    "report_type": selected_type, "topic": topic, "purpose": purpose,
                    "audience": audience, "include_visuals": include_visuals
                })
                if selected_type in ReportGenerationUI.REPORT_TYPES:
                    st.session_state.report_config["outline"] = ReportGenerationUI.REPORT_TYPES[selected_type]["default_outline"].copy()
                st.success("✅ 설정이 저장되었습니다!")
                st.rerun()

    @staticmethod
    def _display_outline_configuration():
        st.subheader("📑 목차 구성")
        if not st.session_state.report_config.get("outline"):
            st.info("보고서 설정을 저장하면 기본 목차가 나타납니다.")
            return

        for i, section in enumerate(st.session_state.report_config["outline"]):
            with st.expander(f"📄 {i+1}. {section['title']}", expanded=True):
                section["title"] = st.text_input("섹션 제목:", value=section["title"], key=f"title_{i}")
                section["content_guide"] = st.text_area("내용 가이드:", value=section["content_guide"], key=f"guide_{i}")
        
        # Logic to add/remove sections can be added here if needed.

    @staticmethod
    def _display_generation_interface():
        st.subheader("🚀 보고서 생성 실행")
        if not st.session_state.report_config.get('topic'):
            st.warning("보고서를 생성하려면 먼저 주제를 입력하고 설정을 저장해주세요.")
            return

        if st.button("📋 보고서 생성하기", use_container_width=True, type="primary"):
            # Set a flag to indicate generation is in progress and rerun.
            st.session_state.generation_in_progress = True
            st.rerun()

    @staticmethod
    def _generate_report(config: Dict[str, Any]):
        try:
            vector_store_manager = st.session_state.get("vector_store_manager")
            config['llm_model'] = st.session_state.selected_llm_model
            config['temperature'] = st.session_state.llm_temperature
            
            st.info("보고서 생성을 시작합니다...")
            status_text = st.empty()
            report_placeholder = st.empty()
            
            final_report_content = None
            
            with st.spinner("보고서 생성 중... 잠시만 기다려주세요."):
                for event in run_report_generation_graph(config, vector_store_manager):
                    node_name = list(event.keys())[0]
                    state_update = event[node_name]

                    if 'process_steps' in state_update:
                        status_text.text('\\n'.join(state_update['process_steps']))
                    
                    if 'report_draft' in state_update and state_update['report_draft']:
                        report_placeholder.markdown(state_update['report_draft'] + "▌")
                    
                    if 'final_report' in state_update and state_update['final_report']:
                        final_report_content = state_update['final_report']

            # Once the loop is finished, store the final result and clean up the progress flag.
            if final_report_content:
                st.session_state.generated_report = {"content": final_report_content, "config": config}
            else:
                st.error("오류: 보고서 생성에 실패했거나 최종 결과가 비어있습니다.")
            
            # Clean up the progress flag and rerun to display the final report.
            del st.session_state.generation_in_progress
            st.rerun()

        except Exception as e:
            del st.session_state.generation_in_progress
            st.error(f"❌ 보고서 생성 중 심각한 오류 발생: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    @staticmethod
    def _display_download_options(report_content: str, config: Dict[str, Any]):
        st.subheader("💾 다운로드")
        filename = f"{config.get('topic', 'report')}_{datetime.now().strftime('%Y%m%d')}"
        col1, col2 = st.columns(2)
        col1.download_button("📄 Markdown 다운로드", report_content, f"{filename}.md", "text/markdown", use_container_width=True)
        col2.download_button("⚙️ 설정 다운로드", json.dumps(config, ensure_ascii=False, indent=2), f"{filename}_config.json", "application/json", use_container_width=True) 