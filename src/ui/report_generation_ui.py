"""Report Generation UI module for the RAG application."""

import streamlit as st
import base64
from typing import Dict, Any, List
from datetime import datetime
import json

from ..config import *
from ..utils.llm_manager import LLMManager
from ..rag_systems.report_generation_rag import ReportGenerationRAG


class ReportGenerationUI:
    """UI components for report generation functionality."""
    
    # Predefined report types with their characteristics
    REPORT_TYPES = {
        "연구보고서": {
            "description": "학술적 연구 결과를 체계적으로 정리한 보고서",
            "default_outline": [
                {"title": "서론", "content_guide": "연구 배경, 목적, 필요성을 설명"},
                {"title": "이론적 배경", "content_guide": "관련 이론과 선행 연구를 검토"},
                {"title": "연구 방법", "content_guide": "연구 방법론과 절차를 설명"},
                {"title": "연구 결과", "content_guide": "연구를 통해 얻은 결과를 제시"},
                {"title": "결론 및 제언", "content_guide": "연구 결과를 종합하고 향후 과제를 제시"}
            ]
        },
        "시장분석보고서": {
            "description": "특정 시장의 현황과 전망을 분석한 보고서",
            "default_outline": [
                {"title": "시장 개요", "content_guide": "시장의 기본 현황과 규모를 소개"},
                {"title": "시장 동향", "content_guide": "최근 시장 변화와 트렌드를 분석"},
                {"title": "경쟁 현황", "content_guide": "주요 경쟁업체와 경쟁 구조를 분석"},
                {"title": "SWOT 분석", "content_guide": "시장의 강점, 약점, 기회, 위협을 분석"},
                {"title": "시장 전망", "content_guide": "향후 시장 전망과 성장 가능성을 예측"}
            ]
        },
        "정책보고서": {
            "description": "정책 방향과 대안을 제시하는 보고서",
            "default_outline": [
                {"title": "정책 현황", "content_guide": "현재 정책 상황과 문제점을 분석"},
                {"title": "환경 분석", "content_guide": "정책 환경과 영향 요인을 분석"},
                {"title": "정책 대안", "content_guide": "가능한 정책 대안들을 제시"},
                {"title": "효과 분석", "content_guide": "각 대안의 예상 효과를 분석"},
                {"title": "정책 제언", "content_guide": "최적 정책안과 실행 방안을 제안"}
            ]
        },
        "기술동향보고서": {
            "description": "특정 기술 분야의 현황과 전망을 분석한 보고서",
            "default_outline": [
                {"title": "기술 개요", "content_guide": "기술의 정의와 특징을 설명"},
                {"title": "기술 현황", "content_guide": "현재 기술 개발 수준과 상용화 현황"},
                {"title": "주요 동향", "content_guide": "최신 기술 동향과 혁신 사례"},
                {"title": "시장 현황", "content_guide": "기술 관련 시장 규모와 전망"},
                {"title": "향후 전망", "content_guide": "기술 발전 방향과 미래 전망"}
            ]
        },
        "사업계획서": {
            "description": "사업의 목적과 실행 계획을 담은 보고서",
            "default_outline": [
                {"title": "사업 개요", "content_guide": "사업의 목적, 비전, 목표를 설명"},
                {"title": "시장 분석", "content_guide": "목표 시장과 고객을 분석"},
                {"title": "사업 전략", "content_guide": "차별화 전략과 경쟁 우위를 설명"},
                {"title": "운영 계획", "content_guide": "사업 운영 방식과 조직을 설명"},
                {"title": "재무 계획", "content_guide": "수익 모델과 재무 전망을 제시"}
            ]
        },
        "백서(White Paper)": {
            "description": "특정 주제에 대한 공식적이고 권위있는 문서",
            "default_outline": [
                {"title": "개요", "content_guide": "주제의 배경과 중요성을 설명"},
                {"title": "현황 분석", "content_guide": "현재 상황과 주요 이슈를 분석"},
                {"title": "핵심 내용", "content_guide": "주제의 핵심 내용과 세부사항을 설명"},
                {"title": "사례 연구", "content_guide": "관련 사례와 교훈을 제시"},
                {"title": "향후 방향", "content_guide": "미래 전망과 권고사항을 제시"}
            ]
        }
    }
    
    @staticmethod
    def display_report_generation_tab():
        """Display report generation tab."""
        st.header("📋 보고서 생성")
        
        # Check vector store availability
        if not ReportGenerationUI._check_vector_store():
            return
        
        # Display report generation interface
        ReportGenerationUI._display_report_configuration()
    
    @staticmethod
    def _check_vector_store():
        """Check if vector store is available."""
        vector_store_manager = st.session_state.get("vector_store_manager")
        vector_store = None
        
        if vector_store_manager:
            try:
                vector_store = vector_store_manager.get_vector_store()
            except Exception as e:
                st.warning(f"⚠️ 기존 벡터 스토어 확인 실패: {str(e)}")
                vector_store = None
        
        if vector_store is None:
            st.warning("📋 벡터 스토어가 필요합니다.")
            st.info("**다음 중 하나를 수행하세요:**")
            st.markdown("""
            1. **📚 문서 로딩** 탭에서 문서를 로드한 후 **🔍 벡터 스토어** 탭에서 새 벡터 스토어 생성
            2. **🔍 벡터 스토어** 탭에서 기존에 저장된 벡터 스토어 로딩
            """)
            return False
        
        # Display vector store info
        st.success("✅ 벡터 스토어 준비 완료!")
        return True
    
    @staticmethod
    def _display_report_configuration():
        """Display report configuration interface."""
        st.subheader("📊 보고서 설정")
        
        # Initialize session state for report configuration
        if "report_config" not in st.session_state:
            st.session_state.report_config = {
                "report_type": "연구보고서",
                "topic": "",
                "purpose": "정책 결정 지원",
                "audience": "경영진/의사결정자",
                "language": "한국어",
                "target_length": "medium",
                "include_visuals": False,
                "citation_style": "simple",
                "outline": []
            }
        
        # Main configuration form
        with st.form("report_config_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Report Type
                st.subheader("📝 보고서 유형")
                selected_type = st.selectbox(
                    "보고서 유형을 선택하세요:",
                    options=list(ReportGenerationUI.REPORT_TYPES.keys()),
                    index=list(ReportGenerationUI.REPORT_TYPES.keys()).index(
                        st.session_state.report_config["report_type"]
                    ),
                    help="보고서 유형에 따라 기본 목차가 자동으로 설정됩니다."
                )
                
                # Display description
                if selected_type:
                    st.info(f"💡 {ReportGenerationUI.REPORT_TYPES[selected_type]['description']}")
                
                # Topic
                st.subheader("🎯 주제")
                topic = st.text_input(
                    "보고서의 주제를 입력하세요:",
                    value=st.session_state.report_config["topic"],
                    placeholder="예: 인공지능 산업 동향",
                    help="구체적이고 명확한 주제를 입력하면 더 좋은 보고서가 생성됩니다."
                )
                
                # Purpose
                st.subheader("🎯 목적")
                purpose = st.text_input(
                    "보고서의 목적을 입력하세요:",
                    value=st.session_state.report_config["purpose"],
                    placeholder="예: 정책 결정 지원을 위한 시장 현황 파악"
                )
            
            with col2:
                # Audience
                st.subheader("👥 대상 독자")
                audience = st.text_input(
                    "주요 대상 독자를 입력하세요:",
                    value=st.session_state.report_config["audience"],
                    placeholder="예: 경영진, 정책담당자, 투자자 등"
                )
                
                # Language
                st.subheader("🌐 언어")
                language = st.selectbox(
                    "보고서 작성 언어:",
                    options=["한국어", "영어"],
                    index=["한국어", "영어"].index(st.session_state.report_config["language"])
                )
                
                # Target Length
                st.subheader("📏 분량")
                length_options = {
                    "간단 (5-10페이지)": "short",
                    "보통 (10-20페이지)": "medium", 
                    "상세 (20-30페이지)": "long",
                    "매우 상세 (30페이지 이상)": "very_long"
                }
                
                length_option = st.selectbox(
                    "목표 분량을 선택하세요:",
                    options=list(length_options.keys()),
                    index=list(length_options.values()).index(
                        st.session_state.report_config["target_length"]
                    ),
                    help="분량에 따라 각 섹션의 상세도가 조절됩니다."
                )
                
                # Additional Options
                st.subheader("⚙️ 추가 옵션")
                include_visuals = st.checkbox(
                    "시각 요소 플레이스홀더 포함",
                    value=st.session_state.report_config["include_visuals"],
                    help="차트, 그래프 등을 위한 플레이스홀더를 추가합니다."
                )
                
                citation_style = st.selectbox(
                    "인용 스타일:",
                    options=["simple", "detailed", "none"],
                    index=["simple", "detailed", "none"].index(st.session_state.report_config["citation_style"]),
                    format_func=lambda x: {"simple": "간단", "detailed": "상세", "none": "없음"}[x],
                    help="참고자료 표시 방식을 선택합니다."
                )
            
            # Update session state
            if st.form_submit_button("설정 저장", use_container_width=True):
                st.session_state.report_config.update({
                    "report_type": selected_type,
                    "topic": topic,
                    "purpose": purpose,
                    "audience": audience,
                    "language": language,
                    "target_length": length_options[length_option],
                    "include_visuals": include_visuals,
                    "citation_style": citation_style
                })
                
                # Set default outline based on report type
                if selected_type in ReportGenerationUI.REPORT_TYPES:
                    st.session_state.report_config["outline"] = ReportGenerationUI.REPORT_TYPES[selected_type]["default_outline"].copy()
                
                st.success("✅ 설정이 저장되었습니다!")
                st.rerun()
        
        # Display outline configuration
        ReportGenerationUI._display_outline_configuration()
        
        # Generate report button
        ReportGenerationUI._display_generation_interface()
    
    @staticmethod
    def _display_outline_configuration():
        """Display outline configuration interface."""
        st.subheader("📑 목차 구성")
        
        config = st.session_state.report_config
        
        if not config.get("outline"):
            st.info("먼저 보고서 설정을 저장하여 기본 목차를 불러오세요.")
            return
        
        st.write(f"**{config['report_type']}**의 기본 목차입니다. 필요에 따라 수정하거나 추가할 수 있습니다.")
        
        # Display current outline
        for i, section in enumerate(config["outline"]):
            with st.expander(f"📄 {i+1}. {section['title']}", expanded=False):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    # Edit section title
                    new_title = st.text_input(
                        "섹션 제목:",
                        value=section["title"],
                        key=f"section_title_{i}"
                    )
                    
                    # Edit content guide
                    new_guide = st.text_area(
                        "내용 가이드:",
                        value=section["content_guide"],
                        key=f"section_guide_{i}",
                        help="이 섹션에 어떤 내용이 포함되어야 하는지 설명해주세요."
                    )
                
                with col2:
                    # Delete button
                    if st.button("🗑️ 삭제", key=f"delete_{i}", disabled=len(config["outline"]) <= 1):
                        config["outline"].pop(i)
                        st.rerun()
                
                # Update section info
                config["outline"][i] = {
                    "title": new_title,
                    "content_guide": new_guide
                }
        
        # Add new section
        st.write("---")
        with st.form("add_section_form"):
            st.write("**새 섹션 추가**")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                new_section_title = st.text_input("섹션 제목:", placeholder="예: 시사점")
                new_section_guide = st.text_area("내용 가이드:", placeholder="예: 분석 결과의 시사점을 도출하고 의미를 해석")
            
            with col2:
                if st.form_submit_button("섹션 추가", use_container_width=True):
                    if new_section_title and new_section_guide:
                        config["outline"].append({
                            "title": new_section_title,
                            "content_guide": new_section_guide
                        })
                        st.success(f"✅ '{new_section_title}' 섹션이 추가되었습니다!")
                        st.rerun()
                    else:
                        st.error("섹션 제목과 내용 가이드를 모두 입력해주세요.")
    
    @staticmethod
    def _display_generation_interface():
        """Display report generation interface."""
        st.subheader("🚀 보고서 생성")
        
        config = st.session_state.report_config
        
        # Display current configuration summary
        with st.expander("📋 현재 설정 요약", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**보고서 유형:** {config['report_type']}")
                st.write(f"**주제:** {config.get('topic', 'N/A')}")
                st.write(f"**목적:** {config.get('purpose', 'N/A')}")
                st.write(f"**대상 독자:** {config.get('audience', 'N/A')}")
            
            with col2:
                length_map = {
                    "short": "간단 (5-10페이지)",
                    "medium": "보통 (10-20페이지)",
                    "long": "상세 (20-30페이지)",
                    "very_long": "매우 상세 (30페이지 이상)"
                }
                citation_map = {
                    "simple": "간단",
                    "detailed": "상세",
                    "none": "없음"
                }
                st.write(f"**언어:** {config['language']}")
                st.write(f"**분량:** {length_map.get(config['target_length'], 'N/A')}")
                st.write(f"**시각 요소:** {'포함' if config['include_visuals'] else '미포함'}")
                st.write(f"**인용 스타일:** {citation_map.get(config['citation_style'], 'N/A')}")
            
            st.write(f"**목차 구성:** {len(config.get('outline', []))}개 섹션")
            for i, section in enumerate(config.get('outline', []), 1):
                st.write(f"  {i}. {section['title']}")
        
        # Validation
        ready_to_generate = True
        issues = []
        
        if not config.get('topic'):
            issues.append("주제가 입력되지 않았습니다.")
            ready_to_generate = False
        
        if not config.get('outline'):
            issues.append("목차가 구성되지 않았습니다.")
            ready_to_generate = False
        
        if issues:
            st.error("❌ 다음 사항을 확인해주세요:")
            for issue in issues:
                st.write(f"- {issue}")
        
        # Generate button
        if st.button(
            "📋 보고서 생성하기",
            disabled=not ready_to_generate,
            use_container_width=True,
            type="primary"
        ) and ready_to_generate:
            ReportGenerationUI._generate_report(config)
    
    @staticmethod
    def _generate_report(config: Dict[str, Any]):
        """Generate report based on configuration."""
        try:
            # Initialize RAG system
            vector_store_manager = st.session_state.get("vector_store_manager")
            llm_manager = LLMManager(
                st.session_state.selected_llm_model,
                OLLAMA_BASE_URL,
                temperature=st.session_state.llm_temperature
            )
            
            report_rag = ReportGenerationRAG(vector_store_manager, llm_manager)
            
            # Generate report
            report_content = report_rag.generate_report(config)
            
            # Display generated report
            st.success("✅ 보고서 생성 완료!")
            
            # Show report preview
            with st.expander("📖 보고서 미리보기", expanded=True):
                st.markdown(report_content)
            
            # Download options
            ReportGenerationUI._display_download_options(report_content, config)
            
            # Save to session state
            st.session_state.generated_report = {
                "content": report_content,
                "config": config,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            st.error(f"❌ 보고서 생성 중 오류가 발생했습니다: {str(e)}")
    
    @staticmethod
    def _display_download_options(report_content: str, config: Dict[str, Any]):
        """Display download options for the generated report."""
        st.subheader("💾 다운로드")
        
        # Generate filename
        topic = config.get('topic', '보고서').replace(' ', '_')
        report_type = config.get('report_type', '').replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{topic}_{report_type}_{timestamp}"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Markdown download
            st.download_button(
                label="📄 Markdown 다운로드",
                data=report_content,
                file_name=f"{filename}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col2:
            # HTML download
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{config.get('topic', '보고서')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
        h1, h2, h3 {{ color: #333; }}
        h1 {{ border-bottom: 2px solid #333; }}
        h2 {{ border-bottom: 1px solid #666; }}
        blockquote {{ border-left: 4px solid #ddd; padding-left: 20px; margin: 20px 0; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
{report_content}
</body>
</html>"""
            
            st.download_button(
                label="🌐 HTML 다운로드",
                data=html_content,
                file_name=f"{filename}.html",
                mime="text/html",
                use_container_width=True
            )
        
        with col3:
            # Config download
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            st.download_button(
                label="⚙️ 설정 다운로드",
                data=config_json,
                file_name=f"{filename}_config.json",
                mime="application/json",
                use_container_width=True
            ) 