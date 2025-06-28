"""Report Generation RAG implementation."""

from typing import List, Dict, Any, Optional
import time
import streamlit as st
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM

from ..utils.vector_store import VectorStoreManager
from ..utils.llm_manager import LLMManager


class ReportGenerationRAG:
    """Report Generation RAG implementation for structured report creation."""
    
    def __init__(self, vector_store_manager: VectorStoreManager, llm_manager: LLMManager):
        """Initialize Report Generation RAG system.
        
        Args:
            vector_store_manager: Vector store manager instance
            llm_manager: LLM manager instance
        """
        self.vector_store_manager = vector_store_manager
        self.llm_manager = llm_manager
        self.name = "Report Generation RAG"
        self.description = "구조화된 보고서 생성을 위한 전문 RAG 시스템"
        
    def retrieve_for_topic(self, topic: str, k: int = 10) -> List[Document]:
        """Retrieve relevant documents for a specific topic.
        
        Args:
            topic: Topic to search for
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        vector_store = self.vector_store_manager.get_vector_store()
        if vector_store is None:
            st.error("벡터 스토어가 초기화되지 않았습니다.")
            return []
            
        try:
            # Create multiple search queries for better coverage
            search_queries = [
                topic,
                f"{topic} 동향",
                f"{topic} 분석",
                f"{topic} 현황",
                f"{topic} 전망"
            ]
            
            all_docs = []
            seen_content = set()
            
            for query in search_queries:
                try:
                    docs = vector_store.similarity_search(query, k=k//len(search_queries)+1)
                    for doc in docs:
                        # Remove duplicates based on content
                        content_hash = hash(doc.page_content[:500])
                        if content_hash not in seen_content:
                            all_docs.append(doc)
                            seen_content.add(content_hash)
                except Exception as e:
                    st.warning(f"검색 쿼리 '{query}' 실행 중 오류: {str(e)}")
                    continue
            
            # Sort by relevance and return top k
            return all_docs[:k]
            
        except Exception as e:
            st.error(f"문서 검색 실패: {str(e)}")
            return []
    
    def generate_report_section(self, section_title: str, section_content_guide: str, 
                              context_docs: List[Document], report_config: Dict[str, Any]) -> str:
        """Generate a specific section of the report.
        
        Args:
            section_title: Title of the section
            section_content_guide: Guide for what to include in this section
            context_docs: Relevant documents for this section
            report_config: Configuration for the report
            
        Returns:
            Generated section content in markdown format
        """
        if not context_docs:
            context = "관련 문서가 없습니다."
        else:
            context = "\n\n".join([f"문서: {doc.metadata.get('source', 'Unknown')}\n내용: {doc.page_content}" 
                                 for doc in context_docs])
        
        # Limit context length
        max_context_length = 6000
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        # Create section generation prompt
        prompt = f"""당신은 전문적인 보고서 작성자입니다. 다음 정보를 바탕으로 보고서의 한 섹션을 작성해주세요.

**보고서 정보:**
- 보고서 유형: {report_config.get('report_type', 'N/A')}
- 전체 주제: {report_config.get('topic', 'N/A')}
- 목적: {report_config.get('purpose', 'N/A')}
- 대상 독자: {report_config.get('audience', 'N/A')}
- 언어: {report_config.get('language', '한국어')}

**현재 작성할 섹션:**
- 섹션 제목: {section_title}
- 섹션 내용 가이드: {section_content_guide}

**참고 문서:**
{context}

**작성 지침:**
1. 제공된 참고 문서의 내용을 바탕으로 작성하세요
2. 마크다운 형식으로 작성하세요
3. 대상 독자에 맞는 적절한 언어와 설명 수준을 사용하세요
4. 구체적인 데이터나 사실이 있으면 포함하세요
5. 필요시 하위 섹션을 만들어 구조화하세요
6. 참고한 문서가 있으면 내용 말미에 간단히 언급하세요

**출력 형식:**
## {section_title}

[섹션 내용을 여기에 작성]

섹션 내용만 작성하고, 추가 설명은 하지 마세요."""

        try:
            # Generate section content
            section_content = ""
            for chunk in self.llm_manager.generate_response_stream(
                prompt=prompt,
                context=""  # Context is already included in prompt
            ):
                section_content += chunk
            
            return section_content.strip()
            
        except Exception as e:
            st.error(f"섹션 '{section_title}' 생성 실패: {str(e)}")
            return f"## {section_title}\n\n섹션 생성 중 오류가 발생했습니다."
    
    def generate_report(self, report_config: Dict[str, Any]) -> str:
        """Generate a complete report based on configuration.
        
        Args:
            report_config: Configuration dictionary containing report parameters
            
        Returns:
            Generated report in markdown format
        """
        start_time = time.time()
        
        # Extract configuration
        report_type = report_config.get('report_type', '연구보고서')
        topic = report_config.get('topic', '')
        purpose = report_config.get('purpose', '')
        outline = report_config.get('outline', [])
        target_length = report_config.get('target_length', 'medium')
        audience = report_config.get('audience', '일반인')
        language = report_config.get('language', '한국어')
        include_visuals = report_config.get('include_visuals', False)
        citation_style = report_config.get('citation_style', 'simple')
        
        st.subheader("📊 보고서 생성 진행 상황")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Retrieve relevant documents for the main topic
        status_text.text("1. 주제 관련 문서 검색 중...")
        progress_bar.progress(10)
        
        main_docs = self.retrieve_for_topic(topic, k=20)
        
        if not main_docs:
            st.warning("관련 문서를 찾을 수 없습니다. 일반적인 보고서 구조로 생성합니다.")
        
        # Step 2: Generate report header
        status_text.text("2. 보고서 헤더 생성 중...")
        progress_bar.progress(20)
        
        report_content = self._generate_report_header(report_config)
        
        # Step 3: Generate each section
        total_sections = len(outline)
        if total_sections == 0:
            st.error("목차가 비어있습니다.")
            return "목차가 정의되지 않았습니다."
        
        for i, section_info in enumerate(outline):
            section_title = section_info.get('title', f'섹션 {i+1}')
            section_guide = section_info.get('content_guide', '이 섹션의 내용을 작성하세요.')
            
            status_text.text(f"3. 섹션 생성 중: {section_title}")
            progress = 20 + (i + 1) * 60 // total_sections
            progress_bar.progress(progress)
            
            # Get relevant documents for this specific section
            section_docs = self.retrieve_for_topic(f"{topic} {section_title}", k=10)
            
            # Use main docs if no specific docs found
            if not section_docs:
                section_docs = main_docs[:5]
            
            # Generate section content
            section_content = self.generate_report_section(
                section_title, section_guide, section_docs, report_config
            )
            
            report_content += "\n\n" + section_content
        
        # Step 4: Generate conclusion and references
        status_text.text("4. 결론 및 참고자료 생성 중...")
        progress_bar.progress(90)
        
        conclusion = self._generate_conclusion(topic, purpose, main_docs, report_config)
        report_content += "\n\n" + conclusion
        
        # Add references if requested
        if citation_style != 'none':
            references = self._generate_references(main_docs, citation_style)
            report_content += "\n\n" + references
        
        # Step 5: Final formatting
        status_text.text("5. 최종 포맷팅 중...")
        progress_bar.progress(100)
        
        # Add visual elements placeholder if requested
        if include_visuals:
            report_content = self._add_visual_placeholders(report_content)
        
        total_time = time.time() - start_time
        status_text.text(f"보고서 생성 완료! (소요 시간: {total_time:.1f}초)")
        
        return report_content
    
    def _generate_report_header(self, report_config: Dict[str, Any]) -> str:
        """Generate report header with title and metadata."""
        report_type = report_config.get('report_type', '보고서')
        topic = report_config.get('topic', '')
        purpose = report_config.get('purpose', '')
        
        from datetime import datetime
        current_date = datetime.now().strftime("%Y년 %m월 %d일")
        
        header = f"""# {topic} {report_type}

**작성일:** {current_date}  
**목적:** {purpose}  
**대상 독자:** {report_config.get('audience', '일반인')}  

---

## 개요

본 보고서는 {topic}에 대한 {report_type.lower()}로, {purpose}를 목적으로 작성되었습니다.
"""
        
        return header
    
    def _generate_conclusion(self, topic: str, purpose: str, context_docs: List[Document], 
                           report_config: Dict[str, Any]) -> str:
        """Generate conclusion section."""
        context = "\n\n".join([doc.page_content for doc in context_docs[:5]])
        max_context_length = 3000
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        prompt = f"""다음 정보를 바탕으로 보고서의 결론 섹션을 작성해주세요.

**보고서 정보:**
- 주제: {topic}
- 목적: {purpose}
- 보고서 유형: {report_config.get('report_type', 'N/A')}

**참고 문서:**
{context}

**요구사항:**
1. 보고서의 주요 내용을 요약하세요
2. 핵심 발견사항이나 결과를 제시하세요
3. 향후 전망이나 제언을 포함하세요
4. 마크다운 형식으로 작성하세요

**출력 형식:**
## 결론

[결론 내용을 여기에 작성]

### 주요 발견사항
- [발견사항 1]
- [발견사항 2]
- [발견사항 3]

### 향후 전망 및 제언
[전망 및 제언 내용]
"""
        
        try:
            conclusion = ""
            for chunk in self.llm_manager.generate_response_stream(
                prompt=prompt,
                context=""
            ):
                conclusion += chunk
            
            return conclusion.strip()
            
        except Exception as e:
            st.error(f"결론 생성 실패: {str(e)}")
            return "## 결론\n\n결론 생성 중 오류가 발생했습니다."
    
    def _generate_references(self, docs: List[Document], citation_style: str) -> str:
        """Generate references section."""
        if not docs:
            return ""
        
        references = "## 참고자료\n\n"
        
        # Get unique sources
        sources = set()
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            if source != 'Unknown':
                sources.add(source)
        
        if citation_style == 'detailed':
            for i, source in enumerate(sorted(sources), 1):
                references += f"{i}. {source}\n"
        else:  # simple
            references += "본 보고서는 다음 문서들을 참고하여 작성되었습니다:\n\n"
            for source in sorted(sources):
                references += f"- {source}\n"
        
        return references
    
    def _add_visual_placeholders(self, content: str) -> str:
        """Add placeholders for visual elements."""
        # Add chart placeholders after certain sections
        visual_sections = ['현황', '분석', '동향', '통계', '데이터']
        
        for section in visual_sections:
            if section in content:
                content = content.replace(
                    f"## {section}",
                    f"## {section}\n\n*[여기에 {section} 관련 차트나 그래프 삽입]*\n"
                )
        
        return content
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the Report Generation RAG system."""
        return {
            "name": self.name,
            "description": self.description,
            "components": [
                "Vector Store (Multi-query Search)",
                "LLM (Structured Generation)",
                "Report Formatter",
                "Citation Manager"
            ],
            "features": [
                "구조화된 보고서 생성",
                "사용자 정의 목차 지원",
                "다양한 보고서 유형 지원",
                "자동 인용 및 참고자료 생성",
                "마크다운 형식 출력",
                "시각 요소 플레이스홀더"
            ],
            "limitations": [
                "LLM 토큰 길이 제한",
                "복잡한 수식이나 표 생성 제한",
                "실시간 데이터 업데이트 불가"
            ]
        } 