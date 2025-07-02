"""Utility functions for Report Generation RAG."""

from typing import List, Dict, Any
from langchain_core.documents import Document
from ..utils.vector_store import VectorStoreManager
from ..utils.llm_manager import LLMManager
from datetime import datetime

def retrieve_for_topic(vector_store_manager: VectorStoreManager, topic: str, k: int = 10) -> List[Document]:
    """Retrieve relevant documents for a specific topic."""
    vector_store = vector_store_manager.get_vector_store()
    if not vector_store:
        return []
        
    search_queries = [topic, f"{topic} 동향", f"{topic} 분석", f"{topic} 현황", f"{topic} 전망"]
    all_docs = []
    seen_content = set()
    
    for query in search_queries:
        try:
            docs = vector_store.similarity_search(query, k=k//len(search_queries)+1)
            for doc in docs:
                content_hash = hash(doc.page_content[:500])
                if content_hash not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(content_hash)
        except Exception:
            continue
    return all_docs[:k]

def generate_report_section(
    llm_manager: LLMManager,
    section_title: str,
    section_content_guide: str,
    context_docs: List[Document],
    report_config: Dict[str, Any]
) -> str:
    """Generate a specific section of the report."""
    context = "\\n\\n".join([f"문서: {doc.metadata.get('source', 'Unknown')}\\n내용: {doc.page_content}" for doc in context_docs])
    context = context[:6000] + "..." if len(context) > 6000 else context

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
1. 제공된 참고 문서의 내용을 바탕으로 작성하세요.
2. 마크다운 형식으로 작성하세요.
3. 대상 독자에 맞는 적절한 언어와 설명 수준을 사용하세요.
4. 구체적인 데이터나 사실이 있으면 포함하세요.
5. 참고한 문서가 있으면 내용 말미에 간단히 언급하세요.

**출력 형식:**
## {section_title}

[섹션 내용을 여기에 작성]

섹션 내용만 작성하고, 추가 설명은 하지 마세요."""

    return llm_manager.generate_response(prompt=prompt, context="")

def generate_report_header(report_config: Dict[str, Any]) -> str:
    """Generate report header with title and metadata."""
    report_type = report_config.get('report_type', '보고서')
    topic = report_config.get('topic', '')
    purpose = report_config.get('purpose', '')
    current_date = datetime.now().strftime("%Y년 %m월 %d일")
    
    return f"""# {topic} {report_type}

**작성일:** {current_date}  
**목적:** {purpose}  
**대상 독자:** {report_config.get('audience', '일반인')}  

---

## 개요

본 보고서는 {topic}에 대한 {report_type.lower()}로, {purpose}를 목적으로 작성되었습니다.
"""

def generate_conclusion(llm_manager: LLMManager, context_docs: List[Document], report_config: Dict[str, Any]) -> str:
    """Generate conclusion section."""
    context = "\\n\\n".join([doc.page_content for doc in context_docs[:5]])
    context = context[:3000] + "..." if len(context) > 3000 else context
    
    prompt = f"""다음 정보를 바탕으로 보고서의 결론 섹션을 작성해주세요.

**보고서 정보:**
- 주제: {report_config.get('topic', 'N/A')}
- 목적: {report_config.get('purpose', 'N/A')}
- 보고서 유형: {report_config.get('report_type', 'N/A')}

**참고 문서:**
{context}

**요구사항:**
1. 보고서의 주요 내용을 요약하세요.
2. 핵심 발견사항이나 결과를 제시하세요.
3. 향후 전망이나 제언을 포함하세요.
4. 마크다운 형식으로 작성하세요.

**출력 형식:**
## 결론

[결론 내용을 여기에 작성]
"""
    return llm_manager.generate_response(prompt=prompt, context="")

def generate_references(docs: List[Document], citation_style: str) -> str:
    """Generate references section."""
    if not docs or citation_style == 'none':
        return ""
    
    references = "## 참고자료\\n\\n"
    sources = sorted({doc.metadata.get('source', 'Unknown') for doc in docs if doc.metadata.get('source') != 'Unknown'})
    
    if citation_style == 'detailed':
        references += "\\n".join(f"{i}. {source}" for i, source in enumerate(sources, 1))
    else: # simple
        references += "본 보고서는 다음 문서들을 참고하여 작성되었습니다:\\n\\n"
        references += "\\n".join(f"- {source}" for source in sources)
    
    return references

def add_visual_placeholders(content: str) -> str:
    """Add placeholders for visual elements."""
    visual_sections = ['현황', '분석', '동향', '통계', '데이터']
    for section in visual_sections:
        if f"## {section}" in content:
            content = content.replace(f"## {section}", f"## {section}\\n\\n*[여기에 {section} 관련 차트나 그래프 삽입]*\\n")
    return content 