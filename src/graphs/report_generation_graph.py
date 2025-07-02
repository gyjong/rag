from typing import List, Dict, Any, TypedDict, Optional, Generator
from langgraph.graph import StateGraph, END, START
from langchain_core.documents import Document

from ..rag_systems import report_generation_rag as report_utils
from ..utils.llm_manager import LLMManager
from ..utils.vector_store import VectorStoreManager
from ..config import OLLAMA_BASE_URL
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Graph State Definition ---
class ReportGenerationState(TypedDict):
    report_config: Dict[str, Any]
    vector_store_manager: VectorStoreManager
    main_docs: Optional[List[Document]]
    report_draft: Optional[str]
    validation_feedback: Optional[str]
    final_report: Optional[str]
    process_steps: List[str]
    generated_sections: List[str]

# --- Node Functions ---
def initial_doc_retrieval_node(state: ReportGenerationState) -> Dict[str, Any]:
    """Retrieve initial documents for the main topic."""
    logger.info("--- Node: Initial Document Retrieval ---")
    state['process_steps'].append("1. 주제 관련 문서 검색 중...")
    
    topic = state['report_config']['topic']
    main_docs = report_utils.retrieve_for_topic(state['vector_store_manager'], topic, k=20)
    
    logger.info(f"Retrieved {len(main_docs)} documents.")
    return {"main_docs": main_docs, "process_steps": state['process_steps']}

def generate_draft_node(state: ReportGenerationState) -> Generator[Dict[str, Any], None, None]:
    """Generate the draft of the full report, streaming section by section and token by token."""
    logger.info("--- Node: Report Draft Generation (Streaming) ---")
    state['process_steps'].append("2. 보고서 초안 생성 중...")
    
    config = state['report_config']
    main_docs = state['main_docs']
    llm_manager = LLMManager(config.get('llm_model'), OLLAMA_BASE_URL, config.get('temperature'))
    
    # 1. Generate and stream header
    report_draft = report_utils.generate_report_header(config)
    yield {"report_draft": report_draft}

    # 2. Generate and stream each section
    for i, section_info in enumerate(config['outline']):
        title, guide = section_info['title'], section_info['content_guide']
        
        current_step = f"  - {i+1}/{len(config['outline'])} 섹션 생성 중: {title}"
        if current_step not in state['process_steps']:
            state['process_steps'].append(current_step)
        yield {"process_steps": state['process_steps']}
        
        logger.info(current_step)
        
        # Add section title to the draft
        report_draft += f"\\n\\n## {title}\\n\\n"
        yield {"report_draft": report_draft}

        section_docs = report_utils.retrieve_for_topic(state['vector_store_manager'], f"{config['topic']} {title}", k=5)
        context_docs = section_docs or main_docs[:5]
        
        # Stream the section content token by token
        for token in report_utils.generate_report_section(llm_manager, title, guide, context_docs, config):
            report_draft += token
            yield {"report_draft": report_draft}

    # 3. Generate and stream conclusion
    state['process_steps'].append("  - 결론 생성 중...")
    yield {"process_steps": state['process_steps']}
    logger.info("  - Generating conclusion...")
    report_draft += "\\n\\n## 결론\\n\\n"
    yield {"report_draft": report_draft}
    conclusion_stream = report_utils.generate_conclusion(llm_manager, main_docs, config) # Assuming this also streams
    for token in conclusion_stream:
        report_draft += token
        yield {"report_draft": report_draft}

    # 4. Generate and stream references
    state['process_steps'].append("  - 참고자료 생성 중...")
    yield {"process_steps": state['process_steps']}
    logger.info("  - Generating references...")
    references = report_utils.generate_references(main_docs, config.get('citation_style'))
    if references:
        report_draft += "\\n\\n" + references
    
    yield {"report_draft": report_draft, "process_steps": state['process_steps']}

def validate_report_node(state: ReportGenerationState) -> Dict[str, Any]:
    """Validate the generated report draft against requirements."""
    logger.info("--- Node: Report Validation ---")
    state['process_steps'].append("3. 생성된 보고서 검증 중...")
    
    config = state['report_config']
    draft = state['report_draft']
    llm_manager = LLMManager(config.get('llm_model'), OLLAMA_BASE_URL, config.get('temperature'))

    prompt = f"""
다음 보고서 초안이 아래의 요구사항을 잘 만족하는지 검토하고, 개선점을 제시해주세요.

**요구사항:**
- 주제: {config.get('topic')}
- 목적: {config.get('purpose')}
- 대상 독자: {config.get('audience')}
- 목차: {[s['title'] for s in config.get('outline', [])]}
- 분량: {config.get('target_length')}
- 언어: {config.get('language')}

**보고서 초안:**
---
{draft[:4000]}...
---

**검토 결과 (개선점 위주로 간결하게):**
"""
    feedback = llm_manager.generate_response(prompt, "")
    logger.info(f"Validation Feedback: {feedback}")
    
    return {"validation_feedback": feedback, "process_steps": state['process_steps']}

def finalise_report_node(state: ReportGenerationState) -> Dict[str, Any]:
    """Finalises the report by revising the draft based on validation feedback."""
    logger.info("--- Node: Revising and Finalising Report ---")
    state['process_steps'].append("4. 검증 결과 기반으로 보고서 수정 중...")
    
    config = state['report_config']
    draft = state['report_draft']
    feedback = state['validation_feedback']
    
    llm_manager = LLMManager(config.get('llm_model'), OLLAMA_BASE_URL, config.get('temperature'))

    # Create a prompt for revision
    revision_prompt = f"""
당신은 전문 편집자입니다. 아래의 보고서 초안과 검토 피드백을 바탕으로, 보고서를 최종 완성본으로 수정해주세요.
피드백을 반영하여 완성도를 높이고, 자연스러운 흐름을 갖는 전체 보고서를 다시 작성해야 합니다.

**기존 보고서 초안:**
---
{draft}
---

**검토 피드백 및 개선 요청사항:**
---
{feedback}
---

**수정된 최종 보고서 (피드백을 모두 반영하여 완성된 Full-Text):**
"""

    # Generate the revised report
    revised_report = llm_manager.generate_response(revision_prompt, "")
    logger.info("Report has been revised based on feedback.")
    
    final_report = revised_report
    
    # Optionally add visual placeholders
    if config.get('include_visuals'):
        final_report = report_utils.add_visual_placeholders(final_report)
    
    # Append the validation summary for transparency
    final_report += f"""

---
### **보고서 자동 검증 요약**
{feedback}
"""
    state['process_steps'].append("✅ 보고서 생성 완료!")
    return {"final_report": final_report, "process_steps": state['process_steps']}

# --- Graph Definition ---
def create_report_generation_graph():
    workflow = StateGraph(ReportGenerationState)
    workflow.add_node("initial_retrieval", initial_doc_retrieval_node)
    workflow.add_node("generate_draft", generate_draft_node)
    workflow.add_node("validate_report", validate_report_node)
    workflow.add_node("finalise_report", finalise_report_node)
    
    workflow.add_edge(START, "initial_retrieval")
    workflow.add_edge("initial_retrieval", "generate_draft")
    workflow.add_edge("generate_draft", "validate_report")
    workflow.add_edge("validate_report", "finalise_report")
    workflow.add_edge("finalise_report", END)
    
    return workflow.compile()

def run_report_generation_graph(report_config: Dict[str, Any], vector_store_manager: VectorStoreManager):
    graph = create_report_generation_graph()
    initial_state = {
        "report_config": report_config,
        "vector_store_manager": vector_store_manager,
        "process_steps": [],
        "generated_sections": [],
    }
    return graph.stream(initial_state) 