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
    """Generate the draft of the full report, streaming section by section."""
    logger.info("--- Node: Report Draft Generation ---")
    state['process_steps'].append("2. 보고서 초안 생성 중...")
    
    config = state['report_config']
    main_docs = state['main_docs']
    llm_manager = LLMManager(config.get('llm_model'), OLLAMA_BASE_URL, config.get('temperature'))
    
    # Generate and yield header
    report_content = report_utils.generate_report_header(config)
    yield {"report_draft": report_content}

    # Generate and yield sections
    all_sections_content = []
    for i, section_info in enumerate(config['outline']):
        title, guide = section_info['title'], section_info['content_guide']
        
        current_step = f"  - {i+1}/{len(config['outline'])} 섹션 생성 중: {title}"
        if current_step not in state['process_steps']:
            state['process_steps'].append(current_step)

        logger.info(current_step)
        
        section_docs = report_utils.retrieve_for_topic(state['vector_store_manager'], f"{config['topic']} {title}", k=5)
        context_docs = section_docs or main_docs[:5]
        
        section_content = report_utils.generate_report_section(llm_manager, title, guide, context_docs, config)
        all_sections_content.append(section_content)

        # Yield the updated draft
        current_draft = report_content + "\\n\\n" + "\\n\\n---\\n\\n".join(all_sections_content)
        yield {"report_draft": current_draft, "process_steps": state['process_steps']}
    
    # Combine for final draft before conclusion
    report_content += "\\n\\n" + "\\n\\n---\\n\\n".join(all_sections_content)

    # Generate and yield conclusion
    state['process_steps'].append("  - 결론 생성 중...")
    logger.info("  - Generating conclusion...")
    conclusion = report_utils.generate_conclusion(llm_manager, main_docs, config)
    report_content += "\\n\\n---\\n\\n" + conclusion
    yield {"report_draft": report_content, "process_steps": state['process_steps']}

    # Generate and yield references
    state['process_steps'].append("  - 참고자료 생성 중...")
    logger.info("  - Generating references...")
    references = report_utils.generate_references(main_docs, config.get('citation_style'))
    if references:
        report_content += "\\n\\n---\\n\\n" + references
    
    # The final return for the node state.
    yield {"report_draft": report_content, "process_steps": state['process_steps']}

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
    """Finalise the report. For now, it just passes the draft through."""
    logger.info("--- Node: Finalising Report ---")
    state['process_steps'].append("4. 보고서 최종화 중...")
    
    draft = state['report_draft']
    feedback = state['validation_feedback']
    
    final_report = draft
    if config := state.get('report_config', {}).get('include_visuals'):
        final_report = report_utils.add_visual_placeholders(final_report)
    
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