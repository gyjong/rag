import logging
from typing import List, Dict, Any, TypedDict, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END, START
from langchain_core.runnables import Runnable

# TranslationRAG 클래스 대신 유틸리티 함수들을 직접 임포트
from ..rag_systems import translation_rag as translation_utils
from ..utils.llm_manager import LLMManager
from ..config import OLLAMA_BASE_URL, langfuse_handler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- State Definition ---

class TranslationState(TypedDict):
    """번역 그래프의 상태 정의"""
    # 입력
    original_text: str
    source_lang: str
    target_lang: str
    use_paragraph_mode: bool
    document_title: Optional[str]
    llm_model: str
    temperature: float

    # 중간 상태
    chunks_to_translate: List[str]
    
    # 출력 및 최종 결과
    success: bool
    error: Optional[str]
    translated_text: str
    markdown_content: Optional[str]
    
    # UI 호환성을 위한 결과 필드
    paragraph_pairs: Optional[List[Dict[str, Any]]]
    sentence_pairs: Optional[List[Dict[str, Any]]]
    total_paragraphs: Optional[int]
    total_sentences: Optional[int]
    translation_mode: str
    timestamp: str

# --- Node Functions ---

def split_text_node(state: TranslationState) -> Dict[str, Any]:
    """입력 텍스트를 단락 또는 문장으로 분할합니다."""
    logger.info("--- 1. 텍스트 분할 시작 ---")
    if state["use_paragraph_mode"]:
        chunks = translation_utils.split_into_paragraphs(state["original_text"])
        logger.info(f"{len(chunks)}개의 단락으로 분할되었습니다.")
    else:
        chunks = translation_utils.split_into_sentences(state["original_text"])
        logger.info(f"{len(chunks)}개의 문장으로 분할되었습니다.")
        
    if not chunks:
        return {"success": False, "error": "번역할 내용이 없습니다."}

    return {"chunks_to_translate": chunks, "success": True}


def translate_chunks_node(state: TranslationState) -> Dict[str, Any]:
    """분할된 텍스트 조각들을 번역합니다."""
    logger.info("--- 2. 텍스트 조각 번역 시작 ---")
    llm_manager = LLMManager(state["llm_model"], OLLAMA_BASE_URL, state["temperature"])
    
    chunks = state["chunks_to_translate"]
    translated_items = []
    translated_chunk_details = []

    document_context = ""
    if state["use_paragraph_mode"]:
        document_context = f"문서 제목: {state['document_title']}" if state['document_title'] else ""
        if len(state['original_text']) > 500:
            first_chunk = chunks[0][:200] + "..." if len(chunks[0]) > 200 else chunks[0]
            document_context += f"\\n문서 요약: {first_chunk}"

    for i, chunk in enumerate(chunks):
        logger.info(f"번역 중: {i+1}/{len(chunks)}")
        
        if len(chunk.strip()) <= 10:
            translation = chunk
            skipped = True
        else:
            if state["use_paragraph_mode"]:
                translation = translation_utils.translate_paragraph(
                    llm_manager, chunk, state["source_lang"], state["target_lang"], document_context
                )
            else:
                translation = translation_utils.translate_sentence(
                    llm_manager, chunk, state["source_lang"], state["target_lang"]
                )
            skipped = False

        translated_items.append(translation)
        translated_chunk_details.append({"original": chunk, "translated": translation, "skipped": skipped})

    separator = '\\n\\n' if state["use_paragraph_mode"] else '\\n'
    translated_text = separator.join(translated_items)
    
    output = {"translated_text": translated_text}
    if state["use_paragraph_mode"]:
        output["paragraph_pairs"] = translated_chunk_details
        output["total_paragraphs"] = len(chunks)
    else:
        output["sentence_pairs"] = translated_chunk_details
        output["total_sentences"] = len(chunks)

    return output

def format_output_node(state: TranslationState) -> Dict[str, Any]:
    """번역된 텍스트를 마크다운 형식으로 구성합니다."""
    logger.info("--- 3. 마크다운 포맷팅 시작 ---")
    llm_manager = LLMManager(state["llm_model"], OLLAMA_BASE_URL, state["temperature"])

    markdown_content = translation_utils.organize_and_format_translation(
        llm_manager, state["translated_text"], state["document_title"], state["target_lang"]
    )
    return {"markdown_content": markdown_content}

# --- Conditional Edge ---

def decide_to_format(state: TranslationState) -> str:
    """번역 후 마크다운 포맷팅을 진행할지 결정합니다."""
    logger.info("--- 포맷팅 필요 여부 확인 ---")
    if state["use_paragraph_mode"]:
        logger.info("결정: 포맷팅 진행")
        return "format"
    else:
        logger.info("결정: 종료")
        return "end"

# --- Graph Definition ---

def create_translation_graph() -> Runnable:
    """번역 RAG 워크플로우를 정의하고 컴파일합니다."""
    workflow = StateGraph(TranslationState)

    workflow.add_node("split", split_text_node)
    workflow.add_node("translate", translate_chunks_node)
    workflow.add_node("format", format_output_node)

    workflow.add_edge(START, "split")
    workflow.add_edge("split", "translate")
    workflow.add_conditional_edges(
        "translate",
        decide_to_format,
        {"format": "format", "end": END}
    )
    workflow.add_edge("format", END)
    
    graph = workflow.compile()
    logger.info("번역 그래프가 성공적으로 컴파일되었습니다.")
    return graph

# --- Execution Function ---

def run_translation_graph(**kwargs) -> Dict[str, Any]:
    """번역 그래프를 실행하고 최종 결과를 반환합니다."""
    graph = create_translation_graph()
    
    initial_state = {
        "original_text": kwargs.get("text"),
        "source_lang": kwargs.get("source_lang"),
        "target_lang": kwargs.get("target_lang"),
        "use_paragraph_mode": kwargs.get("use_paragraph_mode"),
        "document_title": kwargs.get("document_title"),
        "llm_model": kwargs.get("llm_model"),
        "temperature": kwargs.get("temperature"),
        "success": True,
        "error": None,
        "translated_text": "",
        "markdown_content": None,
        "paragraph_pairs": [],
        "sentence_pairs": [],
        "total_paragraphs": 0,
        "total_sentences": 0,
        "translation_mode": "",
        "timestamp": ""
    }

    final_state = graph.invoke(initial_state, config={"callbacks": [langfuse_handler]})

    final_result = {**final_state}
    final_result.update({
        "success": final_state.get("success", True),
        "source_language": final_state.get("source_lang"),
        "target_language": final_state.get("target_lang"),
        "translation_mode": "paragraph" if final_state.get("use_paragraph_mode") else "sentence",
        "timestamp": datetime.now().isoformat()
    })

    return final_result 