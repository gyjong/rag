"""LangGraph implementation for JSON-based RAG services."""
import logging
from typing import TypedDict, List, Dict, Any, Literal, Optional

from langgraph.graph import StateGraph, END
from ..rag_systems import json_rag
from ..utils.llm_manager import LLMManager

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JsonRagState(TypedDict):
    """JSON RAG 그래프의 상태를 정의합니다."""
    query: str
    service_type: Literal["bus", "menu"]
    bus_processor: Optional[json_rag.JSONDataProcessor]
    menu_processor: Optional[json_rag.JSONDataProcessor]
    search_results: List[Dict[str, Any]]
    context: str
    answer: str
    error: Optional[str]

# --- Graph Nodes ---

def search_node(state: JsonRagState) -> Dict[str, Any]:
    """쿼리와 서비스 유형에 따라 적절한 JSON 데이터 소스를 검색합니다."""
    logger.info(f"--- 1. '{state['service_type']}' 서비스 검색 시작 ---")
    try:
        if state["service_type"] == "bus":
            if not state['bus_processor']:
                return {"error": "버스 데이터 프로세서가 초기화되지 않았습니다."}
            results = json_rag.search_bus_routes_utility(state["bus_processor"], state["query"])
        elif state["service_type"] == "menu":
            if not state['menu_processor']:
                return {"error": "메뉴 데이터 프로세서가 초기화되지 않았습니다."}
            results = json_rag.search_menu_items_utility(state["menu_processor"], state["query"])
        else:
            return {"error": f"알 수 없는 서비스 유형입니다: {state['service_type']}"}
        
        logger.info(f"검색 완료: {len(results)}개 결과.")
        return {"search_results": results, "error": None}
    except Exception as e:
        logger.error(f"검색 노드에서 오류 발생: {e}", exc_info=True)
        return {"error": f"검색 중 오류 발생: {e}"}

def format_context_node(state: JsonRagState) -> Dict[str, Any]:
    """검색 결과를 LLM이 사용할 컨텍스트로 포맷팅합니다."""
    logger.info("--- 2. 컨텍스트 포맷팅 시작 ---")
    if state.get("error"):
        return {}

    search_results = state.get("search_results", [])
    if not search_results:
        logger.info("컨텍스트를 생성할 검색 결과가 없습니다.")
        return {"context": ""}

    try:
        if state["service_type"] == "bus":
            context = json_rag.format_bus_context(search_results)
        elif state["service_type"] == "menu":
            context = json_rag.format_menu_context(search_results)
        else:
            # 이 경우는 발생하지 않아야 함
            return {"error": f"컨텍스트 포맷팅 중 알 수 없는 서비스 유형 발견: {state['service_type']}"}
        
        logger.info(f"컨텍스트 생성 완료 (길이: {len(context)}).")
        return {"context": context}
    except Exception as e:
        logger.error(f"컨텍스트 포맷팅 중 오류 발생: {e}", exc_info=True)
        return {"error": f"컨텍스트 생성 중 오류 발생: {e}"}


def generate_answer_node(state: JsonRagState, llm_manager: LLMManager) -> Dict[str, Any]:
    """컨텍스트와 쿼리를 기반으로 최종 답변을 스트리밍으로 생성합니다."""
    logger.info("--- 3. 최종 답변 생성 시작 ---")
    if state.get("error"):
        return {"answer": state["error"]}

    query = state["query"]
    context = state["context"]
    service_type = state["service_type"]
    data_type = "버스 정보" if service_type == "bus" else "메뉴 정보"

    try:
        answer_stream = json_rag.generate_answer_stream(llm_manager, query, context, data_type)
        
        # 스트림을 소비하여 전체 답변을 구성
        full_answer = "".join(list(answer_stream))
        
        logger.info("최종 답변 생성 완료.")
        return {"answer": full_answer}
    except Exception as e:
        logger.error(f"답변 생성 노드에서 오류 발생: {e}", exc_info=True)
        return {"answer": f"답변 생성 중 오류 발생: {e}"}

# --- Graph Logic ---

def should_continue(state: JsonRagState) -> Literal["format_context", "generate_answer", "__end__"]:
    """다음에 실행할 노드를 결정합니다."""
    if state.get("error"):
        logger.warning(f"오류 발생으로 그래프 실행 중단: {state['error']}")
        return "__end__"
    
    if not state.get("search_results"):
        logger.info("검색 결과가 없으므로 답변 생성으로 바로 이동합니다.")
        return "generate_answer"
        
    return "format_context"


def create_json_rag_graph(llm_manager: LLMManager) -> StateGraph:
    """JSON RAG 서비스용 LangGraph 워크플로우를 생성하고 컴파일합니다."""
    workflow = StateGraph(JsonRagState)

    # 노드 추가
    workflow.add_node("search", search_node)
    workflow.add_node("format_context", format_context_node)
    workflow.add_node("generate_answer", lambda state: generate_answer_node(state, llm_manager))

    # 엣지 및 진입점 설정
    workflow.set_entry_point("search")
    workflow.add_conditional_edges(
        "search",
        should_continue,
        {
            "format_context": "format_context",
            "generate_answer": "generate_answer",
            "__end__": END
        }
    )
    workflow.add_edge("format_context", "generate_answer")
    workflow.add_edge("generate_answer", END)

    # 그래프 컴파일
    graph = workflow.compile()
    logger.info("JSON RAG 그래프가 성공적으로 컴파일되었습니다.")
    return graph 