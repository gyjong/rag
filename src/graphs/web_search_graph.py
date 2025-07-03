from typing import List, Dict, Any, TypedDict, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, END, START
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..rag_systems import web_search_rag as web_search_utils
from ..utils.llm_manager import LLMManager
from ..config import OLLAMA_BASE_URL, langfuse_handler
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Graph State Definition ---
class WebSearchState(TypedDict):
    """웹 검색 RAG 그래프의 상태"""
    original_query: str
    num_results: int
    llm_model: str
    temperature: float
    refined_queries: Optional[Dict[str, str]]
    search_results: Optional[List[Dict[str, Any]]]
    final_answer: Optional[str]
    process_steps: List[str]  # 진행 상황을 추적하기 위한 리스트

# --- Node Functions ---

def refine_query_node(state: WebSearchState) -> Dict[str, Any]:
    """사용자 쿼리를 정제하여 검색에 최적화된 쿼리를 생성합니다."""
    logger.info("--- 1. 쿼리 정제 시작 ---")
    state['process_steps'].append("1. 쿼리 정제 중...")
    llm_manager = LLMManager(state["llm_model"], OLLAMA_BASE_URL, state["temperature"])
    refined_queries = web_search_utils.refine_query(llm_manager, state["original_query"])
    logger.info(f"정제된 쿼리: {refined_queries}")
    return {"refined_queries": refined_queries, "process_steps": state['process_steps']}

def search_web_node(state: WebSearchState) -> Dict[str, Any]:
    """정제된 쿼리를 사용하여 웹을 검색합니다."""
    logger.info("--- 2. 웹 검색 시작 ---")
    state['process_steps'].append("2. 웹 검색 실행 중...")
    query_to_search = state["refined_queries"].get("korean") or state["refined_queries"].get("english", state["original_query"])
    search_results = web_search_utils.search_web(query_to_search, state["num_results"])
    logger.info(f"{len(search_results)}개의 검색 결과 발견.")
    return {"search_results": search_results, "process_steps": state['process_steps']}

def generate_answer_node(state: WebSearchState) -> Dict[str, Any]:
    """검색 결과를 바탕으로 최종 답변을 생성합니다."""
    logger.info("--- 3. 최종 답변 생성 시작 ---")
    state['process_steps'].append("3. 검색 결과 분석 및 최종 답변 생성 중...")
    llm_manager = LLMManager(state["llm_model"], OLLAMA_BASE_URL, state["temperature"])
    final_answer = web_search_utils.generate_final_response(llm_manager, state["original_query"], state["search_results"])
    logger.info("최종 답변 생성 완료.")
    state['process_steps'].append("4. 답변 생성 완료!")
    return {"final_answer": final_answer, "process_steps": state['process_steps']}

# --- Graph Definition ---
def create_web_search_graph() -> Runnable:
    """웹 검색 RAG 워크플로우를 정의하고 컴파일합니다."""
    workflow = StateGraph(WebSearchState)

    # 노드 추가
    workflow.add_node("refine_query", refine_query_node)
    workflow.add_node("search_web", search_web_node)
    workflow.add_node("generate_answer", generate_answer_node)

    # 엣지 연결
    workflow.add_edge(START, "refine_query")
    workflow.add_edge("refine_query", "search_web")
    workflow.add_edge("search_web", "generate_answer")
    workflow.add_edge("generate_answer", END)

    # 그래프 컴파일
    graph = workflow.compile()
    logger.info("웹 검색 그래프가 성공적으로 컴파일되었습니다.")
    return graph

# --- 실행 함수 ---
def run_web_search_graph(query: str, num_results: int, llm_model: str, temperature: float) -> Dict[str, Any]:
    """웹 검색 그래프를 실행하고 결과를 반환합니다."""
    graph = create_web_search_graph()
    initial_state = {
        "original_query": query,
        "num_results": num_results,
        "llm_model": llm_model,
        "temperature": temperature,
        "process_steps": []
    }
    
    final_state = graph.invoke(initial_state, config={"callbacks": [langfuse_handler]})
    
    # 최종 상태에서 필요한 정보만 추출하여 반환
    return {
        "original_query": query,
        "refined_queries": final_state.get("refined_queries"),
        "search_results": final_state.get("search_results"),
        "final_answer": final_state.get("final_answer"),
        "process_steps": final_state.get("process_steps")
    }

if __name__ == '__main__':
    # 테스트 실행
    test_query = "2025년 AI 트렌드는?"
    test_num_results = 3
    test_llm_model = "llama3.2:8b" # 시스템에 설치된 모델로 변경
    test_temperature = 0.1

    print(f"테스트 질문: {test_query}")
    results = run_web_search_graph(test_query, test_num_results, test_llm_model, test_temperature)

    print("\n--- 최종 결과 ---")
    print(f"원본 질문: {results.get('original_query')}")
    print(f"정제된 쿼리: {results.get('refined_queries')}")
    print(f"검색된 문서 수: {len(results.get('search_results', []))}")
    print(f"최종 답변: {results.get('final_answer')}")
    print(f"진행 과정: {results.get('process_steps')}") 