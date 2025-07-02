"""
LangGraph implementation for the Document Discovery RAG system.
Defines three distinct graphs for:
1. Generating summaries for all documents.
2. Discovering relevant documents based on a query.
3. Performing a detailed search within a single document.
"""
import logging
from typing import TypedDict, List, Dict, Any, Tuple, Optional

from langgraph.graph import StateGraph, END

from src.rag_systems import document_discovery_rag as dd_rag
from src.utils.llm_manager import LLMManager
from src.utils.embeddings import EmbeddingManager

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Graph 1: Summary Generation ---

class SummaryGenerationState(TypedDict):
    available_docs: List[Dict[str, Any]]
    summaries: Dict[str, Dict[str, Any]]
    total_docs: int
    processed_docs: int
    current_message: str

def get_docs_and_summaries_node(state: SummaryGenerationState) -> Dict[str, Any]:
    logger.info("--- 요약 생성: 1. 사용 가능 문서 및 기존 요약 로드 ---")
    available_docs = dd_rag.get_available_documents()
    summaries = dd_rag.load_document_summaries()
    return {
        "available_docs": available_docs,
        "summaries": summaries,
        "total_docs": len(available_docs),
        "processed_docs": 0
    }

def generate_summaries_node(state: SummaryGenerationState, llm_manager: LLMManager) -> Dict[str, Any]:
    logger.info("--- 요약 생성: 2. 신규 문서 요약 생성 시작 ---")
    summaries = state['summaries'].copy()
    processed_count = 0
    
    for doc_info in state['available_docs']:
        filename = doc_info["filename"]
        if filename not in summaries:
            logger.info(f"'{filename}' 요약 생성 중...")
            summary_data = dd_rag.generate_single_document_summary(llm_manager, filename)
            if summary_data:
                summaries[filename] = summary_data
                dd_rag.save_document_summaries(summaries) # 진행상황 실시간 저장
            processed_count += 1
    
    logger.info("모든 신규 문서 요약 생성 완료.")
    return {
        "summaries": summaries,
        "processed_docs": state['processed_docs'] + processed_count,
        "current_message": "요약 생성 완료"
    }

def create_summary_generation_graph(llm_manager: LLMManager):
    workflow = StateGraph(SummaryGenerationState)
    workflow.add_node("get_initial_state", get_docs_and_summaries_node)
    workflow.add_node("generate_summaries", lambda state: generate_summaries_node(state, llm_manager))
    workflow.set_entry_point("get_initial_state")
    workflow.add_edge("get_initial_state", "generate_summaries")
    workflow.add_edge("generate_summaries", END)
    return workflow.compile()

# --- Graph 2: Document Discovery ---

class DocumentDiscoveryState(TypedDict):
    query: str
    top_k: int
    relevant_docs: List[Tuple[str, int, str]]

def find_relevant_docs_node(state: DocumentDiscoveryState, llm_manager: LLMManager) -> Dict[str, Any]:
    logger.info(f"--- 문서 발견: 1. '{state['query']}' 관련 문서 검색 ---")
    relevant_docs = dd_rag.find_relevant_documents(llm_manager, state['query'], state['top_k'])
    logger.info(f"{len(relevant_docs)}개의 관련 문서를 찾았습니다.")
    return {"relevant_docs": relevant_docs}

def create_document_discovery_graph(llm_manager: LLMManager):
    workflow = StateGraph(DocumentDiscoveryState)
    workflow.add_node("find_relevant_docs", lambda state: find_relevant_docs_node(state, llm_manager))
    workflow.set_entry_point("find_relevant_docs")
    workflow.add_edge("find_relevant_docs", END)
    return workflow.compile()

# --- Graph 3: Detailed Search ---

class DetailedSearchState(TypedDict):
    filename: str
    query: str
    top_k: int
    vector_store_config: Dict[str, Any]
    answer: str
    relevant_chunks: List[Dict[str, Any]]
    error: Optional[str]

def detailed_search_node(state: DetailedSearchState, llm_manager: LLMManager, embedding_manager: EmbeddingManager) -> Dict[str, Any]:
    logger.info(f"--- 상세 검색: 1. '{state['filename']}' 문서에서 '{state['query']}' 검색 ---")
    result = dd_rag.perform_detailed_search(
        llm_manager=llm_manager,
        embedding_manager=embedding_manager,
        vector_store_manager_config=state['vector_store_config'],
        filename=state['filename'],
        query=state['query'],
        top_k=state['top_k']
    )
    if "error" in result:
        return {"error": result["error"]}
    return {
        "answer": result["answer"],
        "relevant_chunks": result["relevant_chunks"],
        "error": None
    }

def create_detailed_search_graph(llm_manager: LLMManager, embedding_manager: EmbeddingManager):
    workflow = StateGraph(DetailedSearchState)
    # The `partial` usage or a lambda is necessary to pass extra arguments to the node
    workflow.add_node(
        "detailed_search",
        lambda state: detailed_search_node(state, llm_manager, embedding_manager)
    )
    workflow.set_entry_point("detailed_search")
    workflow.add_edge("detailed_search", END)
    return workflow.compile() 