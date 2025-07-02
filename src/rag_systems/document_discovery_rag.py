"""
Document Discovery RAG Utilities
문서 발견 및 상세 검색을 위한 2단계 RAG 시스템 유틸리티 함수 모음
"""
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import re

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.components.document_loader import DocumentLoader
from src.utils.llm_manager import LLMManager
from src.utils.embeddings import EmbeddingManager
from src.utils.vector_store import VectorStoreManager
from src.config import DOCS_FOLDER as DOCS_DIR, JSON_OUTPUT_FOLDER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants and Prompts ---

SUMMARY_CACHE_DIR = JSON_OUTPUT_FOLDER / "document_summaries"
SUMMARY_CACHE_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_CACHE_FILE = SUMMARY_CACHE_DIR / "document_summaries.json"

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["document_content", "document_name"],
    template="""다음 문서의 내용을 분석하여 요약해주세요:

문서명: {document_name}

문서 내용:
{document_content}

다음 형식으로 요약해주세요:
1. 주요 주제: (문서의 핵심 주제)
2. 키워드: (중요한 키워드들을 쉼표로 구분)
3. 내용 요약: (3-5문장으로 핵심 내용 요약)
4. 문서 유형: (보고서/논문/정책문서/기술문서 등)
5. 대상 독자: (일반인/전문가/정책결정자 등)

요약:"""
)

RELEVANCE_PROMPT = PromptTemplate(
    input_variables=["query", "document_summaries"],
    template="""사용자 질문: {query}

다음은 사용 가능한 문서들의 요약입니다:
{document_summaries}

사용자 질문과 각 문서의 관련성을 0-100점으로 평가하고, 관련성이 높은 순서대로 정렬해주세요.

다음 형식으로 답변해주세요:
[문서명] 점수: XX점, 관련성 설명: (왜 관련이 있는지 간단히 설명)

예시:
[문서1.pdf] 점수: 85점, 관련성 설명: 사용자가 묻는 AI 정책과 직접적으로 관련된 정부 정책 문서
[문서2.pdf] 점수: 70점, 관련성 설명: AI 기술 동향을 다루어 질문과 부분적으로 관련"""
)

DETAIL_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""다음 문서 내용을 바탕으로 질문에 답변해주세요:

문서 내용:
{context}

질문: {question}

답변 시 다음 사항을 고려해주세요:
1. 문서에서 직접적으로 언급된 내용을 우선적으로 활용
2. 구체적인 수치나 데이터가 있으면 포함
3. 관련된 맥락이나 배경 정보도 함께 제공
4. 문서에 없는 내용은 추측하지 말고 "문서에 명시되지 않음"이라고 명시

답변:"""
)

# --- Utility Functions ---

def get_available_documents() -> List[Dict[str, Any]]:
    """사용 가능한 문서 목록과 기본 정보 반환"""
    document_loader = DocumentLoader()
    documents = []
    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    for pdf_file in pdf_files:
        doc_info = document_loader.get_document_info(pdf_file.name)
        if doc_info:
            documents.append(doc_info)
    return documents

def load_document_summaries() -> Dict[str, Dict[str, Any]]:
    """캐시된 문서 요약 로드"""
    if SUMMARY_CACHE_FILE.exists():
        try:
            with open(SUMMARY_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"요약 캐시 로드 실패: {e}")
            return {}
    return {}

def save_document_summaries(summaries: Dict[str, Dict[str, Any]]):
    """문서 요약을 캐시에 저장"""
    try:
        with open(SUMMARY_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"요약 캐시 저장 실패: {e}")

def generate_single_document_summary(llm_manager: LLMManager, filename: str) -> Optional[Dict[str, Any]]:
    """단일 문서 요약 생성"""
    document_loader = DocumentLoader()
    try:
        documents = document_loader.load_document(filename)
        if not documents:
            return None
        
        content = "\n\n".join([doc.page_content for doc in documents[:3]])
        content = content[:4000] + "..." if len(content) > 4000 else content

        chain = SUMMARY_PROMPT | llm_manager.get_llm() | StrOutputParser()
        summary_text = chain.invoke({"document_content": content, "document_name": filename})
        
        doc_info = document_loader.get_document_info(filename)
        return {
            "filename": filename, "summary": summary_text, "generated_at": datetime.now().isoformat(),
            "pages": doc_info.get("pages", 0), "size_mb": doc_info.get("size_mb", 0),
            "preview": doc_info.get("preview", ""), "title": doc_info.get("title", filename)
        }
    except Exception as e:
        logger.error(f"문서 요약 생성 실패 ({filename}): {e}")
        return None

def parse_relevance_result(result_text: str) -> List[Tuple[str, int, str]]:
    """관련성 평가 결과 파싱"""
    relevant_docs = []
    # 정규 표현식을 사용하여 더 안정적으로 파싱
    pattern = re.compile(r"\[(.+?)\]\s*점수:\s*(\d+)\s*점,\s*관련성 설명:\s*(.+)", re.DOTALL)
    lines = result_text.strip().split('\n')
    for line in lines:
        match = pattern.search(line.strip())
        if match:
            filename, score, explanation = match.groups()
            relevant_docs.append((filename.strip(), int(score), explanation.strip()))
        # 예전 포맷 호환
        elif '[' in line and ']' in line and '점수:' in line:
             try:
                start_bracket, end_bracket = line.find('['), line.find(']')
                filename = line[start_bracket+1:end_bracket]
                score_start = line.find('점수:') + 3
                score_end = line.find('점', score_start)
                score = int(line[score_start:score_end].strip())
                explanation_start = line.find('관련성 설명:')
                explanation = line[explanation_start+7:].strip() if explanation_start != -1 else ""
                relevant_docs.append((filename, score, explanation))
             except (ValueError, IndexError):
                continue

    relevant_docs.sort(key=lambda x: x[1], reverse=True)
    return relevant_docs

def find_relevant_documents(llm_manager: LLMManager, query: str, top_k: int = 5) -> List[Tuple[str, int, str]]:
    """사용자 질문과 관련된 문서들을 찾기"""
    summaries = load_document_summaries()
    if not summaries:
        return []
    
    summary_text = "\n".join([f"[{fname}]\n{sdata['summary']}\n" for fname, sdata in summaries.items()])
    
    try:
        chain = RELEVANCE_PROMPT | llm_manager.get_llm() | StrOutputParser()
        relevance_result = chain.invoke({"query": query, "document_summaries": summary_text})
        parsed_results = parse_relevance_result(relevance_result)
        return parsed_results[:top_k]
    except Exception as e:
        logger.error(f"관련성 평가 실패: {e}")
        return []

def perform_detailed_search(
    llm_manager: LLMManager,
    embedding_manager: EmbeddingManager,
    vector_store_manager_config: dict,
    filename: str,
    query: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """선택된 문서에 대한 상세 검색 수행"""
    document_loader = DocumentLoader()
    try:
        documents = document_loader.process_documents_for_rag([filename])
        if not documents:
            return {"error": "문서를 로드하거나 청크로 분할할 수 없습니다."}

        # 새로운 임시 VectorStoreManager 인스턴스 생성
        temp_collection_name = f"temp_{Path(filename).stem}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        temp_vector_store_manager = VectorStoreManager(
            embeddings=embedding_manager.get_embeddings(),
            vector_store_type=vector_store_manager_config.get("vector_store_type", "faiss"),
            collection_name=temp_collection_name
        )
        
        temp_vector_store = temp_vector_store_manager.create_vector_store(documents)
        retriever = temp_vector_store.as_retriever(search_kwargs={"k": top_k})
        relevant_chunks = retriever.get_relevant_documents(query)
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        
        chain = DETAIL_QA_PROMPT | llm_manager.get_llm() | StrOutputParser()
        answer = chain.invoke({"context": context, "question": query})
        
        try:
            temp_vector_store_manager.cleanup()
        except Exception as e:
            logger.warning(f"임시 벡터스토어 정리 실패: {e}")

        return {
            "filename": filename,
            "answer": answer,
            "relevant_chunks": [{"content": c.page_content, "metadata": c.metadata} for c in relevant_chunks],
            "total_chunks_found": len(relevant_chunks)
        }
    except Exception as e:
        logger.error(f"상세 검색 실패: {e}", exc_info=True)
        return {"error": f"상세 검색 중 오류 발생: {e}"}
        
def get_document_overview(llm_manager: LLMManager, filename: str) -> Dict[str, Any]:
    """문서 개요 정보 반환 (캐시 확인 후 없으면 생성)"""
    summaries = load_document_summaries()
    if filename in summaries:
        return summaries[filename]
    
    logger.info(f"'{filename}'에 대한 캐시된 요약이 없어 새로 생성합니다.")
    summary_data = generate_single_document_summary(llm_manager, filename)
    if summary_data:
        summaries[filename] = summary_data
        save_document_summaries(summaries)
        return summary_data
    return {"error": f"'{filename}'의 개요 정보를 가져올 수 없습니다."} 