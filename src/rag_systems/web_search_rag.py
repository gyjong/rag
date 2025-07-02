"""Web Search RAG system utility functions."""

import logging
import time
from typing import List, Dict, Any
import requests
from urllib.parse import urlparse
import re

try:
    from bs4 import BeautifulSoup
    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False

try:
    from googlesearch import search
    HAS_GOOGLESEARCH = True
except ImportError:
    HAS_GOOGLESEARCH = False

from langchain_core.prompts import PromptTemplate
from ..utils.llm_manager import LLMManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Prompts ---

QUERY_REFINEMENT_PROMPT = PromptTemplate(
    input_variables=["original_query"],
    template="""다음 질문을 분석하고 웹 검색에 최적화된 검색 쿼리를 생성해주세요.

원본 질문: {original_query}

검색 쿼리 생성 시 고려사항:
1. 핵심 키워드 추출
2. 검색 의도 파악
3. 관련 용어 포함
4. 최신 정보가 필요한지 판단

검색 쿼리 (한국어와 영어 각각):
한국어: 
영어: 

검색 의도:
"""
)

CONTENT_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["query", "search_results"],
    template="""다음 검색 결과를 분석하여 질문에 대한 포괄적이고 전문적인 답변을 생성해주세요.

질문: {query}

검색 결과:
{search_results}

답변 생성 시 고려사항:
1. 검색 결과의 신뢰성 평가
2. 정보의 일관성 확인
3. 최신성 고려
4. 전문적이고 정확한 답변
5. 출처 정보 포함

전문가 답변:
"""
)

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["query", "web_content", "analysis"],
    template="""다음 정보를 종합하여 최종 전문가 답변을 생성해주세요.

질문: {query}

웹 검색 내용:
{web_content}

분석 결과:
{analysis}

최종 답변 구성:
1. 핵심 내용 요약
2. 상세 설명
3. 관련 통계나 데이터 (있는 경우)
4. 전망이나 시사점
5. 참고 자료

최종 전문가 답변:
"""
)

# --- Utility Functions ---

def refine_query(llm_manager: LLMManager, original_query: str) -> Dict[str, str]:
    """Refine the original query for better web search with robust parsing."""
    try:
        prompt_text = QUERY_REFINEMENT_PROMPT.format(original_query=original_query)
        result = llm_manager.get_llm().invoke(prompt_text)
        result_text = result.content if hasattr(result, 'content') else str(result)
        logger.info(f"Query refinement result: {result_text}")

        refined_queries = {"korean": "", "english": "", "intent": ""}
        
        # Robust parsing for queries
        korean_match = re.search(r'한국어:.*?[\* \-]\s*?"(.*?)"', result_text, re.DOTALL) or re.search(r'한국어:.*?[\* \-]\s*(.*)', result_text)
        if korean_match:
            refined_queries["korean"] = korean_match.group(1).strip()

        english_match = re.search(r'영어:.*?[\* \-]\s*?"(.*?)"', result_text, re.DOTALL) or re.search(r'영어:.*?[\* \-]\s*(.*)', result_text)
        if english_match:
            refined_queries["english"] = english_match.group(1).strip()

        # Robust parsing for intent: find the last occurrence
        intent_start_index = result_text.rfind('**검색 의도:**')
        if intent_start_index != -1:
            intent_text = result_text[intent_start_index + len('**검색 의도:**'):].strip()
            refined_queries["intent"] = intent_text
        else: # Fallback
            intent_match = re.search(r'검색 의도[:\s*]+(.*)', result_text, re.DOTALL)
            if intent_match:
                refined_queries["intent"] = intent_match.group(1).strip()


        if not refined_queries["korean"] and not refined_queries["english"]:
            refined_queries["korean"] = original_query
        if not refined_queries["intent"]:
            refined_queries["intent"] = "정보 검색"
            
        logger.info(f"Parsed refined queries: {refined_queries}")
        return refined_queries
    except Exception as e:
        logger.error(f"Query refinement error: {e}")
        return {"korean": original_query, "english": original_query, "intent": "정보 검색"}

def search_web(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Search the web using Google search, compatible with different library versions."""
    if not HAS_GOOGLESEARCH or not HAS_BEAUTIFULSOUP:
        logger.error("Required packages (googlesearch-python, beautifulsoup4) are not installed.")
        return [{"url": "https://error", "title": "패키지 오류", "content": "필수 패키지가 설치되지 않았습니다.", "domain": "error"}]

    search_results = []
    try:
        logger.info(f"Starting web search for query: {query}, aiming for {num_results} results.")
        
        # More robust way to get results, avoiding parameter issues
        search_urls_generator = search(query, lang='ko')
        search_urls = []
        for i, url in enumerate(search_urls_generator):
            if i >= num_results:
                break
            search_urls.append(url)

        logger.info(f"Found {len(search_urls)} URLs.")

        for i, url in enumerate(search_urls):
            time.sleep(1) # To avoid rate limiting
            try:
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                title = soup.find('title').text.strip() if soup.find('title') else "제목 없음"
                
                for script_or_style in soup(["script", "style"]):
                    script_or_style.decompose()
                text = ' '.join(t.strip() for t in soup.get_text().split() if t.strip())
                
                search_results.append({
                    "url": url, "title": title, "content": text[:3000], "domain": urlparse(url).netloc
                })
            except Exception as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                search_results.append({"url": url, "title": "내용 로드 실패", "content": "", "domain": urlparse(url).netloc})
    except Exception as e:
        logger.error(f"Web search error: {e}")
    return search_results

def generate_final_response(llm_manager: LLMManager, query: str, search_results: List[Dict[str, Any]]) -> str:
    """Generate final comprehensive response."""
    try:
        web_content = ""
        for i, result in enumerate(search_results, 1):
            web_content += f"\\n[출처 {i}] {result['title']} ({result['domain']})\\n{result['content'][:500]}...\\n"
        
        prompt_text = SUMMARY_PROMPT.format(query=query, web_content=web_content, analysis="통합된 웹 검색 결과")
        result = llm_manager.get_llm().invoke(prompt_text)
        final_response = result.content if hasattr(result, 'content') else str(result)

        final_response += "\\n\\n=== 참고 자료 ===\\n"
        for i, res in enumerate(search_results, 1):
            final_response += f"{i}. {res['title']}\\n   {res['url']}\\n"
        return final_response
    except Exception as e:
        logger.error(f"Final response generation error: {e}")
        return "최종 답변 생성 중 오류가 발생했습니다." 