"""Web Search RAG system with external search integration."""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
import requests
from urllib.parse import urlparse, urljoin
import re
import json

# Try to import optional packages
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

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

from ..utils.llm_manager import LLMManager
from ..config import OLLAMA_BASE_URL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSearchRAG:
    """Web Search RAG system with external search integration."""
    
    def __init__(self, llm_model: str, base_url: str = OLLAMA_BASE_URL, temperature: float = 0.1):
        """Initialize WebSearchRAG system.
        
        Args:
            llm_model: LLM model name
            base_url: Ollama base URL
            temperature: LLM temperature
        """
        self.llm_manager = LLMManager(llm_model, base_url, temperature)
        self.llm = self.llm_manager.get_llm()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Prompts for different stages
        self.query_refinement_prompt = PromptTemplate(
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
        
        self.content_analysis_prompt = PromptTemplate(
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
        
        self.summary_prompt = PromptTemplate(
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
    
    def refine_query(self, original_query: str) -> Dict[str, str]:
        """Refine the original query for better web search.
        
        Args:
            original_query: Original user query
            
        Returns:
            Dictionary with refined queries and search intent
        """
        try:
            # Format the prompt
            prompt_text = self.query_refinement_prompt.format(original_query=original_query)
            
            # Call LLM directly
            result = self.llm.invoke(prompt_text)
            
            # Extract content from result
            if hasattr(result, 'content'):
                result_text = result.content
            else:
                result_text = str(result)
            
            logger.info(f"Query refinement result: {result_text}")
            
            # Parse the result to extract different components
            lines = result_text.strip().split('\n')
            refined_queries = {"korean": "", "english": "", "intent": ""}
            
            # Try to extract using regex patterns first
            import re
            
            # Pattern for Korean query
            korean_patterns = [
                r'한국어[:\s*]+(["\']?)([^"\']+)\1',
                r'\*\s*한국어[:\s*]+(["\']?)([^"\']+)\1',
                r'한국어.*?["\']([^"\']+)["\']',
            ]
            
            # Pattern for English query  
            english_patterns = [
                r'영어[:\s*]+(["\']?)([^"\']+)\1',
                r'\*\s*영어[:\s*]+(["\']?)([^"\']+)\1',
                r'영어.*?["\']([^"\']+)["\']',
            ]
            
            # Try regex patterns first
            for pattern in korean_patterns:
                match = re.search(pattern, result_text, re.IGNORECASE)
                if match:
                    korean_query = match.group(2).strip()
                    if " OR " in korean_query:
                        korean_query = korean_query.split(" OR ")[0].strip()
                    refined_queries["korean"] = korean_query
                    break
            
            for pattern in english_patterns:
                match = re.search(pattern, result_text, re.IGNORECASE)
                if match:
                    english_query = match.group(2).strip()
                    if " OR " in english_query:
                        english_query = english_query.split(" OR ")[0].strip()
                    refined_queries["english"] = english_query
                    break
            
            # Fallback to line-by-line parsing if regex fails
            if not refined_queries["korean"] or not refined_queries["english"]:
                current_section = None
                for line in lines:
                    line = line.strip()
                    # Look for various Korean query patterns
                    if "한국어:" in line and not refined_queries["korean"]:
                        korean_part = line.split("한국어:")[-1].strip()
                        # Remove quotes and OR operators, take the first query
                        korean_part = korean_part.replace('"', '').replace("'", '')
                        if " OR " in korean_part:
                            korean_part = korean_part.split(" OR ")[0].strip()
                        refined_queries["korean"] = korean_part
                    elif "영어:" in line and not refined_queries["english"]:
                        english_part = line.split("영어:")[-1].strip()
                        # Remove quotes and OR operators, take the first query
                        english_part = english_part.replace('"', '').replace("'", '')
                        if " OR " in english_part:
                            english_part = english_part.split(" OR ")[0].strip()
                        refined_queries["english"] = english_part
                    elif "검색 의도:" in line and not refined_queries["intent"]:
                        current_section = "intent"
                        # Also try to extract intent from the same line
                        if ":" in line:
                            intent_part = line.split(":")[-1].strip()
                            if intent_part:
                                refined_queries["intent"] = intent_part
                    elif current_section == "intent" and line and not line.startswith("**"):
                        refined_queries["intent"] += line + " "
            
            refined_queries["intent"] = refined_queries["intent"].strip()
            
            # Fallback if parsing failed
            if not refined_queries["korean"] and not refined_queries["english"]:
                refined_queries["korean"] = original_query
                refined_queries["english"] = original_query
            
            if not refined_queries["intent"]:
                refined_queries["intent"] = "정보 검색"
            
            logger.info(f"Parsed refined queries: {refined_queries}")
            
            return refined_queries
            
        except Exception as e:
            logger.error(f"Query refinement error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "korean": original_query,
                "english": original_query,
                "intent": "정보 검색"
            }
    
    def search_web(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web using Google search.
        
        Args:
            query: Search query
            num_results: Number of results to retrieve
            
        Returns:
            List of search results with URL, title, and content
        """
        search_results = []
        
        if not HAS_GOOGLESEARCH:
            logger.error("googlesearch-python package not installed")
            return [{
                "url": "https://error",
                "title": "패키지 오류",
                "content": "googlesearch-python 패키지가 설치되지 않았습니다. pip install googlesearch-python 을 실행해주세요.",
                "domain": "error"
            }]
        
        if not HAS_BEAUTIFULSOUP:
            logger.error("beautifulsoup4 package not installed")
            return [{
                "url": "https://error",
                "title": "패키지 오류",
                "content": "beautifulsoup4 패키지가 설치되지 않았습니다. pip install beautifulsoup4 를 실행해주세요.",
                "domain": "error"
            }]
        
        try:
            logger.info(f"Starting web search for query: {query}")
            
            # Use googlesearch library to get URLs
            # Note: pause parameter removed in newer versions, using manual delay instead
            try:
                # Try different parameter combinations for different versions
                logger.info(f"Attempting search with query: '{query}', num_results: {num_results}")
                
                # First try: with language
                try:
                    search_urls = list(search(query, num=num_results, lang='ko'))
                    logger.info(f"Search with 'num' parameter succeeded")
                except TypeError:
                    # Fallback: use num_results parameter
                    try:
                        search_urls = list(search(query, num_results=num_results, lang='ko'))
                        logger.info(f"Search with 'num_results' parameter succeeded")
                    except TypeError:
                        # Fallback: minimal parameters
                        search_urls = list(search(query))
                        logger.info(f"Search with minimal parameters succeeded")
                
                # Manual pause to avoid rate limiting
                time.sleep(2)
                
                logger.info(f"Found {len(search_urls)} URLs: {search_urls[:3] if len(search_urls) > 3 else search_urls}")
                
            except Exception as search_error:
                logger.error(f"Google search failed: {search_error}")
                import traceback
                logger.error(f"Search error traceback: {traceback.format_exc()}")
                
                # Try without language specification
                try:
                    logger.info("Retrying search without language specification...")
                    search_urls = list(search(query))
                    time.sleep(2)
                    logger.info(f"Found {len(search_urls)} URLs on retry: {search_urls[:3] if len(search_urls) > 3 else search_urls}")
                except Exception as retry_error:
                    logger.error(f"Retry also failed: {retry_error}")
                    logger.error(f"Retry error traceback: {traceback.format_exc()}")
                    search_urls = []
            
            for i, url in enumerate(search_urls):
                try:
                    logger.info(f"Fetching content from URL {i+1}: {url}")
                    
                    # Add delay to avoid rate limiting
                    if i > 0:
                        time.sleep(1)
                    
                    # Fetch content from URL
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    
                    # Parse content with BeautifulSoup
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract title
                    title = soup.find('title')
                    title_text = title.text.strip() if title else "제목 없음"
                    
                    # Extract main content
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Get text content
                    text = soup.get_text()
                    
                    # Clean up text
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    
                    # Limit text length
                    if len(text) > 3000:
                        text = text[:3000] + "..."
                    
                    search_results.append({
                        "url": url,
                        "title": title_text,
                        "content": text,
                        "domain": urlparse(url).netloc
                    })
                    
                    logger.info(f"Successfully fetched content from {url} (length: {len(text)})")
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch content from {url}: {e}")
                    search_results.append({
                        "url": url,
                        "title": "내용 로드 실패",
                        "content": "웹페이지 내용을 가져올 수 없습니다.",
                        "domain": urlparse(url).netloc
                    })
                    
        except Exception as e:
            logger.error(f"Web search error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
        logger.info(f"Web search completed. Found {len(search_results)} results")
        return search_results
    
    def analyze_search_results(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Analyze search results and generate comprehensive response.
        
        Args:
            query: Original query
            search_results: List of search results
            
        Returns:
            Analyzed response
        """
        try:
            # Format search results for analysis
            formatted_results = ""
            for i, result in enumerate(search_results, 1):
                formatted_results += f"\n=== 검색 결과 {i} ===\n"
                formatted_results += f"제목: {result['title']}\n"
                formatted_results += f"URL: {result['url']}\n"
                formatted_results += f"도메인: {result['domain']}\n"
                formatted_results += f"내용: {result['content'][:1000]}...\n"
            
            # Format the prompt
            prompt_text = self.content_analysis_prompt.format(
                query=query, 
                search_results=formatted_results
            )
            
            # Call LLM directly
            result = self.llm.invoke(prompt_text)
            
            # Extract content from result
            if hasattr(result, 'content'):
                analysis = result.content
            else:
                analysis = str(result)
            
            logger.info(f"Analysis result: {analysis[:200]}...")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Search results analysis error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return "검색 결과 분석 중 오류가 발생했습니다."
    
    def generate_final_response(self, query: str, search_results: List[Dict[str, Any]], analysis: str) -> str:
        """Generate final comprehensive response.
        
        Args:
            query: Original query
            search_results: List of search results
            analysis: Analysis result
            
        Returns:
            Final expert response
        """
        try:
            # Format web content
            web_content = ""
            for i, result in enumerate(search_results, 1):
                web_content += f"\n[출처 {i}] {result['title']} ({result['domain']})\n"
                web_content += f"{result['content'][:500]}...\n"
            
            # Format the prompt
            prompt_text = self.summary_prompt.format(
                query=query,
                web_content=web_content,
                analysis=analysis
            )
            
            # Call LLM directly
            result = self.llm.invoke(prompt_text)
            
            # Extract content from result
            if hasattr(result, 'content'):
                final_response = result.content
            else:
                final_response = str(result)
            
            logger.info(f"Final response generated: {final_response[:200]}...")
            
            # Add source information
            final_response += "\n\n=== 참고 자료 ===\n"
            for i, result in enumerate(search_results, 1):
                final_response += f"{i}. {result['title']}\n   {result['url']}\n"
            
            return final_response
            
        except Exception as e:
            logger.error(f"Final response generation error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return "최종 답변 생성 중 오류가 발생했습니다."
    
    def search_and_answer(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Complete web search and answer pipeline.
        
        Args:
            query: User query
            num_results: Number of search results to retrieve
            
        Returns:
            Dictionary with all results and process information
        """
        process_info = {
            "original_query": query,
            "refined_queries": {},
            "search_results": [],
            "analysis": "",
            "final_answer": "",
            "process_steps": []
        }
        
        try:
            # Step 1: Query refinement
            process_info["process_steps"].append("🔍 질문 의도 분석 및 검색 쿼리 정제")
            refined_queries = self.refine_query(query)
            process_info["refined_queries"] = refined_queries
            
            # Step 2: Web search
            process_info["process_steps"].append("🌐 웹 검색 실행")
            
            # Choose the best search query
            search_query = refined_queries.get("korean", "").strip()
            if not search_query:
                search_query = refined_queries.get("english", "").strip()
            if not search_query:
                search_query = query
            
            logger.info(f"Using search query: {search_query}")
            search_results = self.search_web(search_query, num_results)
            process_info["search_results"] = search_results
            
            # Step 3: Analysis
            process_info["process_steps"].append("📊 검색 결과 분석")
            analysis = self.analyze_search_results(query, search_results)
            process_info["analysis"] = analysis
            
            # Step 4: Final response
            process_info["process_steps"].append("📝 전문가 답변 생성")
            final_answer = self.generate_final_response(query, search_results, analysis)
            process_info["final_answer"] = final_answer
            
            process_info["process_steps"].append("✅ 완료")
            
        except Exception as e:
            logger.error(f"Search and answer pipeline error: {e}")
            process_info["final_answer"] = f"처리 중 오류가 발생했습니다: {str(e)}"
        
        return process_info 