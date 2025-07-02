"""JSON-based RAG utilities for structured data queries."""

import json
import re
from typing import List, Dict, Any, Optional, Iterator
from datetime import datetime
from difflib import SequenceMatcher
import logging

from ..utils.llm_manager import LLMManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JSONDataProcessor:
    """JSON 데이터 처리 및 검색을 위한 클래스."""
    
    def __init__(self, json_file_path: str, data_type: str):
        """Initialize JSON data processor.
        
        Args:
            json_file_path: JSON 파일 경로
            data_type: 데이터 타입 (bus_schedule, menu 등)
        """
        self.json_file_path = json_file_path
        self.data_type = data_type
        self.data = self._load_json_data()
        
    def _load_json_data(self) -> Dict[str, Any]:
        """JSON 파일을 로드합니다."""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"JSON 파일 로드 실패: {str(e)}")
            return {}
    
    def search_bus_routes(self, query: str) -> List[Dict[str, Any]]:
        """버스 노선 정보를 검색합니다."""
        if not self.data or 'bus_routes' not in self.data:
            return []
        
        query_lower = query.lower()
        results = []
        
        for route in self.data['bus_routes']:
            searchable_text = [
                route.get('bus_number', ''), route.get('route_name', ''),
                route.get('departure_terminal', {}).get('name', ''),
                route.get('arrival_terminal', {}).get('name', ''),
                route.get('route_type', ''), route.get('notes', '')
            ]
            for stop in route.get('intermediate_stops', []):
                searchable_text.append(stop.get('stop_name', ''))
            
            combined_text = ' '.join(filter(None, searchable_text)).lower()
            similarity = SequenceMatcher(None, query_lower, combined_text).ratio()
            
            if any(keyword in text.lower() for keyword in query_lower.split() for text in searchable_text if text) or similarity > 0.3:
                route_copy = route.copy()
                route_copy['relevance_score'] = similarity
                results.append(route_copy)
        
        return sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)[:5]
    
    def search_menu(self, query: str, date_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """메뉴 정보를 검색합니다."""
        if not self.data or 'menu_plan' not in self.data:
            return []
        
        query_lower = query.lower()
        results = []
        daily_menus = self.data['menu_plan'].get('daily_menus', [])
        
        for daily_menu in daily_menus:
            if date_filter and daily_menu.get('date') != date_filter:
                continue
            
            for meal in daily_menu.get('meals', []):
                searchable_text = [
                    meal.get('meal_type', ''), meal.get('cuisine_type', ''),
                    daily_menu.get('day_of_week', ''), daily_menu.get('date', '')
                ]
                for item in meal.get('menu_items', []):
                    searchable_text.extend([item.get('name', ''), item.get('category', '')])
                
                combined_text = ' '.join(filter(None, searchable_text)).lower()
                similarity = SequenceMatcher(None, query_lower, combined_text).ratio()
                
                if any(keyword in text.lower() for keyword in query_lower.split() for text in searchable_text if text) or similarity > 0.2:
                    meal_copy = meal.copy()
                    meal_copy['date'] = daily_menu.get('date')
                    meal_copy['day_of_week'] = daily_menu.get('day_of_week')
                    meal_copy['relevance_score'] = similarity
                    results.append(meal_copy)
        
        return sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)[:10]

    def get_today_menu(self) -> List[Dict[str, Any]]:
        """오늘의 메뉴를 가져옵니다."""
        today = datetime.now().strftime('%Y-%m-%d')
        return self.search_menu('', date_filter=today)
    
    def get_weekly_menu(self) -> List[Dict[str, Any]]:
        """이번 주 메뉴를 가져옵니다."""
        if not self.data or 'menu_plan' not in self.data:
            return []
        return self.data['menu_plan'].get('daily_menus', [])


def search_bus_routes_utility(processor: JSONDataProcessor, query: str) -> List[Dict[str, Any]]:
    """Utility to search bus routes."""
    logger.info(f"버스 정보 검색 시작: {query}")
    results = processor.search_bus_routes(query)
    logger.info(f"{len(results)}개의 버스 노선 검색 결과 발견.")
    return results

def search_menu_items_utility(processor: JSONDataProcessor, query: str) -> List[Dict[str, Any]]:
    """Utility to search menu items, handling special keywords."""
    logger.info(f"메뉴 정보 검색 시작: {query}")
    query_lower = query.lower()
    
    if '오늘' in query_lower or 'today' in query_lower:
        results = processor.get_today_menu()
    elif '이번주' in query_lower or '주간' in query_lower or 'weekly' in query_lower:
        results = processor.get_weekly_menu()
    else:
        # Check for specific date in YYYY-MM-DD format
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', query)
        date_filter = date_match.group(1) if date_match else None
        results = processor.search_menu(query, date_filter)
        
    logger.info(f"{len(results)}개의 메뉴 검색 결과 발견.")
    return results


def format_bus_context(bus_results: List[Dict[str, Any]]) -> str:
    """버스 검색 결과를 컨텍스트로 포맷팅합니다."""
    context_parts = []
    for bus in bus_results:
        context = f"""
버스 정보:
- 버스 번호: {bus.get('bus_number', 'N/A')}
- 노선명: {bus.get('route_name', 'N/A')}
- 노선 유형: {bus.get('route_type', 'N/A')}
- 운영업체: {bus.get('operator', 'N/A')}
- 출발지: {bus.get('departure_terminal', {}).get('name', 'N/A')}
- 도착지: {bus.get('arrival_terminal', {}).get('name', 'N/A')}
- 예상 소요시간: {bus.get('estimated_duration_minutes', 'N/A')}분
"""
        if bus.get('intermediate_stops'):
            stops = [stop.get('stop_name', '') for stop in bus.get('intermediate_stops', [])]
            context += f"- 중간 정류장: {' → '.join(stops)}\n"
        context_parts.append(context)
    return "\n" + "="*50 + "\n".join(context_parts)

def format_menu_context(menu_results: List[Dict[str, Any]]) -> str:
    """메뉴 검색 결과를 컨텍스트로 포맷팅합니다."""
    context_parts = []
    for menu in menu_results:
        context = f"""
메뉴 정보:
- 날짜: {menu.get('date', 'N/A')} ({menu.get('day_of_week', 'N/A')})
- 식사: {menu.get('meal_type', 'N/A')} ({menu.get('service_time', 'N/A')})
- 요리 종류: {menu.get('cuisine_type', 'N/A')}
"""
        menu_items = menu.get('menu_items', [])
        if menu_items:
            context += "- 구성:\n"
            for item in menu_items:
                context += f"  • {item.get('category', '')}: {item.get('name', '')} ({item.get('calories', 'N/A')}kcal)\n"
        context_parts.append(context)
    return "\n" + "="*50 + "\n".join(context_parts)

def generate_answer_stream(llm_manager: LLMManager, query: str, context: str, data_type: str) -> Iterator[str]:
    """컨텍스트를 바탕으로 자연스러운 답변을 스트리밍으로 생성합니다."""
    if not context:
        yield f"요청하신 {data_type}을(를) 찾을 수 없습니다. 다른 검색어를 시도해보세요."
        return

    instruction = f"""주어진 '{data_type}' 데이터를 바탕으로 다음 질문에 대해 친절하고 정확하게 답변해 주세요.
- 가장 관련성 높은 정보를 우선적으로 제시하세요.
- 사용자가 이해하기 쉽도록 구조화된 형태로 답변하세요.

질문: {query}"""

    try:
        logger.info(f"LLM 답변 생성 시작. 데이터 타입: {data_type}")
        response_stream = llm_manager.generate_response_stream(
            prompt=instruction,
            context=context
        )
        for chunk in response_stream:
            yield chunk
        logger.info("LLM 답변 생성 완료.")
    except Exception as e:
        logger.error(f"답변 생성 중 오류 발생: {str(e)}")
        yield f"답변 생성 중 오류가 발생했습니다: {str(e)}" 