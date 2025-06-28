"""JSON-based RAG implementation for structured data queries."""

import json
import time
from typing import List, Dict, Any, Optional
import streamlit as st
from datetime import datetime, timedelta
from difflib import SequenceMatcher

from ..utils.llm_manager import LLMManager


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
            st.error(f"JSON 파일 로드 실패: {str(e)}")
            return {}
    
    def search_bus_routes(self, query: str) -> List[Dict[str, Any]]:
        """버스 노선 정보를 검색합니다."""
        if not self.data or 'bus_routes' not in self.data:
            return []
        
        query_lower = query.lower()
        results = []
        
        for route in self.data['bus_routes']:
            # 버스 번호, 노선명, 출발지, 도착지, 정류장 등에서 검색
            searchable_text = [
                route.get('bus_number', ''),
                route.get('route_name', ''),
                route.get('departure_terminal', {}).get('name', ''),
                route.get('arrival_terminal', {}).get('name', ''),
                route.get('route_type', ''),
                route.get('notes', '')
            ]
            
            # 중간 정류장도 검색 대상에 포함
            for stop in route.get('intermediate_stops', []):
                searchable_text.append(stop.get('stop_name', ''))
            
            # 텍스트 유사도 계산
            combined_text = ' '.join(searchable_text).lower()
            similarity = SequenceMatcher(None, query_lower, combined_text).ratio()
            
            # 키워드가 포함되어 있거나 유사도가 높은 경우 결과에 추가
            if any(keyword in text.lower() for keyword in query_lower.split() for text in searchable_text) or similarity > 0.3:
                route_copy = route.copy()
                route_copy['relevance_score'] = similarity
                results.append(route_copy)
        
        # 관련도 순으로 정렬
        return sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)[:5]
    
    def search_menu(self, query: str, date_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """메뉴 정보를 검색합니다."""
        if not self.data or 'menu_plan' not in self.data:
            return []
        
        query_lower = query.lower()
        results = []
        
        daily_menus = self.data['menu_plan'].get('daily_menus', [])
        
        for daily_menu in daily_menus:
            # 날짜 필터링
            if date_filter and daily_menu.get('date') != date_filter:
                continue
            
            for meal in daily_menu.get('meals', []):
                # 식사 유형, 요리 종류, 메뉴 아이템에서 검색
                searchable_text = [
                    meal.get('meal_type', ''),
                    meal.get('cuisine_type', ''),
                    daily_menu.get('day_of_week', ''),
                    daily_menu.get('date', '')
                ]
                
                # 메뉴 아이템들도 검색 대상에 포함
                for item in meal.get('menu_items', []):
                    searchable_text.extend([
                        item.get('name', ''),
                        item.get('category', '')
                    ])
                
                # 텍스트 유사도 계산
                combined_text = ' '.join(searchable_text).lower()
                similarity = SequenceMatcher(None, query_lower, combined_text).ratio()
                
                # 키워드가 포함되어 있거나 유사도가 높은 경우 결과에 추가
                if any(keyword in text.lower() for keyword in query_lower.split() for text in searchable_text) or similarity > 0.2:
                    meal_copy = meal.copy()
                    meal_copy['date'] = daily_menu.get('date')
                    meal_copy['day_of_week'] = daily_menu.get('day_of_week')
                    meal_copy['relevance_score'] = similarity
                    results.append(meal_copy)
        
        # 관련도 순으로 정렬
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


class JSONRAG:
    """JSON 파일 기반 RAG 시스템."""
    
    def __init__(self, llm_manager: LLMManager):
        """Initialize JSON RAG system.
        
        Args:
            llm_manager: LLM manager instance
        """
        self.llm_manager = llm_manager
        self.name = "JSON RAG"
        self.description = "JSON 구조화 데이터 기반 RAG 시스템"
        self.processors = {}
        
    def add_json_source(self, source_name: str, json_file_path: str, data_type: str):
        """JSON 데이터 소스를 추가합니다."""
        self.processors[source_name] = JSONDataProcessor(json_file_path, data_type)
    
    def query_bus_info(self, query: str) -> Dict[str, Any]:
        """버스 정보를 검색합니다."""
        start_time = time.time()
        
        if 'bus_schedule' not in self.processors:
            return {
                "question": query,
                "answer": "버스 스케줄 데이터가 로드되지 않았습니다.",
                "data_source": "bus_schedule",
                "total_time": time.time() - start_time,
                "rag_type": self.name
            }
        
        # 버스 정보 검색
        st.subheader("🚌 버스 정보 검색 중...")
        bus_results = self.processors['bus_schedule'].search_bus_routes(query)
        
        if not bus_results:
            return {
                "question": query,
                "answer": "요청하신 버스 정보를 찾을 수 없습니다. 다른 검색어를 시도해보세요.",
                "data_source": "bus_schedule",
                "search_results": [],
                "total_time": time.time() - start_time,
                "rag_type": self.name
            }
        
        # 검색 결과를 컨텍스트로 구성
        context = self._format_bus_context(bus_results)
        
        # LLM을 사용해 자연스러운 답변 생성
        st.subheader("🤖 답변 생성 중...")
        answer = self._generate_answer(query, context, "버스 정보")
        
        total_time = time.time() - start_time
        
        return {
            "question": query,
            "answer": answer,
            "data_source": "bus_schedule",
            "search_results": bus_results,
            "total_time": total_time,
            "rag_type": self.name
        }
    
    def query_menu_info(self, query: str, date_filter: Optional[str] = None) -> Dict[str, Any]:
        """메뉴 정보를 검색합니다."""
        start_time = time.time()
        
        if 'menu' not in self.processors:
            return {
                "question": query,
                "answer": "메뉴 데이터가 로드되지 않았습니다.",
                "data_source": "menu",
                "total_time": time.time() - start_time,
                "rag_type": self.name
            }
        
        # 메뉴 정보 검색
        st.subheader("🍽️ 메뉴 정보 검색 중...")
        
        # 특별한 키워드 처리
        if '오늘' in query or 'today' in query.lower():
            menu_results = self.processors['menu'].get_today_menu()
        elif '이번주' in query or '주간' in query or 'weekly' in query.lower():
            menu_results = self.processors['menu'].get_weekly_menu()
        else:
            menu_results = self.processors['menu'].search_menu(query, date_filter)
        
        if not menu_results:
            return {
                "question": query,
                "answer": "요청하신 메뉴 정보를 찾을 수 없습니다. 다른 검색어를 시도해보세요.",
                "data_source": "menu",
                "search_results": [],
                "total_time": time.time() - start_time,
                "rag_type": self.name
            }
        
        # 검색 결과를 컨텍스트로 구성
        context = self._format_menu_context(menu_results)
        
        # LLM을 사용해 자연스러운 답변 생성
        st.subheader("🤖 답변 생성 중...")
        answer = self._generate_answer(query, context, "메뉴 정보")
        
        total_time = time.time() - start_time
        
        return {
            "question": query,
            "answer": answer,
            "data_source": "menu",
            "search_results": menu_results,
            "total_time": total_time,
            "rag_type": self.name
        }
    
    def _format_bus_context(self, bus_results: List[Dict[str, Any]]) -> str:
        """버스 검색 결과를 컨텍스트로 포맷팅합니다."""
        context_parts = []
        
        for bus in bus_results:
            context = f"""
버스 정보:
- 버스 번호: {bus.get('bus_number', 'N/A')}
- 노선명: {bus.get('route_name', 'N/A')}
- 노선 유형: {bus.get('route_type', 'N/A')}
- 운영업체: {bus.get('operator', 'N/A')}
- 출발지: {bus.get('departure_terminal', {}).get('name', 'N/A')} ({bus.get('departure_terminal', {}).get('address', 'N/A')})
- 도착지: {bus.get('arrival_terminal', {}).get('name', 'N/A')} ({bus.get('arrival_terminal', {}).get('address', 'N/A')})
- 거리: {bus.get('distance_km', 'N/A')}km
- 예상 소요시간: {bus.get('estimated_duration_minutes', 'N/A')}분
"""
            
            # 중간 정류장 정보
            if bus.get('intermediate_stops'):
                context += "- 중간 정류장: "
                stops = [stop.get('stop_name', '') for stop in bus.get('intermediate_stops', [])]
                context += " → ".join(stops) + "\n"
            
            # 요금 정보
            fares = bus.get('fares', {})
            if fares:
                context += f"- 요금: 성인 {fares.get('adult', {}).get('card', 'N/A')}원(카드), "
                context += f"학생 {fares.get('student', {}).get('card', 'N/A')}원(카드)\n"
            
            # 운행 시간
            schedule = bus.get('schedule', [])
            if schedule:
                sched = schedule[0]
                context += f"- 운행시간: {sched.get('departure_time', 'N/A')} ~ {sched.get('last_bus', 'N/A')}\n"
                context += f"- 배차간격: {sched.get('frequency_minutes', 'N/A')}분\n"
            
            # 편의시설
            if bus.get('facilities'):
                context += f"- 편의시설: {', '.join(bus.get('facilities', []))}\n"
            
            # 운행 정보
            if bus.get('notes'):
                context += f"- 운행 정보: {bus.get('notes', '')}\n"
            
            context_parts.append(context)
        
        return "\n" + "="*50 + "\n".join(context_parts)
    
    def _format_menu_context(self, menu_results: List[Dict[str, Any]]) -> str:
        """메뉴 검색 결과를 컨텍스트로 포맷팅합니다."""
        context_parts = []
        
        for menu in menu_results:
            context = f"""
메뉴 정보:
- 날짜: {menu.get('date', 'N/A')} ({menu.get('day_of_week', 'N/A')})
- 식사: {menu.get('meal_type', 'N/A')} ({menu.get('service_time', 'N/A')})
- 요리 종류: {menu.get('cuisine_type', 'N/A')}
- 총 칼로리: {menu.get('total_calories', 'N/A')}kcal
- 가격: {menu.get('price', 'N/A')}원
"""
            
            # 메뉴 아이템
            menu_items = menu.get('menu_items', [])
            if menu_items:
                context += "- 구성:\n"
                for item in menu_items:
                    context += f"  • {item.get('category', '')}: {item.get('name', '')} ({item.get('calories', 'N/A')}kcal)\n"
            
            # 알레르기 정보
            if menu.get('allergens'):
                context += f"- 알레르기 유발요소: {', '.join(menu.get('allergens', []))}\n"
            
            # 식단 옵션
            dietary_options = menu.get('dietary_options', {})
            if dietary_options:
                options = []
                if dietary_options.get('vegetarian_available'):
                    options.append('채식 가능')
                if dietary_options.get('vegan_available'):
                    options.append('비건 가능')
                if dietary_options.get('gluten_free'):
                    options.append('글루텐 프리')
                if dietary_options.get('halal'):
                    options.append('할랄')
                if options:
                    context += f"- 식단 옵션: {', '.join(options)}\n"
            
            context_parts.append(context)
        
        return "\n" + "="*50 + "\n".join(context_parts)
    
    def _generate_answer(self, query: str, context: str, data_type: str) -> str:
        """컨텍스트를 바탕으로 자연스러운 답변을 생성합니다."""
        instruction = f"""위 {data_type} 데이터를 바탕으로 질문에 대해 친절하고 정확한 답변을 제공해주세요. 
답변할 때는 다음 사항을 고려해주세요:
1. 가장 관련성이 높은 정보를 우선적으로 제시
2. 구체적인 수치와 세부사항 포함
3. 사용자가 이해하기 쉽도록 구조화된 형태로 답변
4. 추가로 도움이 될 만한 정보가 있다면 함께 제공

질문: {query}"""

        try:
            answer_placeholder = st.empty()
            full_response = ""
            
            for chunk in self.llm_manager.generate_response_stream(
                prompt=instruction,
                context=context
            ):
                full_response += chunk
                answer_placeholder.markdown(full_response + "▌")
            
            answer_placeholder.markdown(full_response)
            return full_response
            
        except Exception as e:
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    def get_system_info(self) -> Dict[str, Any]:
        """JSON RAG 시스템 정보를 반환합니다."""
        return {
            "name": self.name,
            "description": self.description,
            "components": [
                "JSON Data Processor",
                "Structured Search Engine", 
                "LLM Answer Generator"
            ],
            "features": [
                "구조화된 JSON 데이터 처리",
                "의미론적 검색 및 키워드 매칭",
                "실시간 데이터 접근",
                "컨텍스트 기반 자연어 답변 생성"
            ],
            "data_sources": list(self.processors.keys())
        } 