"""JSON-based services UI for bus schedule and menu information."""

import streamlit as st
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json

from ..rag_systems.json_rag import JSONDataProcessor
from ..utils.llm_manager import LLMManager
from ..graphs.json_rag_graph import create_json_rag_graph, JsonRagState
from ..rag_systems import json_rag as json_rag_utils


class JSONServicesUI:
    """JSON 기반 서비스 UI 클래스."""
    
    def __init__(self):
        """Initialize JSON Services UI."""
        self.name = "JSON 서비스"
        self.description = "구조화된 데이터 기반 정보 조회 서비스"
        
    @st.cache_resource
    def get_json_processors(_self) -> Dict[str, Optional[JSONDataProcessor]]:
        """JSON 데이터 프로세서를 로드하고 캐시합니다."""
        processors = {"bus": None, "menu": None}
        try:
            bus_schedule_path = os.path.join("docs", "bus_schedule.json")
            if os.path.exists(bus_schedule_path):
                processors["bus"] = JSONDataProcessor(bus_schedule_path, "bus_schedule")
                st.success("✅ 버스 스케줄 데이터 로드됨")
            else:
                st.warning("⚠️ 버스 스케줄 파일을 찾을 수 없습니다.")

            menu_path = os.path.join("docs", "menu.json")
            if os.path.exists(menu_path):
                processors["menu"] = JSONDataProcessor(menu_path, "menu")
                st.success("✅ 메뉴 데이터 로드됨")
            else:
                st.warning("⚠️ 메뉴 파일을 찾을 수 없습니다.")
        except Exception as e:
            st.error(f"데이터 프로세서 초기화 실패: {e}")
        return processors

    @st.cache_resource
    def get_json_rag_graph(_self, _llm_manager: LLMManager):
        """JSON RAG 그래프를 생성하고 캐시합니다."""
        try:
            graph = create_json_rag_graph(_llm_manager)
            st.success("✅ JSON RAG 그래프 생성됨")
            return graph
        except Exception as e:
            st.error(f"JSON RAG 그래프 생성 실패: {e}")
            return None
            
    def run_graph(self, graph, query: str, service_type: str, processors: Dict[str, Optional[JSONDataProcessor]]):
        """그래프를 실행하고 결과를 스트리밍으로 표시합니다."""
        st.subheader("📋 검색 결과")
        answer_placeholder = st.empty()
        full_response = ""
        
        inputs: JsonRagState = {
            "query": query,
            "service_type": service_type,
            "bus_processor": processors.get("bus"),
            "menu_processor": processors.get("menu")
        }

        with st.spinner(f"{'버스' if service_type == 'bus' else '식단'} 정보를 검색하고 답변을 생성 중입니다..."):
            final_state = graph.invoke(inputs)

            answer = final_state.get("answer", "결과를 생성하지 못했습니다.")
            answer_placeholder.markdown(answer)

            # 검색된 상세 정보 표시
            search_results = final_state.get("search_results", [])
            if search_results:
                if service_type == "bus":
                    self.display_bus_details(search_results)
                elif service_type == "menu":
                    self.display_menu_details(search_results)

    def display_bus_details(self, search_results: list):
        with st.expander(f"📊 상세 정보 ({len(search_results)}개 노선)"):
            for i, bus in enumerate(search_results):
                st.write(f"**{i+1}. {bus.get('bus_number', 'N/A')}번 - {bus.get('route_name', 'N/A')}**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**출발지:** {bus.get('departure_terminal', {}).get('name', 'N/A')}")
                    st.write(f"**도착지:** {bus.get('arrival_terminal', {}).get('name', 'N/A')}")
                with col2:
                    st.write(f"**거리:** {bus.get('distance_km', 'N/A')}km")
                    st.write(f"**소요시간:** {bus.get('estimated_duration_minutes', 'N/A')}분")
                with col3:
                    fares = bus.get('fares', {})
                    if fares:
                        st.write(f"**성인 요금:** {fares.get('adult', {}).get('card', 'N/A')}원")
                        st.write(f"**학생 요금:** {fares.get('student', {}).get('card', 'N/A')}원")
                if bus.get('facilities'):
                    st.write(f"**편의시설:** {', '.join(bus.get('facilities', []))}")
                st.divider()

    def display_menu_details(self, search_results: list):
        with st.expander(f"📊 상세 정보 ({len(search_results)}개 메뉴)"):
            for i, menu in enumerate(search_results):
                st.write(f"**{i+1}. {menu.get('date', 'N/A')} ({menu.get('day_of_week', 'N/A')}) - {menu.get('meal_type', 'N/A')}**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**요리 종류:** {menu.get('cuisine_type', 'N/A')}")
                    st.write(f"**서빙 시간:** {menu.get('service_time', 'N/A')}")
                with col2:
                    st.write(f"**총 칼로리:** {menu.get('total_calories', 'N/A')}kcal")
                    st.write(f"**가격:** {menu.get('price', 'N/A')}원")
                with col3:
                    if menu.get('allergens'):
                        st.write(f"**알레르기:** {', '.join(menu.get('allergens', []))}")
                    options = []
                    if menu.get('dietary_options', {}).get('vegetarian_available'): options.append('채식')
                    if menu.get('dietary_options', {}).get('vegan_available'): options.append('비건')
                    if options: st.write(f"**식단 옵션:** {', '.join(options)}")
                menu_items = menu.get('menu_items', [])
                if menu_items:
                    st.write("**메뉴 구성:**")
                    for item in menu_items[:5]:
                        st.write(f"• {item.get('category', '')}: {item.get('name', '')} ({item.get('calories', 'N/A')}kcal)")
                st.divider()

    def render_bus_service(self, graph, processors):
        """버스 안내 서비스 UI를 렌더링합니다."""
        st.header("🚌 출퇴근 버스 안내")
        st.write("서울시 주요 버스 노선 정보를 조회할 수 있습니다.")
        if not graph or not processors.get("bus"):
            st.error("버스 안내 서비스를 사용할 수 없습니다. 시스템 설정을 확인하세요.")
            return

        st.subheader("🔍 버스 정보 검색")
        search_method = st.radio("검색 방법:", ["직접 질문", "카테고리별 검색"], horizontal=True, key="bus_search_method")
        
        query = ""
        if search_method == "직접 질문":
            example_queries = ["강남에서 종로로 가는 버스 있나요?", "1000번 버스 정보", "저상버스나 휠체어 탑승 가능한 버스"]
            selected_example = st.selectbox("예시 질문:", ["직접 입력"] + example_queries, key="bus_example_select")
            query = st.text_area("질문:", value=selected_example if selected_example != "직접 입력" else "", height=100, placeholder="예: 강남에서 종로로 가는 버스 알려주세요", key="bus_query_text")
        else:
            col1, col2 = st.columns(2)
            with col1:
                search_category = st.selectbox("검색 카테고리:", ["버스 번호", "출발지/도착지", "노선 유형", "편의시설"])
            with col2:
                if search_category == "버스 번호": search_term = st.text_input("버스 번호:", placeholder="예: 1000")
                elif search_category == "출발지/도착지": search_term = st.text_input("지역명:", placeholder="예: 강남")
                elif search_category == "노선 유형": search_term = st.selectbox("노선 유형:", ["간선버스", "지선버스", "광역버스"])
                else: search_term = st.selectbox("편의시설:", ["저상버스", "WiFi", "USB충전"])
            if search_term:
                query = f"{search_category}에서 {search_term} 관련 정보를 알려주세요"

        if st.button("🔍 버스 정보 검색", type="primary", key="bus_search_button"):
            if query.strip():
                self.run_graph(graph, query.strip(), "bus", processors)
            else:
                st.warning("검색할 내용을 입력해주세요.")
    
    def render_menu_service(self, graph, processors):
        """식단 안내 서비스 UI를 렌더링합니다."""
        st.header("🍽️ 구내식당 식단 안내")
        st.write("구내식당의 식단 정보를 조회할 수 있습니다.")
        if not graph or not processors.get("menu"):
            st.error("식단 안내 서비스를 사용할 수 없습니다. 시스템 설정을 확인하세요.")
            return

        st.subheader("🔍 식단 정보 검색")
        search_method = st.radio("검색 방법:", ["직접 질문", "날짜별 검색", "메뉴별 검색"], horizontal=True, key="menu_search_method")
        
        query = ""
        if search_method == "직접 질문":
            example_queries = ["오늘 메뉴 뭐예요?", "이번 주 식단표", "짜장면 언제 나와요?", "채식 메뉴 있나요?"]
            selected_example = st.selectbox("예시 질문:", ["직접 입력"] + example_queries, key="menu_example_select")
            query = st.text_area("질문:", value=selected_example if selected_example != "직접 입력" else "", height=100, placeholder="예: 오늘 점심 메뉴 뭐예요?", key="menu_query_text")
        elif search_method == "날짜별 검색":
            col1, col2 = st.columns(2)
            with col1:
                date_option = st.selectbox("날짜 선택:", ["오늘", "내일", "이번 주", "특정 날짜"])
            if date_option == "특정 날짜":
                with col2: selected_date = st.date_input("날짜:", value=datetime.now().date())
                query = f"{selected_date.strftime('%Y-%m-%d')} 식단 알려주세요"
            else:
                query = f"{date_option} 식단 알려주세요"
        else:
            col1, col2 = st.columns(2)
            with col1:
                menu_category = st.selectbox("검색 카테고리:", ["음식 이름", "요리 종류", "식사 시간", "알레르기"])
            with col2:
                if menu_category == "음식 이름": search_term = st.text_input("음식 이름:", placeholder="예: 불고기")
                elif menu_category == "요리 종류": search_term = st.selectbox("요리 종류:", ["한식", "중식", "양식"])
                elif menu_category == "식사 시간": search_term = st.selectbox("식사 시간:", ["조식", "중식", "석식"])
                else: search_term = st.selectbox("알레르기:", ["우유", "계란", "대두", "밀"])
            if search_term:
                query = f"{menu_category}에서 {search_term} 관련 메뉴 알려주세요"
        
        if st.button("🔍 식단 정보 검색", type="primary", key="menu_search_button"):
            if query.strip():
                self.run_graph(graph, query.strip(), "menu", processors)
            else:
                st.warning("검색할 내용을 입력해주세요.")
    
    @staticmethod
    def display_json_services_tab():
        """Handles the UI for JSON-based services like bus schedules and menus."""
        st.header("🏢 JSON 기반 정보 서비스")

        service_options = ["🚌 버스 운행 정보", "오늘의 메뉴"]
        selected_service = st.selectbox("조회할 서비스를 선택하세요:", service_options)

        if selected_service == "🚌 버스 운행 정보":
            JSONServicesUI._display_bus_schedule_ui()
        elif selected_service == "오늘의 메뉴":
            JSONServicesUI._display_menu_ui()
            
    @staticmethod
    def _display_bus_schedule_ui():
        st.subheader("🚌 버스 운행 정보 조회")
        query = st.text_input("버스 번호 또는 정류장 이름을 입력하세요:", key="bus_query")

        if query:
            results = json_rag_utils.search_bus_schedule(query)
            if results:
                st.success(f"'{query}'에 대한 검색 결과:")
                for route in results:
                    with st.expander(f"**{route['route_name']}** ({route['direction']}) - {route['status']}"):
                        st.write(f"**경로:** {route['path']}")
                        st.write(f"**운행 시간:** {route['operating_hours']}")
                        st.write(f"**배차 간격:** {route['headway']}")
                        st.write(f"**주요 정류장:** {', '.join(route['major_stops'])}")
            else:
                st.warning("검색 결과가 없습니다.")
    
    @staticmethod
    def _display_menu_ui():
        st.subheader("🍽️ 오늘의 메뉴 조회")
        query = st.text_input("메뉴 종류 또는 음식 이름을 입력하세요 (예: 한식, 점심):", key="menu_query", value="오늘의 점심 메뉴")

        if query:
            results = json_rag_utils.search_menu(query)
            if results:
                st.success(f"'{query}'에 대한 검색 결과:")
                for menu in results:
                    with st.expander(f"**{menu['category']}** - {menu['restaurant']}"):
                        st.write(f"**메뉴:** {menu['name']}")
                        st.write(f"**가격:** {menu['price']}원")
                        st.write(f"**설명:** {menu['description']}")
                        if menu.get('is_special_of_the_day'):
                            st.info("✨ 오늘의 특별 메뉴입니다!")
            else:
                st.warning("검색 결과가 없습니다.") 