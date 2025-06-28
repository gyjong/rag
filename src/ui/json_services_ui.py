"""JSON-based services UI for bus schedule and menu information."""

import streamlit as st
import os
from datetime import datetime, timedelta
from typing import Optional

from ..rag_systems.json_rag import JSONRAG
from ..utils.llm_manager import LLMManager


class JSONServicesUI:
    """JSON 기반 서비스 UI 클래스."""
    
    def __init__(self):
        """Initialize JSON Services UI."""
        self.name = "JSON 서비스"
        self.description = "구조화된 데이터 기반 정보 조회 서비스"
        
    def setup_json_rag_system(self) -> Optional[JSONRAG]:
        """JSON RAG 시스템을 설정합니다."""
        if "json_rag_system" not in st.session_state:
            try:
                # LLM Manager 초기화
                llm_manager = LLMManager(
                    st.session_state.get("selected_llm_model", "llama3.2:latest"),
                    "http://localhost:11434",
                    temperature=st.session_state.get("llm_temperature", 0.1)
                )
                
                # JSON RAG 시스템 생성
                json_rag = JSONRAG(llm_manager)
                
                # 데이터 소스 추가
                bus_schedule_path = os.path.join("docs", "bus_schedule.json")
                menu_path = os.path.join("docs", "menu.json")
                
                if os.path.exists(bus_schedule_path):
                    json_rag.add_json_source("bus_schedule", bus_schedule_path, "bus_schedule")
                    st.success("✅ 버스 스케줄 데이터 로드됨")
                else:
                    st.warning("⚠️ 버스 스케줄 파일을 찾을 수 없습니다.")
                
                if os.path.exists(menu_path):
                    json_rag.add_json_source("menu", menu_path, "menu")
                    st.success("✅ 메뉴 데이터 로드됨")
                else:
                    st.warning("⚠️ 메뉴 파일을 찾을 수 없습니다.")
                
                st.session_state.json_rag_system = json_rag
                return json_rag
                
            except Exception as e:
                st.error(f"JSON RAG 시스템 초기화 실패: {str(e)}")
                return None
        
        return st.session_state.json_rag_system
    
    def render_bus_service(self):
        """버스 안내 서비스 UI를 렌더링합니다."""
        st.header("🚌 출퇴근 버스 안내")
        st.write("서울시 주요 버스 노선 정보를 조회할 수 있습니다.")
        
        # JSON RAG 시스템 설정
        json_rag = self.setup_json_rag_system()
        if not json_rag:
            st.error("JSON RAG 시스템을 초기화할 수 없습니다.")
            return
        
        # 버스 정보 검색 인터페이스
        st.subheader("🔍 버스 정보 검색")
        
        # 검색 방법 선택
        search_method = st.radio(
            "검색 방법:",
            ["직접 질문", "카테고리별 검색"],
            horizontal=True
        )
        
        if search_method == "직접 질문":
            # 자유 텍스트 검색
            st.write("**자연스러운 질문으로 버스 정보를 검색해보세요:**")
            
            # 예시 질문들
            example_queries = [
                "강남에서 종로로 가는 버스 있나요?",
                "1000번 버스 정보 알려주세요",
                "홍대에서 잠실로 가는 방법은?",
                "저상버스나 휠체어 탑승 가능한 버스 있나요?",
                "5500번 버스 요금이 얼마인가요?"
            ]
            
            selected_example = st.selectbox(
                "예시 질문:",
                ["직접 입력"] + example_queries
            )
            
            if selected_example != "직접 입력":
                query = st.text_area("질문:", value=selected_example, height=100)
            else:
                query = st.text_area("질문:", height=100, placeholder="예: 강남에서 종로로 가는 버스 알려주세요")
        
        else:
            # 카테고리별 검색
            st.write("**카테고리를 선택하여 검색하세요:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                search_category = st.selectbox(
                    "검색 카테고리:",
                    ["버스 번호", "출발지/도착지", "노선 유형", "편의시설"]
                )
            
            with col2:
                if search_category == "버스 번호":
                    search_term = st.text_input("버스 번호:", placeholder="예: 1000, 5500, 302")
                elif search_category == "출발지/도착지":
                    search_term = st.text_input("지역명:", placeholder="예: 강남, 종로, 홍대, 잠실")
                elif search_category == "노선 유형":
                    search_term = st.selectbox("노선 유형:", ["간선버스", "지선버스", "광역버스", "순환버스"])
                elif search_category == "편의시설":
                    search_term = st.selectbox("편의시설:", ["저상버스", "WiFi", "USB충전", "휠체어탑승가능"])
            
            query = f"{search_category}에서 {search_term} 관련 정보를 알려주세요"
        
        # 검색 실행
        if st.button("🔍 버스 정보 검색", type="primary"):
            if query.strip():
                with st.spinner("버스 정보를 검색하고 있습니다..."):
                    result = json_rag.query_bus_info(query.strip())
                    
                    # 결과 표시
                    st.subheader("📋 검색 결과")
                    st.write(result["answer"])
                    
                    # 검색된 버스 상세 정보 표시
                    if result.get("search_results"):
                        with st.expander(f"📊 상세 정보 ({len(result['search_results'])}개 노선)"):
                            for i, bus in enumerate(result['search_results']):
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
                    
                    # 성능 정보
                    st.caption(f"⏱️ 처리 시간: {result['total_time']:.2f}초")
            else:
                st.warning("검색할 내용을 입력해주세요.")
    
    def render_menu_service(self):
        """식단 안내 서비스 UI를 렌더링합니다."""
        st.header("🍽️ 구내식당 식단 안내")
        st.write("구내식당의 식단 정보를 조회할 수 있습니다.")
        
        # JSON RAG 시스템 설정
        json_rag = self.setup_json_rag_system()
        if not json_rag:
            st.error("JSON RAG 시스템을 초기화할 수 없습니다.")
            return
        
        # 메뉴 정보 검색 인터페이스
        st.subheader("🔍 식단 정보 검색")
        
        # 검색 방법 선택
        search_method = st.radio(
            "검색 방법:",
            ["직접 질문", "날짜별 검색", "메뉴별 검색"],
            horizontal=True,
            key="menu_search_method"
        )
        
        if search_method == "직접 질문":
            # 자유 텍스트 검색
            st.write("**자연스러운 질문으로 식단 정보를 검색해보세요:**")
            
            # 예시 질문들
            example_queries = [
                "오늘 메뉴 뭐예요?",
                "이번 주 식단표 보여주세요",
                "짜장면 언제 나와요?",
                "칼로리 낮은 메뉴 추천해주세요",
                "채식 메뉴 있나요?",
                "중식 요리 언제 먹을 수 있어요?"
            ]
            
            selected_example = st.selectbox(
                "예시 질문:",
                ["직접 입력"] + example_queries,
                key="menu_example_select"
            )
            
            if selected_example != "직접 입력":
                query = st.text_area("질문:", value=selected_example, height=100, key="menu_query_text")
            else:
                query = st.text_area("질문:", height=100, placeholder="예: 오늘 점심 메뉴 뭐예요?", key="menu_query_input")
        
        elif search_method == "날짜별 검색":
            # 날짜별 검색
            st.write("**특정 날짜의 식단을 검색하세요:**")
            
            col1, col2 = st.columns(2)
            with col1:
                date_option = st.selectbox(
                    "날짜 선택:",
                    ["오늘", "내일", "이번 주", "특정 날짜"]
                )
            
            with col2:
                if date_option == "특정 날짜":
                    selected_date = st.date_input(
                        "날짜:",
                        value=datetime.now().date(),
                        min_value=datetime(2024, 1, 1).date(),
                        max_value=datetime(2024, 12, 31).date()
                    )
                    query = f"{selected_date.strftime('%Y-%m-%d')} 식단 알려주세요"
                else:
                    query = f"{date_option} 식단 알려주세요"
        
        else:
            # 메뉴별 검색
            st.write("**특정 메뉴나 요리를 검색하세요:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                menu_category = st.selectbox(
                    "검색 카테고리:",
                    ["음식 이름", "요리 종류", "식사 시간", "칼로리", "알레르기"]
                )
            
            with col2:
                if menu_category == "음식 이름":
                    search_term = st.text_input("음식 이름:", placeholder="예: 불고기, 짜장면, 스파게티")
                elif menu_category == "요리 종류":
                    search_term = st.selectbox("요리 종류:", ["한식", "중식", "양식", "일식"])
                elif menu_category == "식사 시간":
                    search_term = st.selectbox("식사 시간:", ["조식", "중식", "석식"])
                elif menu_category == "칼로리":
                    search_term = st.selectbox("칼로리:", ["500kcal 이하", "500-800kcal", "800kcal 이상"])
                elif menu_category == "알레르기":
                    search_term = st.selectbox("알레르기:", ["우유", "계란", "대두", "밀", "견과류"])
            
            query = f"{menu_category}에서 {search_term} 관련 메뉴 알려주세요"
        
        # 검색 실행
        if st.button("🔍 식단 정보 검색", type="primary"):
            if query.strip():
                with st.spinner("식단 정보를 검색하고 있습니다..."):
                    result = json_rag.query_menu_info(query.strip())
                    
                    # 결과 표시
                    st.subheader("📋 검색 결과")
                    st.write(result["answer"])
                    
                    # 검색된 메뉴 상세 정보 표시
                    if result.get("search_results"):
                        with st.expander(f"📊 상세 정보 ({len(result['search_results'])}개 메뉴)"):
                            for i, menu in enumerate(result['search_results']):
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
                                    
                                    dietary_options = menu.get('dietary_options', {})
                                    if dietary_options:
                                        options = []
                                        if dietary_options.get('vegetarian_available'):
                                            options.append('채식')
                                        if dietary_options.get('vegan_available'):
                                            options.append('비건')
                                        if options:
                                            st.write(f"**식단 옵션:** {', '.join(options)}")
                                
                                # 메뉴 구성
                                menu_items = menu.get('menu_items', [])
                                if menu_items:
                                    st.write("**메뉴 구성:**")
                                    for item in menu_items[:5]:  # 상위 5개만 표시
                                        st.write(f"• {item.get('category', '')}: {item.get('name', '')} ({item.get('calories', 'N/A')}kcal)")
                                
                                st.divider()
                    
                    # 성능 정보
                    st.caption(f"⏱️ 처리 시간: {result['total_time']:.2f}초")
            else:
                st.warning("검색할 내용을 입력해주세요.")
    
    def render(self):
        """메인 JSON 서비스 UI를 렌더링합니다."""
        st.title("🏢 JSON 기반 정보 서비스")
        st.write("구조화된 데이터를 활용한 실시간 정보 조회 서비스입니다.")
        
        # 서비스 선택
        service_tabs = st.tabs(["🚌 출퇴근 버스 안내", "🍽️ 구내식당 식단 안내"])
        
        with service_tabs[0]:
            self.render_bus_service()
        
        with service_tabs[1]:
            self.render_menu_service()
        
        # 서비스 정보
        with st.expander("ℹ️ 서비스 정보"):
            st.write("""
            **JSON 기반 정보 서비스**
            
            이 서비스는 구조화된 JSON 데이터를 활용하여 실시간으로 정보를 제공합니다.
            
            **주요 기능:**
            - 🚌 **버스 안내**: 서울시 주요 버스 노선 정보 조회
            - 🍽️ **식단 안내**: 구내식당 주간 식단표 및 메뉴 정보
            - 🔍 **지능형 검색**: 자연어로 질문하여 정확한 답변 받기
            - ⚡ **실시간 처리**: 빠른 검색과 즉시 응답
            
            **데이터 소스:**
            - 버스 정보: `docs/bus_schedule.json`
            - 메뉴 정보: `docs/menu.json`
            """) 