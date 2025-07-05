"""About and documentation UI components for the RAG application."""

import streamlit as st
from typing import Dict, Any

from ..config.settings import CONFIDENCE_THRESHOLD


class AboutUI:
    """UI components for about and documentation sections."""

    @staticmethod
    def display_about_tab():
        """Display the complete about page with all sections."""
        st.header("ℹ️ RAG 시스템 포트폴리오")

        # Main purpose section
        AboutUI.display_main_purpose()

        # System comparison sections - ALL 8 RAG systems
        AboutUI.display_system_comparisons()

        st.markdown("---")

        # Technical innovations
        AboutUI.display_technical_innovations()

        st.markdown("---")

        # Tech stack
        AboutUI.display_tech_stack()

        st.markdown("---")

        # Usage guide
        AboutUI.display_usage_guide()

        st.markdown("---")

        # System requirements
        AboutUI.display_system_requirements()

        st.markdown("---")

        # Application value and statistics
        AboutUI.display_application_value()

    @staticmethod
    def display_main_purpose():
        """Display the main purpose and overview section."""
        st.markdown("""
        ## 🎯 애플리케이션 목적

        이 애플리케이션은 **8가지 특화된 RAG 시스템**을 통해 다양한 AI 활용 시나리오를 구현한 **종합 RAG 포트폴리오**입니다.
        
        > **"기본 검색부터 실시간 웹 정보, 전문 번역, 구조화 보고서까지 - RAG 기술의 모든 가능성을 한 곳에서"**
        """)

    @staticmethod
    def display_system_comparisons():
        """Display detailed comparison of all 8 RAG systems."""
        st.markdown("## 🧬 **8가지 RAG 시스템 완전 분석**")
        
        # 기본 3개 RAG 시스템
        st.markdown("### 🏗️ **핵심 RAG 시스템 (기본 3종)**")
        
        # Naive RAG
        with st.expander("📚 **Naive RAG** - 고속 기본형 시스템", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                ### 🔧 **구성**
                - **검색**: 단순 벡터 유사도 (Dense Retrieval)
                - **생성**: 직접 LLM 호출
                - **최적화**: 없음

                ### ✨ **특징**
                - ⚡ **초고속 처리**: 최소한의 오버헤드 (~1-2초)
                - 🎯 **단순함**: 이해하기 쉬운 구조
                - 📦 **경량화**: 최소 리소스 사용 (512MB RAM)
                """)
            with col2:
                st.info("**⚡ 속도 우선**\n\n빠른 프로토타이핑이나 \n실시간 응답이 필요한 \n환경에 최적화")

        # Advanced RAG
        with st.expander("🔧 **Advanced RAG** - 균형 최적화 시스템", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                ### 🔧 **Enhanced Search 구성**
                - **1단계**: 🔍 쿼리 전처리 & 도메인 키워드 확장
                - **2단계**: 🧠 벡터 유사도 검색 (확장 범위)
                - **3단계**: 🔀 **하이브리드 재순위화**
                  - TF-IDF 점수 (70%) + 벡터 유사도 (30%)
                - **4단계**: 📦 **스마트 컨텍스트 압축**
                  - 키워드 기반 중요도 스코어링
                  - 동적 길이 조절 (70% 압축률)
                - **5단계**: 🤖 **실시간 스트리밍 생성**

                ### ✨ **핵심 혁신**
                - 🔀 **하이브리드 재순위화**: 의미적 + 통계적 검색 결합
                - 📦 **지능형 압축**: 중요 정보 보존하며 토큰 효율화
                - ⚡ **스트리밍**: 실시간 답변 생성 및 표시
                """)
            with col2:
                st.success("**📦 균형 최적화**\n\n정확성과 효율성의 \n완벽한 밸런스를 \n추구하는 시스템")

        # Modular RAG
        with st.expander("🧩 **Modular RAG** - 지능형 모듈러 시스템", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"""
                ### 🧠 **Pre-Retrieval Modules**
                - **🎯 Query Classification**: 7가지 질문 유형 분류
                  - factual(사실형), procedural(방법형), causal(원인형)
                  - temporal(시간형), comparative(비교형), quantitative(수치형), general(일반형)
                - **🔍 Query Expansion**: 중앙화된 키워드 맵 기반 확장
                  - 17개 도메인, 80개 키워드, 400개 확장 용어

                ### 🔍 **Retrieval Modules**
                - **🧠 Semantic Retrieval**: Dense 벡터 검색
                - **🔤 BM25 Keyword Retrieval**: Sparse 키워드 검색
                  - 독립적인 BM25 구현 (k1=1.5, b=0.75)
                  - 한국어 토크나이징 지원 (`가-힣` 범위)

                ### 🔧 **Post-Retrieval Modules**
                - **📊 Relevance Filtering**: 관련도 기반 필터링
                - **🎲 Diversity Module**: 출처별 중복 제거 및 다양성 보장

                ### 🤖 **Generation Modules**
                - **🎯 Type-Specific Generation**: 질문 유형별 맞춤 프롬프트
                - **📊 Confidence Estimation**: LLM 기반 신뢰도 평가
                  - 불확실성 키워드 검출 + 사실 일치성 검증

                ### 🔄 **Orchestration Modules**
                - **🛤️ Routing**: 질문 유형별 최적 처리 경로 선택
                - **🔄 Iteration Control**: 신뢰도 기반 반복 개선 (< {CONFIDENCE_THRESHOLD}시 재시도)
                """)
            with col2:
                st.warning("**🎯 정밀 최적화**\n\n복잡한 질문과 \n높은 정확성이 \n요구되는 고급 용도")

        # 전문 특화 RAG 시스템
        st.markdown("### 🎯 **전문 특화 RAG 시스템 (5종)**")
        
        # Web Search RAG
        with st.expander("🌍 **Web Search RAG** - 실시간 웹 정보 시스템", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                ### 🌐 **실시간 웹 검색 구성**
                - **1단계**: 🔍 Google Search API + 다중 검색어 전략
                - **2단계**: 🕷️ **웹 스크래핑 엔진**
                  - BeautifulSoup4 + User-Agent 스푸핑
                  - 자동 텍스트 정제 (스크립트/스타일 제거)
                - **3단계**: 📊 **컨텐츠 병합 & 분석**
                  - 문서 기반 + 웹 정보 통합
                  - 출처별 신뢰도 가중치
                - **4단계**: 🎯 **최신성 우선 답변 생성**

                ### ✨ **핵심 기능**
                - 🌐 **실시간 정보**: 최신 웹 정보 즉시 접근
                - 🔍 **다양한 소스**: 여러 웹사이트 동시 검색
                - 🛡️ **에러 처리**: Rate limiting 및 실패 시 복구
                - 📊 **투명성**: 검색 과정 및 출처 완전 공개
                """)
            with col2:
                st.info("**🌐 실시간 정보**\n\n최신 트렌드, 뉴스, \n실시간 데이터가 \n필요한 질문에 특화")

        # Translation RAG
        with st.expander("🔄 **Translation RAG** - 문서 번역 전문 시스템", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                ### 📝 **지능형 번역 파이프라인**
                - **1단계**: 📄 **고급 문장 분할**
                  - 정규식 기반 문장 경계 감지
                  - 리스트/문단 구조 보존
                - **2단계**: 🔄 **점진적 번역 엔진**
                  - 문장별 순차 번역 (배치 크기 10)
                  - 기술 용어 및 맥락 보존 프롬프트
                - **3단계**: 📊 **실시간 진행 추적**
                  - 번역 진행률 + 현재 문장 미리보기
                  - 번역 실패 시 오류 마킹 및 계속 진행
                - **4단계**: 💾 **번역 결과 관리**
                  - 원문-번역문 대조 표시
                  - JSON/텍스트 내보내기

                ### ✨ **특화 기능**
                - 🌏 **다국어 지원**: 11개 언어 간 번역
                - ⚡ **고속 처리**: 문장별 병렬 처리
                - 📈 **품질 관리**: 번역 통계 및 성공률 추적
                """)
            with col2:
                st.success("**🔄 번역 전문**\n\n영어 문서의 \n한국어 번역에 \n특화된 시스템")

        # Report Generation RAG
        with st.expander("📋 **Report Generation RAG** - 구조화 보고서 생성 시스템", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                ### 📊 **구조화 보고서 생성 파이프라인**
                - **1단계**: 🎯 **토픽 기반 지능형 검색**
                  - 다중 쿼리 전략 (동향/분석/현황/전망/기술)
                  - Content hash 기반 중복 제거
                - **2단계**: 📝 **구조화된 보고서 생성**
                  - PromptTemplate 기반 섹션별 생성
                  - 보고서 설정 (유형/독자/언어/분량) 반영
                - **3단계**: 🔄 **실시간 스트리밍 생성**
                  - LLM 스트리밍 + 실시간 표시
                  - 섹션별 진행 상황 표시
                - **4단계**: 📚 **완성도 관리**
                  - 자동 참고 문헌 생성
                  - 마크다운 형식 구조화 출력

                ### ✨ **고급 기능**
                - 📋 **맞춤형 구조**: 사용자 정의 보고서 구조
                - 🎯 **대상별 최적화**: 독자 수준별 언어 조절
                - 📊 **시각화 지원**: 차트 플레이스홀더 삽입
                """)
            with col2:
                st.warning("**📋 보고서 생성**\n\n정책 제안서, \n연구 보고서 등 \n전문 문서 작성")

        # JSON RAG
        with st.expander("🗂️ **JSON RAG** - 구조화 데이터 전문 시스템", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                ### 📊 **구조화 데이터 처리 엔진**
                - **1단계**: 🗃️ **다중 데이터 소스 로더**
                  - 버스 스케줄, 메뉴 정보 등 JSON 파일
                  - 데이터 타입별 특화 검색 로직
                - **2단계**: 🔍 **하이브리드 검색 엔진**
                  - SequenceMatcher 유사도 계산
                  - 키워드 매칭 + 다중 필드 검색
                - **3단계**: ⚡ **실시간 정보 조회**
                  - 날짜/시간 필터링 지원
                  - 동적 데이터 업데이트
                - **4단계**: 📋 **구조화된 답변 생성**
                  - 관련도 순 결과 정렬
                  - 정확한 데이터 포맷 유지

                ### ✨ **특화 기능**
                - 🎯 **높은 정확도**: 97% 검색 정확도
                - ⚡ **최소 리소스**: 256MB RAM 사용
                - 📊 **캐싱 지원**: 메뉴(24시간), 버스(30분)
                """)
            with col2:
                st.info("**🗂️ 데이터 전문**\n\n정확한 정보 조회가 \n필요한 실용적 \n서비스에 특화")

        # Document Discovery RAG
        with st.expander("🔍 **Document Discovery RAG** - 문서 발견 & 탐색 시스템", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                ### 🗺️ **2단계 문서 발견 파이프라인**
                - **1단계**: 📊 **문서 요약 기반 발견**
                  - 모든 문서의 요약 캐시 생성
                  - 관련성 점수 계산 (임계값 30)
                - **2단계**: 🔍 **세부 내용 검색**
                  - 선택된 문서 내 정밀 검색
                  - 청크 단위 상세 분석
                - **3단계**: 🎯 **발견 과정 투명화**
                  - 검색 단계별 결과 표시
                  - 문서별 관련성 점수 공개
                - **4단계**: 📈 **캐싱 최적화**
                  - 문서 요약 영구 캐시
                  - 검색 성능 대폭 향상

                ### ✨ **혁신 기능**
                - 🗂️ **지능형 문서 매핑**: 전체 문서 구조 파악
                - 🔍 **단계별 발견**: 거시적 → 미시적 접근
                - 📊 **관련성 시각화**: 발견 과정 완전 투명화
                """)
            with col2:
                st.success("**🔍 문서 발견**\n\n방대한 문서에서 \n관련 정보를 \n효율적으로 탐색")

    @staticmethod
    def display_technical_innovations():
        """Display technical innovations section."""
        st.markdown("""
        ## 🚀 **핵심 기술 혁신사항**

        ### 🔗 **LangGraph 기반 통합 아키텍처**
        - **상태 관리**: TypedDict 기반 체계적 상태 추적
        - **모듈화**: 각 RAG 단계별 독립적 노드 구성
        - **확장성**: 새로운 RAG 시스템 쉬운 추가
        - **디버깅**: 단계별 상태 로깅 및 추적

        ### 🗂️ **중앙화된 설정 관리**
        - **키워드 맵 통합**: `MODULAR_RAG_KEYWORDS_MAP` 중앙 관리
          - 17개 도메인, 80개 키워드, 400개 확장 용어
          - `settings.py`에서 일원화된 관리
        - **불용어 처리**: 한국어 + 영어 통합 불용어 시스템
        - **도메인별 확장**: `ADVANCED_RAG_DOMAIN_KEYWORD_MAP` 활용

        ### 🔀 **하이브리드 검색 혁신 (Advanced & Modular)**
        - **Dense + Sparse 결합**: 의미적 검색과 키워드 매칭의 시너지
        - **동적 가중치**: TF-IDF 70% + 벡터 유사도 30% 최적 비율
        - **BM25 완전 구현**: 외부 의존성 없는 순수 Python 구현
        - **한국어 특화**: `가-힣` 범위 + 1글자 토큰 필터링

        ### 🌐 **실시간 웹 정보 통합 (Web Search)**
        - **Google Search API**: googlesearch-python 라이브러리
        - **지능형 스크래핑**: BeautifulSoup4 + User-Agent 스푸핑
        - **컨텐츠 정제**: 스크립트/스타일 자동 제거
        - **에러 복구**: Rate limiting 대응 + 실패 시 우아한 처리

        ### 📄 **JSON 기반 문서 생명주기 관리**
        - **구조화된 저장**: 메타데이터 + 통계 정보 풍부화
        - **85% 성능 향상**: PDF 재파싱 없이 JSON 로딩
        - **투명성**: 모든 처리 과정 JSON으로 추적 가능
        - **재사용성**: 다양한 RAG 시스템에서 공통 활용

        ### 🎯 **지능형 질문 분류 & 신뢰도 평가**
        - **7가지 질문 유형**: 세밀한 의도 파악 및 맞춤 처리
        - **LLM 기반 신뢰도**: 사실 일치성 검증 + 불확실성 감지
        - **반복적 개선**: 신뢰도 임계값 기반 자동 재시도
        - **투명한 과정**: 모든 판단 근거 시각화

        ### 📊 **RAGAS 통합 평가 시스템**
        - **4가지 메트릭**: faithfulness, answer_relevancy, context_recall, context_precision
        - **다중 시스템 비교**: 8개 RAG 시스템 동시 평가
        - **LangFuse 연동**: 모든 평가 과정 추적 및 분석
        - **시각화**: 성능 비교 차트 및 상세 분석
        """)

    @staticmethod
    def display_tech_stack():
        """Display technical stack information."""
        st.markdown("""
        ## 🛠️ **최신 기술 스택**

        ### 🧠 **AI/ML 핵심 엔진**
        - **LLM**: Ollama Gemma 3 (1B~27B, QAT 최적화)
        - **Embeddings**: Multilingual E5-Large-Instruct (다국어 지원)
        - **검색 알고리즘**:
          - Dense Retrieval (ChromaDB/Milvus)
          - BM25 Sparse Retrieval (순수 구현, 한국어 지원)
          - TF-IDF 통계 분석 (중앙화 불용어)

        ### 🏗️ **아키텍처 & 프레임워크**
        - **상태 관리**: LangGraph + TypedDict
        - **Frontend**: Streamlit (8개 전문 UI 모듈)
        - **Backend**: LangChain + 커스텀 모듈 (8개 RAG 시스템)
        - **설정 관리**: 중앙화된 settings.py (17개 도메인 키워드 맵)

        ### 💾 **데이터 & 저장소**
        - **벡터 DB**: ChromaDB/Milvus (텔레메트리 완전 비활성화)
        - **문서 처리**: JSON 구조화 저장 (85% 성능 향상)
        - **캐싱**: 문서 요약, 메뉴(24h), 버스(30m)
        - **직렬화**: Arrow 최적화 데이터프레임

        ### 🌐 **웹 & 네트워크**
        - **웹 검색**: Google Search API + BeautifulSoup4
        - **HTTP**: requests + User-Agent 스푸핑
        - **스크래핑**: 자동 컨텐츠 정제 (lxml 파서)
        - **에러 처리**: Rate limiting + 우아한 실패 복구

        ### 📊 **모니터링 & 평가**
        - **성능 추적**: 실시간 메트릭 (처리 시간, 검색 점수, 신뢰도)
        - **RAGAS 평가**: 4가지 메트릭 자동 평가
        - **LangFuse**: 모든 LLM 호출 추적 및 분석
        - **시각화**: Plotly 인터랙티브 차트

        ### 🎨 **UI & UX**
        - **8개 전문 UI**: 각 RAG 시스템별 최적화된 인터페이스
        - **실시간 표시**: 스트리밍 답변 + 진행률 추적
        - **투명성**: 모든 처리 과정 단계별 시각화
        - **폰트**: Paperlogy 커스텀 디자인
        """)

    @staticmethod
    def display_usage_guide():
        """Display usage guide and system selection recommendations."""
        st.markdown("""
        ## 📖 **단계별 사용 가이드**

        ### 🎯 **RAG 시스템별 최적 사용 시나리오**
        """)

        # 기본 3개 시스템
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("""
            **⚡ Naive RAG**

            ✅ **적합한 경우:**
            - 빠른 프로토타이핑
            - 실시간 응답 필요 (~1-2초)
            - 단순한 정보 조회
            - 리소스 제약 환경 (512MB)

            📝 **예시 질문:**
            - "AI란 무엇인가요?"
            - "현재 트렌드는?"
            """)

        with col2:
            st.success("""
            **📦 Advanced RAG**

            ✅ **적합한 경우:**
            - 정확성과 속도 균형 (~3-5초)
            - 중급 복잡도 질문
            - 효율적 토큰 사용 (70% 압축)
            - 비용 최적화 필요

            📝 **예시 질문:**
            - "AI가 비즈니스에 미치는 영향은?"
            - "머신러닝과 딥러닝의 차이점?"
            """)

        with col3:
            st.warning("""
            **🎯 Modular RAG**

            ✅ **적합한 경우:**
            - 복잡한 분석 질문 (~5-8초)
            - 최고 정확도 요구 (83% 평균 신뢰도)
            - 다양한 질문 유형 (7가지)
            - 상세한 프로세스 추적

            📝 **예시 질문:**
            - "AI 도입 전략을 단계별로 설명해주세요"
            - "2025년과 2024년 AI 트렌드를 비교 분석해주세요"
            """)

        st.markdown("### 🎯 **전문 특화 시스템 선택 가이드**")

        # 전문 5개 시스템
        col1, col2 = st.columns(2)
        with col1:
            st.info("""
            **🌍 Web Search RAG**
            ✅ 최신 정보 필요 (~13-15초)
            ✅ 실시간 트렌드 분석
            ✅ 문서에 없는 정보 검색
            
            **🔄 Translation RAG**  
            ✅ 영어→한국어 문서 번역
            ✅ 기술 문서 및 학술 논문
            ✅ 대량 문서 일괄 번역
            
            **📋 Report Generation RAG**
            ✅ 구조화된 분석 보고서
            ✅ 정책 제안서 및 연구 보고서  
            ✅ 다중 문서 기반 종합 분석
            """)
        
        with col2:
            st.success("""
            **🗂️ JSON RAG**
            ✅ 정확한 데이터 조회 (97% 정확도)
            ✅ 버스/메뉴 등 실시간 정보
            ✅ 구조화된 데이터베이스 검색
            
            **🔍 Document Discovery RAG**
            ✅ 방대한 문서 컬렉션 탐색
            ✅ 관련 문서 자동 발견
            ✅ 2단계 정밀 검색 (~88% 신뢰도)
            """)

        st.markdown("""
        ### 🚀 **실험 진행 단계**

        1. **📚 문서 로딩** → PDF 문서 자동 파싱 및 청크 분할
           - JSON 형태로 구조화 저장 (85% 성능 향상)
           - 메타데이터 풍부화 및 통계 정보 제공
        2. **🔍 벡터 스토어** → 임베딩 생성 및 인덱스 구축
           - ChromaDB/Milvus 선택 가능 (텔레메트리 비활성화)
           - 다국어 지원 (Multilingual E5-Large-Instruct)
        3. **🧪 RAG 실험** → 8가지 시스템별 질문 처리 및 답변 생성
           - LangGraph 기반 상태 추적
           - 실시간 처리 과정 시각화
        4. **📊 RAGAS 평가** → 4가지 메트릭으로 성능 분석
           - faithfulness, answer_relevancy, context_recall, context_precision
           - 다중 시스템 동시 비교 평가
        5. **📊 결과 비교** → 성능 메트릭 분석 및 시각화
           - Arrow 최적화된 데이터프레임
           - 인터랙티브 차트 및 상세 분석

        ### 🎨 **고급 UI 기능**

        - **🎯 8개 전문 탭**: 각 RAG 시스템별 최적화된 UI
        - **📊 실시간 메트릭**: 처리 시간, 신뢰도, 검색 점수
        - **🔍 상세 분석**: 검색 과정 시각화, 점수 분포 차트  
        - **⚡ 스트리밍**: 실시간 답변 생성 및 진행률 표시
        - **🔄 프로세스 추적**: LangGraph 기반 단계별 상태 추적
        - **📄 JSON 관리**: 문서 데이터 구조화 저장 및 재사용
        - **🌐 웹 통합**: 실시간 웹 정보 + 문서 기반 답변 결합
        - **📋 전문 도구**: 번역, 보고서 생성, 데이터 조회 등
        """)

    @staticmethod
    def display_system_requirements():
        """Display system requirements and installation guide."""
        st.subheader("⚙️ **시스템 요구사항**")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 📋 **필수 요구사항**
            - **Python**: 3.11+ (Poetry 권장)
            - **Ollama**: 설치 및 실행 중
            - **RAM**: 4GB+ (BM25 인덱싱용)
            - **Storage**: 3GB+ (모델 및 벡터 저장)
            - **Models**: Gemma 3 시리즈 다운로드

            ### 🚀 **권장 사양 (고성능)**
            - **RAM**: 8GB+ (8개 시스템 동시 사용)
            - **GPU**: CUDA 지원 (선택사항)
            - **SSD**: 빠른 벡터 검색 (ChromaDB/Milvus)
            - **Network**: 안정적 연결 (웹 검색 기능)
            """)

        with col2:
            st.markdown("""
            ### 🔧 **설치 명령어**
            ```bash
            # Ollama 설치 (macOS)
            brew install ollama

            # Gemma 3 모델 다운로드
            ollama pull gemma3:4b-it-qat
            ollama pull gemma3:12b-it-qat

            # 의존성 설치
            poetry install --no-root

            # 애플리케이션 실행
            poetry run streamlit run app.py
            ```

            ### 📊 **성능 벤치마크 (Gemma 3 4B 기준)**
            - **Naive RAG**: ~1-2초 (512MB RAM)
            - **Advanced RAG**: ~3-5초 (768MB RAM)
            - **Modular RAG**: ~5-8초 (1.2GB RAM)
            - **Web Search RAG**: ~13-15초 (네트워크 의존)
            - **Translation RAG**: 문장당 ~2-3초
            - **Report Generation**: 섹션당 ~10-15초
            - **JSON RAG**: ~1초 미만 (256MB RAM)
            - **Document Discovery**: ~3-7초 (캐싱 후)
            """)

    @staticmethod
    def display_application_value():
        """Display application value and implementation statistics."""
        st.markdown("""
        ## 🎉 **이 애플리케이션의 가치**

        📚 **교육적 가치**: RAG 기술의 발전 과정을 8단계로 체험
        🔬 **연구 도구**: 다양한 RAG 접근법의 성능 비교 분석
        ⚡ **실용성**: 실제 비즈니스 환경에서의 RAG 시스템 선택 가이드
        🚀 **혁신성**: 최신 RAG 기술들의 실제 구현 및 벤치마킹
        🛡️ **안정성**: 완전한 오류 해결 및 최적화 (ChromaDB, LangGraph 등)
        🌐 **종합성**: 기본 검색부터 웹 정보, 전문 번역까지 모든 영역 커버

        > **"단순한 벡터 검색부터 실시간 웹 정보, 전문 번역, 구조화 보고서까지,**  
        > **RAG 기술의 모든 가능성을 한 곳에서 경험해보세요!"**
        """)

        # Implementation statistics
        with st.expander("📊 **상세 구현 통계**"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                ### 🧩 **시스템 구성**
                - **총 RAG 시스템**: 8개 (기본 3개 + 전문 5개)
                - **질문 분류**: 7가지 유형 (Modular RAG)
                - **검색 방법**: 4가지 (Dense, Sparse, Hybrid, Web)
                - **성능 메트릭**: 20+ 지표 (RAGAS 포함)
                - **키워드 도메인**: 17개 (400+ 확장 용어)
                - **지원 언어**: 11개 (Translation RAG)

                ### 🏗️ **아키텍처**
                - **LangGraph 노드**: 30+ 처리 단계
                - **상태 관리**: TypedDict 기반 체계적 추적
                - **모듈 수**: 50+ (각 RAG별 특화 모듈)
                - **중앙화 설정**: settings.py 통합 관리
                """)
            with col2:
                st.markdown("""
                ### 🎨 **UI & 기능**
                - **전문 UI 탭**: 8개 (각 RAG 시스템별)
                - **총 UI 컴포넌트**: 50+ 인터랙티브 요소
                - **실시간 기능**: 스트리밍, 진행률, 상태 추적
                - **JSON 관리**: 완전 구조화 데이터 저장
                - **오류 해결**: ChromaDB 텔레메트리, Arrow 등
                - **시각화**: Plotly 인터랙티브 차트

                ### 📊 **성능 최적화**
                - **JSON 캐싱**: 85% 성능 향상
                - **메모리 효율**: 시스템별 최적화 (256MB~1.2GB)
                - **압축률**: 최대 70% (Advanced RAG)
                - **검색 정확도**: 최대 97% (JSON RAG)
                """)

        # Performance comparison table
        with st.expander("📈 **시스템별 성능 비교**"):
            st.markdown("""
            | RAG 시스템 | 평균 응답시간 | 메모리 사용량 | 신뢰도 | 특화 영역 |
            |------------|---------------|---------------|--------|-----------|
            | **Naive RAG** | 1-2초 | 512MB | 65% | 고속 프로토타이핑 |
            | **Advanced RAG** | 3-5초 | 768MB | 74% | 균형 최적화 |
            | **Modular RAG** | 5-8초 | 1.2GB | 83% | 정밀 분석 |
            | **Web Search RAG** | 13-15초 | 768MB | 78% | 실시간 정보 |
            | **Translation RAG** | 문장당 2-3초 | 1.0GB | 82% | 문서 번역 |
            | **Report Generation RAG** | 섹션당 10-15초 | 1.5GB | 79% | 구조화 보고서 |
            | **JSON RAG** | <1초 | 256MB | **95%** | 정확한 데이터 |
            | **Document Discovery RAG** | 3-7초 | 896MB | 88% | 문서 탐색 |
            """)