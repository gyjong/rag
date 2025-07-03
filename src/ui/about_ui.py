"""About and documentation UI components for the RAG application."""

import streamlit as st
from typing import Dict, Any


class AboutUI:
    """UI components for about and documentation sections."""

    @staticmethod
    def display_about_tab():
        """Display the complete about page with all sections."""
        st.header("ℹ️ RAG 시스템 소개")

        # Main purpose section
        AboutUI.display_main_purpose()

        # System comparison sections
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

        이 애플리케이션은 **최신 RAG 기술을 활용한** 세 가지 차별화된 RAG (Retrieval-Augmented Generation) 시스템을
        직접 비교하고 실험할 수 있는 **종합 벤치마킹 플랫폼**입니다.
        """)

    @staticmethod
    def display_system_comparisons():
        """Display detailed comparison of the three RAG systems."""
        # Naive RAG
        with st.expander("📚 **Naive RAG** - 기본형 시스템", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                ### 🔧 **구성**
                - **검색**: 단순 벡터 유사도 (Dense Retrieval)
                - **생성**: 직접 LLM 호출
                - **최적화**: 없음

                ### ✨ **특징**
                - ⚡ **초고속 처리**: 최소한의 오버헤드
                - 🎯 **단순함**: 이해하기 쉬운 구조
                - 📦 **경량화**: 최소 리소스 사용
                """)
            with col2:
                st.info("**⚡ 속도 우선**\n\n빠른 프로토타이핑이나 \n실시간 응답이 필요한 \n환경에 최적화")

        # Advanced RAG
        with st.expander("🔧 **Advanced RAG** - Enhanced Search 시스템", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                ### 🔧 **Enhanced Search 구성**
                - **1단계**: 🔍 쿼리 전처리 & 최적화
                - **2단계**: 🧠 벡터 유사도 검색 (확장 범위)
                - **3단계**: 🔀 **하이브리드 재순위화**
                  - TF-IDF 점수 (70%) + 벡터 유사도 (30%)
                - **4단계**: 📦 **스마트 컨텍스트 압축**
                  - 키워드 기반 중요도 스코어링
                  - 동적 길이 조절
                - **5단계**: 🤖 **실시간 스트리밍 생성**

                ### ✨ **핵심 혁신**
                - 🔀 **하이브리드 재순위화**: 의미적 + 통계적 검색 결합
                - 📦 **지능형 압축**: 중요 정보 보존하며 토큰 효율화
                - ⚡ **스트리밍**: 실시간 답변 생성 및 표시
                """)
            with col2:
                st.success("**📦 균형 최적화**\n\n정확성과 효율성의 \n완벽한 밸런스를 \n추구하는 시스템")

        # Modular RAG
        with st.expander("🧩 **Modular RAG** - 지능형 모듈러 시스템", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                ### 🧠 **Pre-Retrieval Modules**
                - **🎯 Query Classification**: 7가지 질문 유형 분류
                  - factual(사실형), procedural(방법형), causal(원인형)
                  - temporal(시간형), comparative(비교형), quantitative(수치형), general(일반형)
                - **🔍 Query Expansion**: 키워드 매핑 기반 확장

                ### 🔍 **Retrieval Modules**
                - **🧠 Semantic Retrieval**: Dense 벡터 검색
                - **🔤 BM25 Keyword Retrieval**: Sparse 키워드 검색
                  - 독립적인 BM25 구현 (k1=1.5, b=0.75)
                  - 한국어 토크나이징 지원

                ### 🔧 **Post-Retrieval Modules**
                - **📊 Relevance Filtering**: 관련도 기반 필터링
                - **🎲 Diversity Module**: 중복 제거 및 다양성 보장

                ### 🤖 **Generation Modules**
                - **🎯 Type-Specific Generation**: 질문 유형별 맞춤 프롬프트
                - **📊 Confidence Estimation**: 답변 신뢰도 평가

                ### 🔄 **Orchestration Modules**
                - **🛤️ Routing**: 질문 유형별 최적 처리 경로 선택
                - **🔄 Iteration Control**: 신뢰도 기반 반복 개선 (< 0.7시 재시도)
                """)
            with col2:
                st.warning("**🎯 정밀 최적화**\n\n복잡한 질문과 \n높은 정확성이 \n요구되는 고급 용도")

    @staticmethod
    def display_technical_innovations():
        """Display technical innovations section."""
        st.markdown("""
        ## 🚀 **기술적 혁신사항**

        ### 🔀 **하이브리드 검색 (Advanced & Modular)**
        - **Dense + Sparse 결합**: 의미적 검색과 키워드 매칭의 시너지
        - **동적 가중치**: TF-IDF 70% + 벡터 유사도 30% 최적 비율
        - **BM25 완전 구현**: 외부 의존성 없는 순수 Python 구현

        ### 🎯 **지능형 질문 분류 (Modular)**
        - **7가지 질문 유형**: 세밀한 의도 파악 및 맞춤 처리
        - **동적 신뢰도 계산**: 매칭 키워드 패턴 분석
        - **시각화 지원**: 분류 과정의 투명한 표시

        ### 📦 **스마트 컨텍스트 관리 (Advanced)**
        - **키워드 기반 압축**: 중요 문장 자동 추출
        - **동적 길이 조절**: 질문 복잡도에 따른 컨텍스트 크기 최적화
        - **토큰 효율성**: 최대 50% 압축률로 비용 절감

        ### 🔄 **반복적 개선 (Modular)**
        - **신뢰도 임계값**: 0.7 미만시 자동 재시도
        - **매개변수 조정**: 검색 범위 동적 확대 (k → k+2)
        - **최대 반복 제한**: 무한 루프 방지

        ### 📄 **JSON 기반 문서 관리**
        - **구조화된 저장**: 원본 문서와 청크 분리 저장
        - **메타데이터 풍부화**: 생성 시간, 파일 정보, 처리 설정
        - **재사용성**: PDF 재파싱 없이 빠른 로딩
        - **투명성**: 모든 처리 과정 JSON으로 추적 가능
        """)

    @staticmethod
    def display_tech_stack():
        """Display technical stack information."""
        st.markdown("""
        ## 🛠️ **고급 기술 스택**

        ### 🧠 **AI/ML 코어**
        - **LLM**: Ollama Gemma 3 (1B~27B, QAT 최적화)
        - **Embeddings**: Multilingual E5-Large-Instruct (다국어 지원)
        - **검색 알고리즘**:
          - Dense Retrieval (FAISS/ChromaDB)
          - BM25 Sparse Retrieval (순수 구현)
          - TF-IDF 통계 분석

        ### 🔧 **프레임워크 & 도구**
        - **Frontend**: Streamlit (실시간 UI)
        - **Backend**: LangChain + 커스텀 모듈
        - **벡터 DB**: ChromaDB (영구 저장, 텔레메트리 비활성화)
        - **시각화**: Plotly (인터랙티브 차트)
        - **폰트**: Paperlogy (커스텀 디자인)
        - **데이터 관리**: JSON 구조화 저장

        ### 📊 **성능 모니터링**
        - **실시간 메트릭**: 처리 시간, 검색 점수, 신뢰도
        - **상세 분석**: 검색 점수 분포, 압축률, 반복 횟수
        - **비교 시각화**: 시스템별 성능 벤치마킹
        - **Arrow 최적화**: 데이터프레임 직렬화 최적화
        """)

    @staticmethod
    def display_usage_guide():
        """Display usage guide and system selection recommendations."""
        st.markdown("""
        ## 📖 **단계별 사용 가이드**

        ### 🎯 **질문 유형별 최적 시스템 선택**
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("""
            **⚡ Naive RAG**

            ✅ **적합한 경우:**
            - 빠른 프로토타이핑
            - 실시간 응답 필요
            - 단순한 정보 조회
            - 리소스 제약 환경

            📝 **예시 질문:**
            - "AI란 무엇인가요?"
            - "현재 트렌드는?"
            """)

        with col2:
            st.success("""
            **📦 Advanced RAG**

            ✅ **적합한 경우:**
            - 정확성과 속도 균형
            - 중급 복잡도 질문
            - 효율적 토큰 사용
            - 비용 최적화 필요

            📝 **예시 질문:**
            - "AI가 비즈니스에 미치는 영향은?"
            - "머신러닝과 딥러닝의 차이점?"
            """)

        with col3:
            st.warning("""
            **🎯 Modular RAG**

            ✅ **적합한 경우:**
            - 복잡한 분석 질문
            - 최고 정확도 요구
            - 다양한 질문 유형
            - 상세한 프로세스 추적

            📝 **예시 질문:**
            - "AI 도입 전략을 단계별로 설명해주세요"
            - "2025년과 2024년 AI 트렌드를 비교 분석해주세요"
            """)

        st.markdown("""
        ### 🚀 **실험 진행 단계**

        1. **📚 문서 로딩** → PDF 문서 자동 파싱 및 청크 분할
           - JSON 형태로 구조화 저장 (documents.json + chunks.json)
           - 메타데이터 풍부화 및 통계 정보 제공
        2. **🔍 벡터 스토어** → 임베딩 생성 및 인덱스 구축
           - ChromaDB 텔레메트리 완전 비활성화
           - FAISS 또는 ChromaDB 선택 가능
        3. **🧪 RAG 실험** → 시스템별 질문 처리 및 답변 생성
           - 10가지 샘플 질문 (유형별 분류)
           - 실시간 처리 과정 시각화
        4. **📊 결과 비교** → 성능 메트릭 분석 및 시각화
           - Arrow 최적화된 데이터프레임
           - 인터랙티브 차트 및 상세 분석

        ### 🎨 **고급 UI 기능**

        - **🎯 질문 유형 가이드**: 10가지 샘플 질문 (유형별 분류)
        - **📊 실시간 메트릭**: 처리 시간, 신뢰도, 검색 점수
        - **🔍 상세 분석**: 검색 과정 시각화, 점수 분포 차트
        - **⚡ 스트리밍**: Advanced RAG 실시간 답변 생성
        - **🔄 프로세스 추적**: Modular RAG 모듈별 처리 과정
        - **📄 JSON 관리**: 문서 데이터 구조화 저장 및 재사용
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
            - **Storage**: 2GB+ (모델 및 벡터 저장)
            - **Models**: Gemma 3 다운로드 필요

            ### 🚀 **권장 사양 (고성능)**
            - **RAM**: 8GB+ (대용량 문서 처리)
            - **GPU**: CUDA 지원 (선택사항)
            - **SSD**: 빠른 벡터 검색
            - **Network**: 안정적 연결 (모델 다운로드)
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

            ### 📊 **성능 벤치마크**
            - **Naive RAG**: ~1-2초 (경량)
            - **Advanced RAG**: ~3-5초 (균형)
            - **Modular RAG**: ~5-8초 (정밀)

            *Gemma 3 4B 모델 기준
            """)

    @staticmethod
    def display_application_value():
        """Display application value and implementation statistics."""
        st.markdown("""
        ## 🎉 **이 애플리케이션의 가치**

        📚 **교육적 가치**: RAG 기술의 발전 과정을 단계적으로 체험
        🔬 **연구 도구**: 다양한 RAG 접근법의 성능 비교 분석
        ⚡ **실용성**: 실제 비즈니스 환경에서의 RAG 시스템 선택 가이드
        🚀 **혁신성**: 최신 RAG 기술들의 실제 구현 및 벤치마킹
        🛡️ **안정성**: 완전한 오류 해결 및 최적화 (ChromaDB, Arrow 등)

        > **"단순한 벡터 검색부터 고도화된 모듈러 아키텍처까지,
        > RAG 기술의 모든 것을 한 곳에서 경험해보세요!"**
        """)

        # Implementation statistics
        with st.expander("📊 **구현 통계**"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                ### 🧩 **시스템 구성**
                - **총 모듈 수**: 12개 (Modular RAG)
                - **질문 분류**: 7가지 유형
                - **검색 방법**: 3가지 (Dense, Sparse, Hybrid)
                - **성능 메트릭**: 15+ 지표
                - **최적화 기법**: 5가지 (압축, 재순위화, 반복 등)
                """)
            with col2:
                st.markdown("""
                ### 🎨 **UI & 기능**
                - **UI 컴포넌트**: 20+ 인터랙티브 요소
                - **탭 구조**: 5개 주요 탭
                - **JSON 관리**: 완전 구조화 데이터 저장
                - **오류 해결**: ChromaDB 텔레메트리, Arrow 직렬화 등
                - **시각화**: Plotly 인터랙티브 차트
                """)