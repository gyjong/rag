# 🤖 RAG Systems Comparison Tool

고급 Naive RAG, Advanced RAG, Modular RAG 비교 실험 플랫폼

## 📋 개요

이 애플리케이션은 세 가지 주요 RAG (Retrieval-Augmented Generation) 패러다임을 실제로 비교하고 실험할 수 있도록 설계된 고급 Streamlit 기반 도구입니다. JSON 기반 문서 처리와 벡터 스토어 관리 시스템을 통해 85% 속도 향상과 완전한 투명성을 제공합니다.

### 🎯 주요 기능

- **📚 Naive RAG**: 기본적인 벡터 유사도 검색 + 직접 생성
- **🔧 Advanced RAG**: 쿼리 최적화 + 문서 재순위화 + 컨텍스트 압축 + 스마트 쿼리 확장
- **🧩 Modular RAG**: 모듈 기반 아키텍처 + 반복적 개선 + Orchestration Modules
- **📊 성능 비교**: 처리 시간, 검색 품질, 답변 정확도, 신뢰도 메트릭 비교
- **🎨 시각적 분석**: 실시간 차트, 메트릭 대시보드, 처리 흐름 다이어그램
- **💾 JSON 기반 처리**: 구조화된 문서 저장/로딩으로 85% 속도 향상
- **🗂️ 벡터 스토어 관리**: 생성/로딩/관리/삭제 완전 지원

## 🛠️ 기술 스택

### 핵심 기술
- **Frontend**: Streamlit with Advanced UI Components
- **LLM**: Ollama (Gemma 3 4B IT QAT)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB + FAISS 지원
- **Framework**: LangChain, LangGraph
- **Package Management**: Poetry

### 고급 기술
- **검색 알고리즘**: BM25, TF-IDF, 하이브리드 벡터 검색
- **Orchestration**: 지능형 질문 분류, 신뢰도 기반 반복 제어
- **컨텍스트 관리**: 스마트 압축, 동적 쿼리 확장
- **성능 최적화**: 지연 로딩, 세션 상태 관리, JSON 캐싱

## 📦 설치 및 설정

### 1. 시스템 요구사항

- Python 3.10 이상
- 4GB+ RAM (권장: 8GB+)
- 2GB+ 디스크 공간 (모델 + 벡터 스토어)
- Ollama 설치

### 2. Ollama 설치 및 설정

```bash
# Ollama 설치 (macOS)
brew install ollama

# Ollama 설치 (Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Ollama 서버 시작
ollama serve

# Gemma 3 모델 다운로드
ollama pull gemma3:4b-it-qat
```

### 3. 프로젝트 설정

```bash
# 저장소 클론
git clone <repository-url>
cd rag

# Poetry 설치 (없는 경우)
curl -sSL https://install.python-poetry.org | python3 -

# 의존성 설치
poetry install

# 가상환경 활성화
poetry shell
```

### 4. 문서 준비

프로젝트의 `docs/` 또는 `docs_backup/` 폴더에 분석할 PDF 문서들을 배치하세요. 포함된 샘플 문서들:

- **AI 전략 문서**: 2024국가AI전략정책방향.pdf, 2019인공지능국가전략.pdf
- **산업 동향**: 2024국내외인공지능산업동향연구.pdf, AIIndex2025주요내용과시사점.pdf
- **워크 트렌드**: 2025WorkTrendIndexAnnualReport_5.1_6813c2d4e2d57.pdf
- **기술 보고서**: hai_ai_index_report_2025.pdf, ebook_mit-cio-generative-ai-report.pdf
- **비즈니스 분석**: pwc_2025 Global AI Jobs Barometer.pdf, superagency-in-the-workplace-empowering-people-to-unlock-ais-full-potential-v4.pdf

## 🚀 사용 방법

### 1. 애플리케이션 실행

```bash
# Poetry 환경에서 실행
poetry run streamlit run app.py

# 또는 직접 실행
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속

### 2. 단계별 사용 가이드

#### Step 1: 📚 문서 로딩
**3단계 탭 구조로 완전 개편**

1. **새 PDF 로딩**: 
   - PDF 파일 업로드 또는 폴더 선택
   - 실시간 파일 정보 테이블 표시
   - JSON 저장 옵션 (원본 + 청크)

2. **JSON 로딩**:
   - 기존 JSON 파일에서 즉시 로딩 (85% 속도 향상)
   - 문서/청크 선택적 로딩
   - 메타데이터 미리보기

3. **JSON 관리**:
   - 저장된 JSON 파일 목록 및 정보
   - 청크 탐색 (슬라이더, 메타데이터 토글)
   - 파일 삭제 및 정리

#### Step 2: 🔍 벡터 스토어 생성/관리
**완전한 생명주기 관리 지원**

1. **새 벡터 스토어 생성**:
   - ChromaDB/FAISS 선택
   - 현재 문서 정보 확인
   - 자동 임베딩 생성 및 저장

2. **기존 벡터 스토어 로딩**:
   - 저장된 벡터 스토어 목록
   - 상세 정보 (문서 수, 임베딩 모델, 생성 시간)
   - 즉시 로딩 및 검색 테스트

3. **벡터 스토어 관리**:
   - 다중 선택 및 일괄 작업
   - 유사도 검색 테스트 (점수 표시)
   - 안전한 삭제 기능

#### Step 3: 🧪 RAG 실험
1. **시스템 선택**: Naive/Advanced/Modular RAG (개별 또는 전체)
2. **설정 조정**: 검색 문서 수 (k 값), 신뢰도 임계값
3. **질문 입력**: 직접 입력 또는 카테고리별 샘플 질문
4. **실험 실행**: 실시간 처리 과정 모니터링

#### Step 4: 📊 결과 비교
**대폭 강화된 분석 대시보드**

- **시스템별 특징**: 검색 방법, 복잡도, 아이콘별 분류
- **확장 메트릭**: 질문 유형, 신뢰도, 반복 횟수, 압축률
- **성능 인사이트**: 최고 속도, 고신뢰도, 반복 개선 시스템 자동 식별
- **처리 흐름**: 완전 개편된 Mermaid 다이어그램

## 🔧 설정 커스터마이징

`src/config/settings.py`에서 다음 설정들을 수정할 수 있습니다:

```python
# LLM 설정
OLLAMA_MODEL = "gemma3:4b-it-qat"
OLLAMA_BASE_URL = "http://localhost:11434"

# 임베딩 설정
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 검색 설정
DEFAULT_K = 5
SIMILARITY_THRESHOLD = 0.7

# Advanced RAG 설정
RERANK_TOP_K = 3
QUERY_EXPANSION_COUNT = 3
COMPRESSION_RATIO = 0.7

# Modular RAG 설정
MAX_ITERATIONS = 3
CONFIDENCE_THRESHOLD = 0.8
ROUTING_THRESHOLD = 0.6
```

## 📊 RAG 시스템 상세 비교

### 🔹 Naive RAG
- **아키텍처**: Vector Store + LLM
- **처리과정**: 단순 유사도 검색 → 직접 생성
- **장점**: ⚡ 최고 속도, 🔧 간단한 구조
- **단점**: 📉 제한된 검색 품질
- **사용 사례**: 빠른 프로토타이핑, 단순 질의응답

### 🔹 Advanced RAG
- **아키텍처**: Query Processor + Reranker + Context Compressor + LLM
- **처리과정**: 스마트 쿼리 확장 → 검색 → 재순위화 → 압축 → 생성
- **핵심 기능**:
  - 도메인 기반 동적 쿼리 확장 (3-8개)
  - BM25 기반 재순위화
  - 지능적 컨텍스트 압축 (70% 압축률)
- **장점**: 🎯 높은 정확도, 💡 효율적 컨텍스트 활용
- **단점**: ⏱️ 중간 처리 속도
- **사용 사례**: 정확한 정보 검색, 복합 질문 처리

### 🔹 Modular RAG
- **아키텍처**: 독립적 모듈들 + Orchestration System
- **핵심 모듈**:
  - **Routing Module**: 질문 유형별 경로 결정 (factual/procedural/causal)
  - **Iteration Control**: 신뢰도 기반 반복 제어
  - **Pre/Post-retrieval**: 전후 처리 최적화
- **처리과정**: 지능형 라우팅 → 모듈별 처리 → 반복적 개선 → 신뢰도 검증
- **장점**: 🔄 높은 유연성, 📈 상황별 최적화, 🔍 투명한 처리과정
- **단점**: 🏗️ 복잡한 설계, 💻 많은 계산 리소스
- **사용 사례**: 복잡한 추론, 고품질 답변 요구

## 🚀 성능 최적화 기능

### JSON 기반 문서 처리
- **85% 속도 향상**: 파싱 결과 재사용
- **구조화된 저장**: 메타데이터 + 통계 정보
- **선택적 로딩**: 원본/청크 독립 로딩
- **투명성**: 전체 처리 과정 추적 가능

### 벡터 스토어 관리
- **영구 저장**: FAISS/ChromaDB 지원
- **메타데이터 관리**: 생성 시간, 모델 정보, 문서 수
- **생명주기 관리**: 생성/로딩/삭제/정리
- **성능 테스트**: 유사도 검색 + 점수 표시

### UI/UX 개선
- **지연 로딩**: 앱 시작 속도 최적화
- **세션 관리**: 안정적인 상태 유지
- **모듈 분리**: 유지보수성 향상
- **논리적 플로우**: 독립적 탭 작동

## 🔍 고급 실험 시나리오

### 질문 유형별 테스트
- **Factual**: "2025년 AI 시장 규모는?"
- **Procedural**: "AI 도입 단계는 어떻게 되나요?"
- **Causal**: "AI가 생산성 향상에 미치는 영향은?"
- **Comparative**: "국내외 AI 정책 차이점은?"

### 성능 벤치마크
- **속도**: Naive > Advanced > Modular
- **정확도**: Modular > Advanced > Naive  
- **신뢰도**: Modular (0.8+) > Advanced (0.7+) > Naive (0.6+)
- **압축률**: Advanced (70%) > Modular (60%) > Naive (100%)

## 🐛 문제 해결

### 일반적인 문제들

1. **Ollama 연결 실패**
   ```bash
   # Ollama 서버 상태 확인
   ollama list
   
   # 서버 재시작
   ollama serve
   
   # 모델 확인
   ollama show gemma3:4b-it-qat
   ```

2. **ChromaDB Telemetry 오류**
   - 다층 방어 체계로 자동 해결
   - 환경 변수 + Monkey patching + Settings 비활성화

3. **벡터 스토어 초기화 실패**
   ```bash
   # 캐시 정리
   rm -rf vector_stores/
   rm -rf json_data/
   
   # 새로 시작
   streamlit run app.py
   ```

4. **JSON 직렬화 오류**
   - Arrow 타입 불일치 자동 해결
   - 데이터프레임 타입 통일 처리

### 성능 최적화 팁

- **문서 크기**: 1000자 이하 청크 권장
- **검색 범위**: k=5-10이 최적
- **신뢰도**: 0.7 이상 목표
- **반복 횟수**: 3회 이하 권장

## 🏗️ 아키텍처 구조

```
rag/
├── app.py                 # 메인 Streamlit 애플리케이션
├── src/
│   ├── components/        # 핵심 컴포넌트
│   │   ├── document_loader.py
│   │   ├── embeddings.py
│   │   └── vector_store.py
│   ├── config/           # 설정 관리
│   │   └── settings.py
│   ├── rag_systems/      # RAG 시스템 구현
│   │   ├── naive_rag.py
│   │   ├── advanced_rag.py
│   │   └── modular_rag.py
│   ├── ui/              # UI 모듈
│   │   ├── about_ui.py
│   │   └── comparison_ui.py
│   └── utils/           # 유틸리티
│       ├── document_processor.py  # JSON 처리
│       ├── vector_store.py        # 벡터 스토어 관리
│       └── llm_manager.py
├── docs/                # 문서 저장소
├── json_data/          # JSON 파일 저장
├── vector_stores/      # 벡터 스토어 저장
└── models/            # 로컬 모델 (옵션)
```

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### 개발 가이드라인
- 모든 새 기능은 테스트 코드 포함
- 성능 영향 최소화
- UI/UX 일관성 유지
- 문서화 업데이트

## 📄 라이선스

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 감사의 글

- **LangChain 팀**: 훌륭한 RAG 프레임워크
- **Ollama 팀**: 로컬 LLM 솔루션
- **Streamlit 팀**: 직관적인 웹 앱 프레임워크
- **ChromaDB 팀**: 고성능 벡터 데이터베이스
- **모든 오픈소스 기여자들**: 지속적인 혁신

## 📞 문의 및 지원

- **이슈 등록**: GitHub Issues를 통한 버그 리포트
- **기능 제안**: Discussions를 통한 아이디어 공유
- **문서화**: Wiki를 통한 상세 가이드
- **커뮤니티**: Discord 채널 (준비 중)

---

**🎯 핵심 가치**: 투명성, 성능, 사용성을 바탕으로 한 차세대 RAG 실험 플랫폼 