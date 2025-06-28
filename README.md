# 🤖 RAG Systems Comparison Tool

단계별 Naive RAG, Advanced RAG, Modular RAG 비교 실험 애플리케이션

## 📋 개요

이 애플리케이션은 세 가지 주요 RAG (Retrieval-Augmented Generation) 패러다임을 실제로 비교하고 실험할 수 있도록 설계된 Streamlit 기반 도구입니다.

### 🎯 주요 기능

- **📚 Naive RAG**: 기본적인 벡터 유사도 검색 + 직접 생성
- **🔧 Advanced RAG**: 쿼리 최적화 + 문서 재순위화 + 컨텍스트 압축
- **🧩 Modular RAG**: 모듈 기반 아키텍처 + 반복적 개선
- **📊 성능 비교**: 처리 시간, 검색 품질, 답변 정확도 비교
- **🎨 시각적 분석**: 실시간 차트 및 메트릭 대시보드

## 🛠️ 기술 스택

- **Frontend**: Streamlit
- **LLM**: Ollama (Gemma 3 4B IT QAT)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB
- **Framework**: LangChain, LangGraph
- **Package Management**: Poetry

## 📦 설치 및 설정

### 1. 시스템 요구사항

- Python 3.10 이상
- 4GB+ RAM (권장: 8GB+)
- 2GB+ 디스크 공간
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
cd naive_advance_modular_rag

# Poetry 설치 (없는 경우)
curl -sSL https://install.python-poetry.org | python3 -

# 의존성 설치
poetry install

# 가상환경 활성화
poetry shell
```

### 4. 문서 준비

프로젝트의 `docs/` 폴더에 분석할 PDF 문서들을 배치하세요. 현재 포함된 문서들:

- `2025WorkTrendIndexAnnualReport_5.1_6813c2d4e2d57.pdf`
- `ebook_mit-cio-generative-ai-report.pdf`
- `hai_ai_index_report_2025.pdf`
- `superagency-in-the-workplace-empowering-people-to-unlock-ais-full-potential-v4.pdf`
- `trends_artificial_intelligence.pdf`

## 🚀 사용 방법

### 1. 애플리케이션 실행

```bash
# Poetry 환경에서 실행
poetry run streamlit run app.py

# 또는 가상환경 활성화 후
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속

### 2. 단계별 사용 가이드

#### Step 1: 📚 문서 로딩
1. "문서 로딩" 탭으로 이동
2. "문서 로딩 시작" 버튼 클릭
3. PDF 문서들이 자동으로 로딩되고 청크로 분할됨

#### Step 2: 🔍 벡터 스토어 생성
1. "벡터 스토어" 탭으로 이동
2. "벡터 스토어 생성" 버튼 클릭
3. 문서 임베딩이 생성되고 ChromaDB에 저장됨
4. 검색 테스트로 정상 작동 확인

#### Step 3: 🧪 RAG 실험
1. "RAG 실험" 탭으로 이동
2. 테스트할 RAG 시스템 선택 (기본: 모든 시스템)
3. 검색할 문서 수 설정 (k 값)
4. 질문 입력 또는 샘플 질문 선택
5. "실험 실행" 버튼 클릭

#### Step 4: 📊 결과 비교
1. "결과 비교" 탭으로 이동
2. 시스템별 성능 메트릭 확인
3. 답변 품질 비교
4. 처리 시간 및 검색 효율성 분석

## 🎨 UI 특징

### 사이드바 정보
- **🧠 LLM 상태**: Ollama 서버 및 모델 가용성
- **📚 문서 상태**: 로딩된 문서 및 청크 정보
- **🔧 설정**: 현재 설정값 표시

### 메인 대시보드
- **단계별 진행**: 각 단계별 상세 로그
- **실시간 메트릭**: 처리 시간, 검색 결과 수 등
- **시각적 비교**: Plotly 차트를 통한 성능 비교
- **상세 분석**: 시스템별 처리 흐름 다이어그램

## 🔧 설정 커스터마이징

`src/config.py`에서 다음 설정들을 수정할 수 있습니다:

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

# Modular RAG 설정
MAX_ITERATIONS = 3
CONFIDENCE_THRESHOLD = 0.8
```

## 📊 RAG 시스템 비교

### Naive RAG
- **구성요소**: Vector Store + LLM
- **처리과정**: 단순 유사도 검색 → 직접 생성
- **장점**: 빠른 속도, 간단한 구조
- **단점**: 제한된 검색 품질

### Advanced RAG
- **구성요소**: Query Preprocessor + Reranker + Context Compressor + LLM
- **처리과정**: 쿼리 최적화 → 검색 → 재순위화 → 압축 → 생성
- **장점**: 높은 정확도, 효율적 컨텍스트 활용
- **단점**: 복잡한 구조, 상대적으로 느린 처리

### Modular RAG
- **구성요소**: 독립적 모듈들 (Pre-retrieval, Retrieval, Post-retrieval, Generation, Orchestration)
- **처리과정**: 모듈별 처리 → 반복적 개선 → 신뢰도 기반 제어
- **장점**: 높은 유연성, 상황별 최적화, 투명한 처리과정
- **단점**: 복잡한 설계, 많은 계산 리소스

## 🔍 샘플 질문

다음과 같은 질문들로 시스템을 테스트해보세요:

- "2025년 AI 트렌드는 무엇인가요?"
- "직장에서 AI를 어떻게 활용할 수 있나요?"
- "인공지능이 업무 생산성에 미치는 영향은?"
- "AI 기술의 미래 전망은 어떻게 되나요?"

## 🐛 문제 해결

### 일반적인 문제들

1. **Ollama 연결 실패**
   ```bash
   # Ollama 서버 상태 확인
   ollama list
   
   # 서버 재시작
   ollama serve
   ```

2. **모델 다운로드 실패**
   ```bash
   # 수동 모델 다운로드
   ollama pull gemma3:4b-it-qat
   ```

3. **메모리 부족 오류**
   - 더 작은 청크 크기 설정 (CHUNK_SIZE 감소)
   - 검색 문서 수 감소 (DEFAULT_K 감소)

4. **임베딩 모델 로딩 실패**
   ```bash
   # 수동 설치
   pip install sentence-transformers
   ```

### 로그 확인

Streamlit 앱 실행 시 터미널에서 상세 로그를 확인할 수 있습니다.

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 라이선스

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 감사의 글

- LangChain 팀의 훌륭한 프레임워크
- Ollama 팀의 로컬 LLM 솔루션
- Streamlit 팀의 직관적인 웹 앱 프레임워크
- 모든 오픈소스 기여자들

## 📞 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해주세요. 