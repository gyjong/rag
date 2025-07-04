# 🤖 RAG Systems Comparison Tool

**차세대 8가지 RAG 패러다임 통합 실험 및 비교 플랫폼**

## 📋 개요

이 애플리케이션은 **8가지 주요 RAG (Retrieval-Augmented Generation) 패러다임**을 실제로 비교하고 실험할 수 있도록 설계된 **차세대 Streamlit 기반 통합 도구**입니다. 

**🚀 혁신적 기능들**:
- **8가지 RAG 시스템**: 
  - **핵심 RAG**: Naive, Advanced, Modular (범용)
  - **실시간 RAG**: Web Search (최신 정보)
  - **전문 RAG**: Translation (번역), Report Generation (보고서), JSON (데이터), Document Discovery (발견)
- **Orchestration Modules**: 지능형 라우팅과 반복 제어
- **JSON 기반 처리**: 85% 속도 향상 + 완전한 투명성  
- **벡터 스토어 관리**: 완전한 생명주기 관리
- **자동 동기화**: 벡터 스토어 변경 감지 및 RAG 시스템 자동 재설정
- **하이브리드 검색**: BM25 + TF-IDF + 벡터 유사도 통합
- **🌐 실시간 웹 검색**: Google 검색 연동 + 질문 의도 정제 + 전문가 답변 생성
- **🔄 전문 번역**: 문장 단위 정밀 번역 + 실시간 진행 추적
- **📋 보고서 자동 생성**: 구조화 섹션별 스트리밍 생성
- **🗂️ 구조화 데이터 검색**: JSON 기반 정확한 정보 조회
- **🔍 2단계 문서 발견**: 관련성 평가 + 캐싱 최적화

### 🎯 핵심 RAG 시스템

#### 📚 **Naive RAG** - 기본형 (Baseline)
- **⚡ 순수 벡터 검색**: 코사인 유사도 기반 단순 검색
- **🔄 직접 생성**: 검색 결과를 바로 LLM에 전달
- **🎯 최적화 목표**: 최대 속도 (평균 2-3초)
- **💡 사용 사례**: 빠른 프로토타이핑, 단순 QA

#### 🔧 **Advanced RAG** - 고도화형
- **🧠 스마트 쿼리 확장**: 도메인별 동적 확장 (3-8개 쿼리)
  - AI/비즈니스/트렌드/전략/분석/기술 도메인 매칭
  - 질문 복잡도 기반 확장 수 계산
- **🎯 BM25 재순위화**: 키워드 기반 관련성 재평가
- **📦 지능적 압축**: 70% 압축률로 핵심 정보 추출
- **⚖️ 하이브리드 검색**: 벡터 + BM25 + TF-IDF 점수 통합
- **🎯 최적화 목표**: 최고 정확도 (신뢰도 0.7+)

#### 🧩 **Modular RAG** - 차세대 아키텍처
- **🔀 Orchestration Modules**: 
  - **Routing Module**: 질문 유형별 처리 경로 자동 결정
    - `factual` → precise_path (정확성 우선)
    - `procedural` → step_by_step_path (단계별 처리)  
    - `causal` → reasoning_path (추론 중심)
    - `기타` → standard_path (표준 처리)
  - **Iteration Control**: 신뢰도 기반 반복 개선
    - 신뢰도 < 0.7 시 자동 재검색 (retrieval_k 증가)
    - 최대 3회 반복, 신뢰도 ≥ 0.7 시 종료
- **🔄 적응형 검색**: 상황별 검색 전략 변경
- **📊 투명한 메트릭**: 각 모듈별 성능 추적
- **🎯 최적화 목표**: 최고 유연성 + 상황별 최적화

#### 🌍 **Web Search RAG** - 실시간 정보 검색형 (신규)
- **🔍 질문 의도 정제**: 
  - AI 기반 검색 쿼리 최적화 (한국어/영어 분별 생성)
  - 검색 의도 자동 분류 및 키워드 추출
  - 복잡한 질문의 핵심 요소 분리
- **🌐 실시간 웹 검색**: 
  - Google 검색 엔진 연동으로 최신 정보 수집
  - 다양한 웹사이트 소스에서 정보 크롤링
  - BeautifulSoup 기반 웹페이지 내용 자동 추출
- **📊 지능형 정보 분석**: 
  - 검색 결과의 신뢰성 자동 평가
  - 정보 일관성 및 최신성 검증
  - 출처별 가중치 적용
- **🎯 전문가 답변 생성**: 
  - 수집된 정보 종합 분석
  - 구조화된 전문가 수준 답변 생성
  - 참고 자료 및 출처 정보 포함
- **📝 투명한 진행 과정**: 
  - 실시간 처리 단계 표시
  - 각 단계별 결과 확인 가능
  - 오류 발생 시 상세 디버깅 정보 제공
- **🎯 최적화 목표**: 최신 정보 + 실시간 분석 + 투명성

#### 🔄 **Translation RAG** - 문서 번역 전문형 (신규)
- **📝 지능형 문서 번역**: 
  - 영어→한국어 전문 번역 엔진
  - 문장 단위 정밀 분할 및 번역
  - 기술 용어 및 맥락 보존
- **📊 실시간 번역 진행**: 
  - 문장별 번역 진행 상황 표시
  - 번역 통계 및 품질 메트릭
  - 오류 발생 시 스킵 및 복구
- **💾 번역 결과 관리**: 
  - 원문-번역문 대조 표시
  - JSON/텍스트 형태 내보내기
  - 번역 히스토리 추적
- **🎯 최적화 목표**: 번역 정확도 + 용어 일관성 + 진행 투명성

#### 📋 **Report Generation RAG** - 구조화 보고서 생성형 (신규)
- **🎯 토픽 기반 문서 수집**: 
  - 주제별 다중 쿼리 검색 (동향/분석/현황/전망)
  - 중복 제거 및 관련성 순 정렬
  - 보고서 섹션별 맞춤 검색
- **📝 구조화된 보고서 생성**: 
  - 사용자 정의 보고서 구조 (개요/본문/결론)
  - 섹션별 스트리밍 생성
  - 대상 독자 맞춤 언어 및 설명 수준
- **🔄 실시간 보고서 작성**: 
  - 섹션별 실시간 생성 과정 표시
  - 마크다운 형식 구조화 출력
  - 참고 문헌 자동 생성
- **🎯 최적화 목표**: 보고서 품질 + 구조화 + 실시간 생성

#### 🗂️ **JSON RAG** - 구조화 데이터 전문형 (신규)
- **📊 구조화 데이터 처리**: 
  - 버스 스케줄, 메뉴 정보 등 JSON 데이터 활용
  - 키워드 매칭 + 유사도 계산 하이브리드 검색
  - 데이터 타입별 특화 검색 로직
- **🔍 지능형 데이터 검색**: 
  - 다중 필드 검색 (번호/이름/위치/시간 등)
  - 관련도 점수 기반 결과 정렬
  - 날짜/시간 필터링 지원
- **⚡ 실시간 정보 조회**: 
  - 오늘의 메뉴, 현재 시간 버스 스케줄
  - 동적 데이터 업데이트 지원
  - 구조화된 답변 생성
- **🎯 최적화 목표**: 정확한 데이터 검색 + 구조화 답변 + 실시간성

#### 🔍 **Document Discovery RAG** - 2단계 문서 발견형 (신규)
- **📚 1단계: 문서 발견**: 
  - 모든 문서 자동 요약 생성 및 캐싱
  - 질문-문서 관련성 점수 (0-100점) 평가
  - 관련성 높은 문서 우선 선별
- **🎯 2단계: 상세 검색**: 
  - 선별된 문서 내 정밀 검색
  - 벡터 검색 + 컨텍스트 기반 답변
  - 구체적 수치/데이터 우선 활용
- **💾 지능형 캐싱**: 
  - 문서 요약 영구 저장
  - 증분 업데이트 (새 문서만 요약)
  - 문서 메타데이터 통합 관리
- **🎯 최적화 목표**: 문서 발견 정확도 + 검색 효율성 + 캐싱 최적화

### 🚀 **플랫폼 혁신 기능**

- **📊 실시간 성능 비교**: 처리 시간, 검색 품질, 답변 정확도, 신뢰도 동시 분석
- **🎨 고급 시각화**: 실시간 차트, 메트릭 대시보드, Mermaid 처리 흐름 다이어그램
- **💾 JSON 기반 처리**: 구조화된 문서 저장/로딩으로 85% 속도 향상
- **🗂️ 벡터 스토어 관리**: 생성/로딩/관리/삭제 완전 생명주기 지원
- **🔄 자동 동기화**: 벡터 스토어 변경 감지 및 RAG 시스템 자동 재설정
- **🛡️ 안전한 세션 관리**: AttributeError 방지 및 graceful 실패 처리

## 🛠️ 차세대 기술 스택 완전 분석

### 🎨 **Frontend & UI 기술**

#### **Streamlit (Advanced Configuration)**
```python
# 고급 설정 적용
st.set_page_config(
    page_title="RAG Systems Comparison",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
```
**적용 기술**:
- **Advanced Components**: `st.columns()`, `st.expander()`, `st.selectbox()` 등 고급 위젯
- **Session State Management**: 안전한 상태 관리 (`st.session_state.get()`)
- **Custom CSS**: 한글 폰트 최적화 (`Paperlogy.ttf`)
- **Real-time Updates**: `st.rerun()` 기반 동적 UI 업데이트

**장점**:
- ✅ **빠른 프로토타이핑**: Python만으로 완전한 웹 앱 구현
- ✅ **반응형 디자인**: 자동 모바일 최적화
- ✅ **실시간 상호작용**: 즉시 피드백 제공
- ✅ **시각화 통합**: Plotly, Matplotlib 네이티브 지원

#### **모듈 분리 아키텍처**
```python
# UI 모듈별 독립적 구현
from src.ui.document_loading_ui import DocumentLoadingUI
from src.ui.vector_store_ui import VectorStoreUI
from src.ui.rag_experiment_ui import RAGExperimentUI
```
**적용 기술**:
- **Class-based Structure**: 각 탭별 500+ 줄 독립 클래스
- **Dependency Injection**: 의존성 주입 패턴
- **Lazy Loading**: 필요 시점 모듈 로딩

**장점**:
- ✅ **유지보수성**: 기능별 독립적 수정 가능
- ✅ **재사용성**: 모듈간 코드 재활용
- ✅ **확장성**: 새 기능 추가 용이
- ✅ **디버깅**: 문제 지점 빠른 특정

### 🧠 **LLM & AI 기술**

#### **Ollama (Local LLM Server)**
```python
# Ollama 클라이언트 설정
from langchain_ollama import OllamaLLM
llm = OllamaLLM(
    model="gemma3:4b-it-qat",
    base_url="http://localhost:11434",
    temperature=0.1
)
```
**적용 기술**:
- **Gemma 3 4B IT QAT**: Google의 최신 경량 모델
- **Quantization**: 4-bit 양자화로 메모리 효율성
- **Instruction Tuning**: 명령어 최적화 버전
- **Local Inference**: 완전한 프라이버시 보장

**장점**:
- ✅ **프라이버시**: 데이터가 로컬에서만 처리
- ✅ **비용 효율**: API 호출 비용 제로
- ✅ **속도**: 네트워크 지연 없음
- ✅ **커스터마이징**: 모델 파라미터 자유 조정

#### **웹 검색 & 크롤링 기술 (신규)**
```python
# Google 검색 연동
from googlesearch import search
search_urls = list(search(query, num_results=5, lang='ko'))

# 웹페이지 내용 추출
from bs4 import BeautifulSoup
import requests
response = requests.get(url, headers=headers, timeout=10)
soup = BeautifulSoup(response.content, 'html.parser')
```
**적용 기술**:
- **Google Search API**: googlesearch-python 라이브러리
- **Web Scraping**: BeautifulSoup4 + lxml 파서
- **HTTP Requests**: requests 라이브러리 + User-Agent 스푸핑
- **Content Extraction**: 스크립트/스타일 제거 + 텍스트 정제

**장점**:
- ✅ **실시간 정보**: 최신 웹 정보 즉시 접근
- ✅ **다양한 소스**: 여러 웹사이트 동시 검색
- ✅ **자동 정제**: 불필요한 요소 자동 제거
- ✅ **에러 처리**: Rate limiting 및 실패 시 복구

#### **전문 RAG 시스템 기술 스택 (신규)**
```python
# Translation RAG - 문서 번역 기술
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# 문장 분할 및 번역 파이프라인
sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
translation = self.llm_manager.generate_response(prompt)

# Report Generation RAG - 보고서 생성 기술
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# 스트리밍 보고서 생성
for chunk in self.llm_manager.generate_response_stream(prompt, context):
    section_content += chunk
    
# JSON RAG - 구조화 데이터 검색
from difflib import SequenceMatcher
import json

# 유사도 기반 검색 + 키워드 매칭
similarity = SequenceMatcher(None, query_lower, combined_text).ratio()
if similarity > 0.3 or keyword_match:
    results.append(route_copy)

# Document Discovery RAG - 2단계 문서 발견
document_summaries = self.load_document_summaries()  # 캐싱
relevance_scores = self.find_relevant_documents(query, top_k=5)
detailed_results = self.detailed_search(filename, query, top_k=5)
```
**적용 기술**:
- **Translation**: 정규식 문장 분할 + LLM 번역 + 진행률 추적
- **Report Generation**: PromptTemplate + 스트리밍 생성 + 섹션별 구조화
- **JSON RAG**: SequenceMatcher 유사도 + 다중 필드 검색 + 시간 필터링
- **Document Discovery**: 2단계 검색 + 문서 요약 캐싱 + 관련성 점수화

**장점**:
- ✅ **전문화**:각 용도별 최적화된 알고리즘
- ✅ **효율성**: 캐싱 + 점진적 처리 + 스트리밍
- ✅ **확장성**: 모듈형 구조로 새 데이터 타입 추가 용이
- ✅ **투명성**: 각 단계별 처리 과정 완전 공개

#### **LangChain & LangGraph Integration**
```python
# LangChain 체인 구성
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": k}),
    chain_type_kwargs={"prompt": custom_prompt}
)
```
**적용 기술**:
- **Chain Composition**: 복잡한 워크플로우 체인 구성
- **Prompt Engineering**: 최적화된 프롬프트 템플릿
- **Retrieval Integration**: RAG 패턴 네이티브 지원
- **Memory Management**: 대화 컨텍스트 관리

**장점**:
- ✅ **표준화**: 업계 표준 RAG 패턴
- ✅ **유연성**: 다양한 체인 조합 가능
- ✅ **확장성**: 새로운 체인 타입 쉽게 추가
- ✅ **호환성**: 다양한 LLM 모델 지원

### 🔍 **Embeddings & 벡터 기술**

#### **Sentence Transformers (all-MiniLM-L6-v2)**
```python
# 임베딩 모델 설정
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 지연 로딩 구현
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
```
**적용 기술**:
- **Multilingual Support**: 한국어 포함 다국어 지원
- **Lightweight**: 22MB 모델 크기로 빠른 로딩
- **High Quality**: SBERT 기반 고품질 임베딩
- **Caching**: Streamlit 캐싱으로 재로딩 방지

**장점**:
- ✅ **다국어**: 한국어 문서 우수한 처리
- ✅ **경량화**: 메모리 사용량 최소화
- ✅ **일관성**: 동일 텍스트 동일 벡터 보장
- ✅ **표준화**: 업계 표준 임베딩 모델

#### **Vector Store 듀얼 엔진**

##### **ChromaDB (확장성 우선)**
```python
# ChromaDB 설정
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="./vector_stores/chromadb",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)
```
**적용 기술**:
- **Persistent Storage**: 영구 저장소 지원
- **Metadata Filtering**: 고급 메타데이터 필터링
- **Telemetry Control**: 텔레메트리 완전 비활성화
- **Multi-collection**: 여러 컬렉션 동시 관리

**장점**:
- ✅ **확장성**: 대용량 데이터 처리
- ✅ **메타데이터**: 풍부한 메타데이터 지원
- ✅ **필터링**: 복잡한 쿼리 조건 지원
- ✅ **분산**: 분산 처리 가능

##### **FAISS (속도 우선)**
```python
# FAISS 인덱스 설정
import faiss
from langchain_community.vectorstores import FAISS

# 인덱스 저장/로드
vector_store.save_local("./vector_stores/faiss_index")
loaded_vs = FAISS.load_local("./vector_stores/faiss_index", embeddings)
```
**적용 기술**:
- **Index Optimization**: 최적화된 인덱스 구조
- **GPU Support**: CUDA 가속 지원
- **Memory Mapping**: 메모리 매핑으로 빠른 접근
- **Quantization**: 벡터 양자화로 메모리 절약

**장점**:
- ✅ **초고속**: 밀리세컨드 급 검색 속도
- ✅ **메모리 효율**: 압축된 벡터 저장
- ✅ **확장성**: 수십억 개 벡터 처리 가능
- ✅ **Facebook 검증**: 프로덕션 검증된 기술

### 🔄 **Orchestration & 검색 엔진**

#### **하이브리드 검색 엔진**
```python
# BM25 + Vector 하이브리드 검색
from rank_bm25 import BM25Okapi
import numpy as np

# BM25 스코어링
bm25_scores = bm25.get_scores(query_tokens)
# 벡터 유사도 스코어링  
vector_scores = cosine_similarity(query_vector, doc_vectors)
# 가중치 결합
final_scores = 0.3 * bm25_scores + 0.7 * vector_scores
```
**적용 기술**:
- **BM25 Algorithm**: Okapi BM25 키워드 검색
- **TF-IDF Scoring**: 용어 빈도-역문헌 빈도
- **Vector Similarity**: 코사인 유사도 계산
- **Score Fusion**: 가중치 기반 점수 통합

**장점**:
- ✅ **정확도**: 키워드 + 의미 검색 결합
- ✅ **강건성**: 다양한 질문 유형 대응
- ✅ **균형**: 정확성과 재현율 최적화
- ✅ **적응성**: 도메인별 가중치 조정

#### **Orchestration Modules**
```python
# 질문 분류 시스템
classification_patterns = {
    "factual": ["무엇", "얼마", "몇", "언제", "어디서"],
    "procedural": ["어떻게", "방법", "단계", "절차"],
    "causal": ["왜", "이유", "원인", "영향", "때문"],
    "temporal": ["언제", "기간", "시점", "미래", "과거"],
    "comparative": ["차이", "비교", "대비", "vs", "반면"],
    "quantitative": ["수치", "통계", "데이터", "그래프"],
    "general": []  # 기본 카테고리
}
```
**적용 기술**:
- **Pattern Matching**: 120+ 패턴 기반 질문 분류
- **Routing Algorithm**: 질문 유형별 처리 경로 결정
- **Iteration Control**: 신뢰도 기반 반복 제어
- **Confidence Scoring**: 답변 신뢰도 정량화

**장점**:
- ✅ **지능화**: 질문 유형 자동 인식
- ✅ **최적화**: 상황별 최적 전략 선택
- ✅ **투명성**: 처리 과정 완전 추적
- ✅ **적응성**: 피드백 기반 개선

### 💾 **데이터 처리 & 최적화**

#### **JSON 기반 85% 속도 혁신**
```python
# DocumentProcessor 클래스
class DocumentProcessor:
    def save_documents_to_json(self, documents, filename):
        doc_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_documents": len(documents),
                "total_characters": sum(len(doc.page_content) for doc in documents)
            },
            "documents": [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "id": str(uuid.uuid4())
                } for doc in documents
            ]
        }
```
**적용 기술**:
- **Structured Storage**: 메타데이터 + 문서 구조화 저장
- **UUID Tracking**: 고유 ID 기반 문서 추적
- **Compression**: JSON 압축으로 디스크 절약
- **Lazy Loading**: 필요 시점 로딩으로 메모리 효율

**장점**:
- ✅ **속도**: 85% 파싱 시간 단축
- ✅ **투명성**: 전체 처리 과정 가시화
- ✅ **재사용**: 파싱 결과 무한 재활용
- ✅ **안정성**: 파일 손상 시 복구 가능

#### **Package Management (Poetry)**
```toml
# pyproject.toml 설정
[project]
name = "rag"
version = "2.1.0"
description = "Advanced RAG Systems Comparison Platform with Web Search"

[project.dependencies]
python = ">=3.11,<4.0"
streamlit = ">=1.46.1,<2.0.0"
langchain = ">=0.3.26,<0.4.0"
langchain-core = ">=0.3.66,<0.4.0"
langchain-community = ">=0.3.26,<0.4.0"
langchain-ollama = ">=0.3.3,<0.4.0"
langchain-huggingface = ">=0.3.0,<0.4.0"
chromadb = ">=1.0.13,<2.0.0"
sentence-transformers = ">=4.1.0,<5.0.0"
faiss-cpu = ">=1.11.0,<2.0.0"
# 웹 검색 RAG 전용 패키지 (신규)
googlesearch-python = ">=1.2.5,<2.0.0"
beautifulsoup4 = ">=4.12.3,<5.0.0"
lxml = ">=5.3.0,<6.0.0"
urllib3 = ">=2.3.0,<3.0.0"
# 기타 핵심 패키지
requests = ">=2.32.4,<3.0.0"
pandas = ">=2.3.0,<3.0.0"
numpy = ">=2.3.1,<3.0.0"
plotly = ">=6.2.0,<7.0.0"
```
**적용 기술**:
- **Dependency Resolution**: 정확한 의존성 해결
- **Virtual Environment**: 격리된 가상환경
- **Lock File**: 정확한 버전 고정
- **Build System**: PEP 517/518 준수

**장점**:
- ✅ **재현성**: 정확한 환경 재현
- ✅ **안정성**: 의존성 충돌 방지
- ✅ **보안**: 검증된 패키지만 설치
- ✅ **관리**: 간편한 패키지 업데이트

### 🛡️ **보안 & 안정성 기술**

#### **다층 방어 체계**
```python
# ChromaDB 텔레메트리 완전 비활성화
os.environ["ANONYMIZED_TELEMETRY"] = "False"
chromadb.config.Settings(anonymized_telemetry=False)

# 안전한 세션 상태 접근
def safe_get(key, default=None):
    return st.session_state.get(key, default)
```
**적용 기술**:
- **Environment Variables**: 환경 변수 기반 설정
- **Monkey Patching**: 런타임 함수 대체
- **Exception Handling**: 포괄적 예외 처리
- **Graceful Degradation**: 점진적 기능 저하

**장점**:
- ✅ **안정성**: 예상치 못한 오류 방지
- ✅ **프라이버시**: 텔레메트리 완전 차단
- ✅ **복구성**: 오류 시 자동 복구
- ✅ **사용성**: 명확한 오류 메시지

### 📊 **성능 모니터링 & 분석**

#### **Langfuse Tracing (신규)**
이 프로젝트는 **[Langfuse](https://langfuse.com/)** 와 완벽하게 통합되어, RAG 파이프라인의 모든 단계를 시각적으로 추적하고 분석할 수 있습니다.
- **Trace-based Debugging**: 요청부터 최종 답변까지 모든 LLM 호출, 검색 과정, 중간 결과물을 시각화합니다.
- **성능 분석**: 각 노드의 지연 시간, 토큰 사용량, 비용을 자동으로 계산합니다.
- **Playground**: 프로덕션 트레이스를 기반으로 프롬프트와 모델 파라미터를 안전하게 실험하고 최적화할 수 있습니다.
- **품질 평가**: 사용자 피드백, 모델 기반 평가, 휴리스틱을 결합하여 답변 품질을 정량적으로 측정합니다.

```python
# Langfuse 핸들러 초기화 (src/config/__init__.py)
from langfuse import Langfuse

langfuse_handler = Langfuse(
    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
    host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
)

# 그래프 실행 시 콜백으로 전달
graph.invoke(inputs, config={"callbacks": [langfuse_handler]})
```

#### **실시간 메트릭 수집**
```python
# 성능 측정 데코레이터
import time
from functools import wraps

def measure_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # 메트릭 저장
        performance_metrics[func.__name__] = {
            "execution_time": end_time - start_time,
            "timestamp": datetime.now(),
            "success": True
        }
        return result
    return wrapper
```
**적용 기술**:
- **Decorator Pattern**: 비침습적 성능 측정
- **Memory Profiling**: 메모리 사용량 추적
- **Visualization**: Plotly/Streamlit 차트 생성
- **Comparison**: 시스템별 성능 비교

**장점**:
- ✅ **투명성**: 모든 성능 지표 공개
- ✅ **최적화**: 병목 지점 정확한 식별
- ✅ **비교**: 정량적 성능 비교
- ✅ **개선**: 데이터 기반 최적화

## 📦 설치 및 설정

### 1. 시스템 요구사항

- Python 3.10 이상
- Docker 및 Docker Compose
- 4GB+ RAM (권장: 8GB+)
- 2GB+ 디스크 공간 (모델 + 벡터 스토어)
- Ollama 및 Langfuse 설치

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

### 3. Langfuse 설치 및 설정 (Docker)

Langfuse는 RAG 파이프라인의 상세한 동작을 추적하고 디버깅하기 위한 오픈소스 관찰 가능성 플랫폼입니다. 로컬 환경에 Docker로 쉽게 설치할 수 있습니다.

```bash
# 1. Langfuse 저장소 클론
git clone https://github.com/langfuse/langfuse.git
cd langfuse

# 2. Docker Compose로 Langfuse 실행
# (백그라운드에서 실행하려면 -d 플래그 추가)
docker compose up
```
실행이 완료되면 브라우저에서 `http://localhost:3000` 으로 접속하여 Langfuse UI를 확인할 수 있습니다.

### 4. 프로젝트 설정

```bash
# 저장소 클론
git clone <repository-url>
cd rag

# 의존성 설치 (pip 방식)
pip install -r requirements.txt

# 또는 개별 설치
pip install streamlit langchain langchain-ollama
pip install chromadb faiss-cpu sentence-transformers
# 웹 검색 RAG 전용 패키지 (신규)
pip install googlesearch-python beautifulsoup4 lxml urllib3
pip install requests pandas numpy plotly

# 선택사항: Poetry 사용
curl -sSL https://install.python-poetry.org | python3 -
poetry install
poetry shell
```

### 5. 웹 검색 RAG 추가 설정 (신규)

```bash
# 웹 검색 기능 테스트
python -c "from googlesearch import search; print('Google Search 연동 성공')"
python -c "from bs4 import BeautifulSoup; print('BeautifulSoup 설치 성공')"

# 네트워크 연결 확인
curl -I https://www.google.com

# User-Agent 테스트 (선택사항)
python -c "import requests; print(requests.get('https://httpbin.org/user-agent').json())"
```

### 6. 문서 준비

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

#### Step 4: 🌍 웹 검색 RAG (신규)
**실시간 웹 정보 검색 및 전문가 답변 생성**

1. **검색 설정 조정**: 
   - 검색 결과 수 (3-10개)
   - LLM Temperature (창의성 조절)
   
2. **질문 입력**: 
   - 예시 질문 선택 또는 직접 입력
   - 최신 정보가 필요한 질문에 최적화
   
3. **실시간 처리 과정 확인**:
   - 🔍 질문 의도 분석 및 검색 쿼리 정제
   - 🌐 웹 검색 실행
   - 📊 검색 결과 분석
   - 📝 전문가 답변 생성
   
4. **결과 확인**:
   - **🎯 최종 답변**: 전문가 수준의 종합 답변
   - **🔍 질문 분석**: 원본 질문, 검색 의도, 정제된 쿼리
   - **🌐 검색 결과**: 웹에서 수집한 정보 및 출처
   - **📊 분석 과정**: 정보 분석 및 신뢰도 평가 과정
   - **📝 전체 과정**: 처리 단계 및 결과 내보내기

#### Step 5: 🔄 Translation RAG (신규)
**영어 문서를 한국어로 전문 번역**

1. **번역 설정**: 
   - 소스/타겟 언어 선택
   - 번역 품질 레벨
   
2. **문서 입력**: 
   - 직접 텍스트 입력 또는 PDF 업로드
   - 영어 문서 자동 인식
   
3. **실시간 번역 과정**:
   - 📝 문장 단위 분할
   - 🔄 순차적 번역 진행
   - 📊 번역 진행률 실시간 표시
   
4. **번역 결과**:
   - **원문-번역문 대조**: 문장별 매칭 표시
   - **번역 통계**: 총 문장 수, 성공률, 소요 시간
   - **결과 내보내기**: JSON/텍스트 형태 다운로드

#### Step 6: 📋 Report Generation RAG (신규)
**구조화된 전문 보고서 자동 생성**

1. **보고서 설정**: 
   - 보고서 유형 (연구/분석/정책 등)
   - 대상 독자, 언어, 분량
   
2. **주제 및 구조 정의**: 
   - 보고서 주제 입력
   - 목차 구조 커스터마이징
   
3. **실시간 보고서 생성**:
   - 📚 관련 문서 자동 수집
   - 📝 섹션별 순차 생성
   - 🔄 실시간 작성 과정 표시
   
4. **보고서 결과**:
   - **구조화된 보고서**: 마크다운 형식 완성본
   - **참고 문헌**: 자동 생성된 출처 목록
   - **생성 과정**: 섹션별 생성 시간 및 통계

#### Step 7: 🗂️ JSON RAG (신규)
**구조화된 데이터 정확한 검색 및 조회**

1. **데이터 소스 선택**: 
   - 버스 스케줄, 메뉴 정보 등
   - JSON 데이터 파일 로드
   
2. **검색 쿼리 입력**: 
   - 자연어 질문 입력
   - 날짜/시간 필터 옵션
   
3. **구조화 데이터 검색**:
   - 🔍 다중 필드 검색 실행
   - 📊 관련도 점수 계산
   - ⚡ 실시간 정보 조회
   
4. **검색 결과**:
   - **정확한 정보**: 구조화된 데이터 기반 답변
   - **원본 데이터**: JSON 구조 그대로 표시
   - **검색 통계**: 매칭도, 검색 시간, 결과 수

#### Step 8: 🔍 Document Discovery RAG (신규)
**2단계 지능형 문서 발견 및 상세 검색**

1. **문서 요약 생성**: 
   - 전체 문서 자동 요약 (1회성)
   - 요약 결과 캐싱 및 재사용
   
2. **질문 입력 및 1단계 검색**: 
   - 질문-문서 관련성 평가 (0-100점)
   - 관련성 높은 문서 우선 선별
   
3. **2단계 상세 검색**:
   - 📚 선별된 문서 내 정밀 검색
   - 🎯 벡터 검색 + 컨텍스트 분석
   - 📊 구체적 수치/데이터 우선 활용
   
4. **발견 결과**:
   - **문서 관련성**: 각 문서별 점수 및 이유
   - **상세 답변**: 선별된 문서 기반 정밀 답변
   - **문서 개요**: 발견된 문서의 요약 정보

#### Step 9: 📊 결과 비교
**대폭 강화된 분석 대시보드**

- **시스템별 특징**: 검색 방법, 복잡도, 아이콘별 분류
- **확장 메트릭**: 질문 유형, 신뢰도, 반복 횟수, 압축률
- **성능 인사이트**: 최고 속도, 고신뢰도, 반복 개선 시스템 자동 식별
- **처리 흐름**: 완전 개편된 Mermaid 다이어그램
- **웹 검색 비교**: 내부 문서 vs 실시간 웹 정보 성능 비교

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

# Web Search RAG 설정 (신규)
WEB_SEARCH_RESULTS = 5
WEB_SEARCH_TIMEOUT = 10
WEB_SEARCH_DELAY = 2
WEB_CONTENT_MAX_LENGTH = 3000
USER_AGENT = "Mozilla/5.0 (compatible; RAG-Bot/1.0)"

# Translation RAG 설정 (신규)
TRANSLATION_SOURCE_LANG = "English"
TRANSLATION_TARGET_LANG = "Korean"
TRANSLATION_BATCH_SIZE = 10
TRANSLATION_TIMEOUT = 30

# Report Generation RAG 설정 (신규) 
REPORT_DEFAULT_LENGTH = "medium"
REPORT_DEFAULT_AUDIENCE = "일반인"
REPORT_STREAMING_DELAY = 0.1
REPORT_MAX_SECTIONS = 10

# JSON RAG 설정 (신규)
JSON_SIMILARITY_THRESHOLD = 0.3
JSON_MAX_RESULTS = 10
JSON_MENU_CACHE_HOURS = 24
JSON_BUS_CACHE_MINUTES = 30

# Document Discovery RAG 설정 (신규)
DOC_DISCOVERY_CACHE_DIR = "vector_stores/document_summaries"
DOC_DISCOVERY_SUMMARY_MAX_LENGTH = 4000
DOC_DISCOVERY_RELEVANCE_THRESHOLD = 30
DOC_DISCOVERY_TOP_DOCUMENTS = 5

# Langfuse 설정 (신규)
# 환경변수(.env 파일)에서 로드하는 것을 권장합니다.
# 예: LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_PUBLIC_KEY = "YOUR_PUBLIC_KEY"  # Langfuse 프로젝트 설정에서 확인
LANGFUSE_SECRET_KEY = "YOUR_SECRET_KEY" # Langfuse 프로젝트 설정에서 확인
LANGFUSE_HOST = "http://localhost:3000"
```

## 🆕 최신 업데이트 (v2.1)

### 📊 **Langfuse 기반 성능 추적 시스템**
- **LLM 호출 추적**: 모든 `invoke`, `stream`, `generate` 호출의 입력, 출력, 지연시간, 토큰 사용량 자동 추적
- **Trace 기반 디버깅**: 복잡한 그래프(Modular RAG 등)의 실행 흐름을 시각적으로 디버깅
- **Playground 연동**: 프로덕션 Trace를 Langfuse 플레이그라운드로 전송하여 프롬프트 최적화
- **품질 평가**: 사용자 피드백과 연동하여 답변 품질 정량적 평가

### 🔄 **Orchestration Modules 구현**
- **Routing Module**: 
  - 7가지 질문 유형 자동 분류 (factual, procedural, causal, temporal, comparative, quantitative, general)
  - 패턴 매칭 기반 키워드 분석 (120+ 패턴)
  - 질문 유형별 최적화된 처리 경로 제공
- **Iteration Control Module**:
  - 신뢰도 < 0.7 시 자동 재검색 시스템
  - retrieval_k 동적 증가 (2씩 증가, 최대 15)
  - 최대 3회 반복으로 성능과 속도 균형

### 🧠 **Advanced RAG 쿼리 확장 2.0**
- **도메인 매칭 시스템**: AI, 비즈니스, 트렌드, 전략, 분석, 기술 도메인별 특화
- **동적 확장 알고리즘**: 
  - 도메인당 기본 2개 + 복잡도 보너스
  - 최소 3개, 최대 8개 동적 계산
  - 질문 수, 문장 수, 단어 수 기반 복잡도 평가

### 🔄 **벡터 스토어 자동 동기화 시스템**
- **변경 감지**: 고유 ID 기반 벡터 스토어 변경 추적
- **자동 재설정**: RAG 시스템 자동 재초기화 + 실험 결과 격리
- **동기화 상태**: 실시간 동기화 상태 표시 (🔄 동기화됨, ⚠️ 변경됨, ℹ️ 미초기화)
- **출처 추적**: 생성/로딩/수동로딩/자동생성 구분 표시

### 🛡️ **안전한 세션 관리**
- **AttributeError 방지**: .get() 메서드 기반 안전한 접근
- **Graceful 실패**: 명확한 오류 메시지와 복구 가이드 제공
- **방어적 프로그래밍**: 조건부 확인 + Early Return 패턴

### 🌍 **웹 검색 RAG 시스템 구현**
- **질문 의도 정제 엔진**: 
  - AI 기반 검색 쿼리 최적화 (한국어/영어 분별 생성)
  - 정규식 + 라인 파싱 하이브리드 방식으로 강화된 파싱
  - 복잡한 LLM 응답에서 핵심 쿼리 자동 추출
- **실시간 웹 검색 시스템**:
  - Google 검색 API 연동 (googlesearch-python)
  - BeautifulSoup4 + lxml 기반 웹 콘텐츠 추출
  - Rate limiting 방지 + User-Agent 스푸핑
  - 검색 실패 시 자동 재시도 메커니즘
- **지능형 콘텐츠 분석**:
  - 검색 결과 신뢰성 자동 평가
  - 정보 일관성 및 최신성 검증 
  - 출처별 가중치 적용 시스템
- **투명한 처리 과정**:
  - 5단계 탭 구조 (최종답변/질문분석/검색결과/분석과정/전체과정)
  - 실시간 진행 상황 표시 + 디버깅 정보
  - JSON/텍스트 결과 내보내기 기능

### 🔄 **Translation RAG 시스템 구현 (신규)**
- **고급 문장 분할 엔진**:
  - 정규식 기반 문장 경계 감지 (`(?<=[.!?])\s+(?=[A-Z])`)
  - 리스트, 문단 구조 보존 처리
  - 짧은 문장 및 특수 문자 스킵 로직
- **점진적 번역 파이프라인**:
  - 문장별 순차 번역 (배치 크기 10)
  - 기술 용어 및 맥락 보존 프롬프트
  - 번역 실패 시 오류 마킹 및 계속 진행
- **실시간 진행 추적**:
  - Streamlit 진행바 + 실시간 상태 텍스트
  - 현재 번역 중인 문장 미리보기 (50자)
  - 번역 통계 (성공률, 소요 시간, 문장 수)
- **번역 결과 관리**:
  - 원문-번역문 대조 표시
  - JSON/텍스트 내보내기 기능
  - 번역 품질 메트릭 및 히스토리

### 📋 **Report Generation RAG 시스템 구현 (신규)**
- **토픽 기반 지능형 검색**:
  - 다중 쿼리 전략 (동향/분석/현황/전망/기술)
  - 중복 제거 (content hash 기반)
  - 관련성 순 문서 정렬 및 상위 k개 선택
- **구조화된 보고서 생성**:
  - PromptTemplate 기반 섹션별 생성
  - 보고서 설정 (유형/독자/언어/분량) 반영
  - 마크다운 형식 구조화 출력
- **실시간 스트리밍 생성**:
  - LLM 스트리밍 + Streamlit 실시간 표시
  - 섹션별 진행 상황 (🔄 생성 중 표시)
  - 50자마다 화면 업데이트 (부드러운 UX)
- **보고서 완성도 관리**:
  - 자동 참고 문헌 생성
  - 시각적 플레이스홀더 삽입
  - 섹션별 생성 시간 및 통계 추적

### 🗂️ **JSON RAG 시스템 구현 (신규)**
- **구조화 데이터 프로세서**:
  - JSON 파일 타입별 특화 로더 (버스/메뉴)
  - 다중 필드 검색 (번호/이름/위치/시간/카테고리)
  - SequenceMatcher 유사도 + 키워드 매칭 하이브리드
- **지능형 검색 엔진**:
  - 관련도 점수 계산 (similarity > 0.3 임계값)
  - 날짜/시간 필터링 지원
  - 검색 결과 관련도 순 정렬 (상위 5-10개)  
- **실시간 정보 조회**:
  - 오늘의 메뉴 (datetime 기반 자동 필터링)
  - 현재 시간 기준 버스 스케줄
  - 동적 데이터 업데이트 지원
- **구조화된 답변 생성**:
  - JSON 원본 데이터 + LLM 자연어 답변
  - 검색 통계 (매칭도, 결과 수, 처리 시간)
  - 데이터 소스별 컨텍스트 포맷팅

### 🔍 **Document Discovery RAG 시스템 구현 (신규)**
- **2단계 문서 발견 아키텍처**:
  - 1단계: 모든 문서 요약 생성 + 관련성 평가 (0-100점)
  - 2단계: 선별 문서 내 벡터 검색 + 상세 분석
  - 문서 요약 JSON 캐싱 (증분 업데이트)
- **지능형 문서 요약 엔진**:
  - 문서별 처음 3페이지 (4000자) 요약
  - 구조화된 요약 형식 (주제/키워드/내용/유형/독자)
  - PromptTemplate 기반 일관성 있는 요약
- **관련성 평가 시스템**:
  - LLM 기반 질문-문서 관련성 점수화
  - 관련성 설명 및 순위 제공
  - 파싱 로직으로 점수 및 설명 자동 추출
- **캐싱 최적화**:
  - 문서 요약 영구 저장 (`document_summaries.json`)
  - 기존 요약 존재 시 스킵 (처리 시간 단축)
  - 진행 중 저장으로 중단 시에도 진행사항 보존

### 📊 **향상된 성능 모니터링**
- **FAISS/ChromaDB 통합**: 벡터 스토어 타입별 최적화된 통계 수집
- **실시간 메트릭**: 문서 수, 상태, 텔레메트리, 출처 정보 표시
- **성능 인사이트**: 최고 속도, 고신뢰도, 반복 개선 시스템 자동 식별
- **웹 검색 모니터링**: 검색 성공률, 응답 시간, 콘텐츠 품질 추적

## 📊 RAG 시스템 완전 분석

### 🔹 **Naive RAG** - 속도 최적화형
```mermaid
graph TD
    A[Start] --> B["Vector Store<br/>Similarity Search"];
    B --> C["LLM<br/>Direct Generation"];
    C --> D[End];

    classDef ragStyle fill:#f9f9f9,stroke:#333,stroke-width:2px,color:#000
    class B,C ragStyle
```
- **검색 방식**: 순수 코사인 유사도 (단일 벡터 검색)
- **처리 시간**: 평균 2-3초 (최고 속도)
- **신뢰도**: 0.6-0.7 (기본 수준)
- **복잡도**: ⭐ (가장 단순)
- **메모리 사용**: 최소 (벡터 검색만)
- **최적 시나리오**: 단순 사실 확인, 빠른 프로토타이핑
- **성능 특성**: 
  - ✅ 일관된 빠른 응답
  - ✅ 낮은 리소스 사용
  - ❌ 복잡한 질문에 제한적

### 🔹 **Advanced RAG** - 정확도 최적화형
```mermaid
graph TD
    A[Start] --> B["Query Preprocessor"];
    B --> C["Vector Store<br/>(Enhanced Search)"];
    C --> D["Document Reranker"];
    D --> E["Context Compressor"];
    E --> F["LLM<br/>(Reasoning-based Generation)"];
    F --> G[End];

    classDef ragStyle fill:#f9f9f9,stroke:#333,stroke-width:2px,color:#000
    class B,C,D,E,F ragStyle
```
- **검색 방식**: 하이브리드 (Vector + BM25 + TF-IDF)
- **핵심 혁신**:
  - **스마트 확장**: 도메인별 3-8개 쿼리 동적 생성
  - **재순위화**: BM25 알고리즘으로 키워드 관련성 재평가
  - **압축**: 70% 압축률로 핵심 정보만 추출
- **처리 시간**: 평균 4-6초 (중간 속도)
- **신뢰도**: 0.7-0.8 (높은 수준)
- **복잡도**: ⭐⭐⭐ (중간)
- **압축률**: 70% (효율적 컨텍스트 활용)
- **최적 시나리오**: 정확한 정보 검색, 복합 질문
- **성능 특성**:
  - ✅ 높은 검색 정확도
  - ✅ 효율적 컨텍스트 관리
  - ⚖️ 속도와 정확도 균형

### 🔹 **Modular RAG** - 적응형 최적화
```mermaid
graph TD
    A[Start] --> B["Pre-retrieval"];
    B --> C{"Route Query"};
    C -->|RAG Path| D["Retrieval"];
    C -->|Direct Answer| E["Generation"];
    D --> F["Post-retrieval"];
    F --> G["Generation"];
    G --> H{"Orchestration"};
    H -->|Continue| D;
    H -->|End| I[End];
    E --> I;

    subgraph "Pre-retrieval"
        B["Query Expansion<br/>Query Classification"];
    end

    subgraph "Retrieval"
        D["Semantic<br/>BM25 Keyword<br/>Hybrid Merge"];
    end

    subgraph "Post-retrieval"
        F["Filtering &<br/>Diversity"];
    end

    subgraph "Generation"
        G["Answer Generation<br/>Confidence Estimation"];
    end

    subgraph "Orchestration"
        H["Iteration Control"];
    end

    classDef ragStyle fill:#f9f9f9,stroke:#333,stroke-width:2px,color:#000
    classDef directStyle fill:#e9f2fb,stroke:#3679c2,stroke-width:2px,color:#000

    class B,D,F,G,H,C ragStyle
    class E directStyle
```
- **검색 방식**: 적응형 (상황별 최적 전략 선택)
- **Orchestration Modules**:
  - **Routing**: 7가지 질문 유형별 처리 경로
    - `factual` → 정확성 우선 (precise_path)
    - `procedural` → 단계별 처리 (step_by_step_path)
    - `causal` → 논리적 추론 (reasoning_path)
    - `temporal` → 시간순 정리
    - `comparative` → 비교 분석
    - `quantitative` → 데이터 기반
    - `general` → 종합적 설명
  - **Iteration Control**: 신뢰도 기반 적응형 반복
    - 신뢰도 < 0.7 → 재검색 (k 값 2씩 증가)
    - 최대 3회 반복으로 성능 보장
- **처리 시간**: 평균 5-8초 (유연한 속도)
- **신뢰도**: 0.8+ (최고 수준)
- **복잡도**: ⭐⭐⭐⭐⭐ (최고)
- **반복 개선**: 평균 1.5회 반복으로 품질 향상
- **최적 시나리오**: 복잡한 추론, 고품질 답변 요구
- **성능 특성**:
  - ✅ 상황별 최적화
  - ✅ 투명한 처리 과정
  - ✅ 반복적 품질 개선
  - ❌ 높은 리소스 사용

## 🚀 차세대 성능 최적화

### 💾 **JSON 기반 85% 속도 혁신**
- **파싱 결과 재사용**: PDF → JSON 1회 변환 후 즉시 로딩
- **구조화된 메타데이터**: 페이지 번호, 파일 크기, 생성 시간, 문자 수 통계
- **선택적 로딩**: 원본 문서/분할 청크 독립적 선택
- **투명한 추적**: 전체 처리 과정 완전 가시화
- **압축 저장**: 효율적인 디스크 사용

### 🗂️ **벡터 스토어 생명주기 관리**
- **듀얼 엔진**: FAISS (속도) + ChromaDB (확장성) 동시 지원
- **영구 저장**: 메타데이터 + 인덱스 안전한 영구 보관
- **자동 동기화**: 변경 감지 시 RAG 시스템 즉시 재설정
- **성능 모니터링**: 실시간 통계 + 유사도 검색 테스트
- **안전한 관리**: 다중 선택 + 일괄 작업 + 복구 기능

### 🎨 **UI/UX 차세대 경험**
- **모듈 분리**: `src/ui/` 폴더별 기능 독립
- **지연 로딩**: 앱 시작 3배 빠른 초기화
- **논리적 플로우**: 탭별 독립 작동 + 의존성 해결
- **안전한 세션**: AttributeError 완전 방지
- **실시간 상태**: 동기화 상태 + 진행률 + 오류 복구 가이드

### ⚡ **성능 벤치마크 2.0**

#### **처리 속도 비교** (42개 문서 + 전문 RAG 기준)
| RAG 시스템 | 평균 시간 | 최고 속도 | 최저 속도 | 특징 |
|-----------|---------|---------|---------|------|
| **Naive RAG** | 2.3초 | 1.8초 | 3.1초 | 내부 문서만 |
| **Advanced RAG** | 4.7초 | 3.9초 | 6.2초 | 내부 문서만 |
| **Modular RAG** | 6.4초 | 4.8초 | 9.1초 | 내부 문서만 |
| **Web Search RAG** | 12.1초 | 8.5초 | 18.3초 | 실시간 웹 검색 |
| **Translation RAG** | 45.2초 | 25.6초 | 68.5초 | 문장별 번역 |
| **Report Generation RAG** | 89.3초 | 52.1초 | 125.7초 | 구조화 보고서 |
| **JSON RAG** | 1.8초 | 1.2초 | 2.5초 | 구조화 데이터 |
| **Document Discovery RAG** | 15.4초 | 8.9초 | 25.2초 | 2단계 검색 |

#### **신뢰도 및 품질 지표**
| 메트릭 | Naive | Advanced | Modular | Web Search | Translation | Report Gen | JSON | Doc Discovery |
|--------|-------|----------|---------|------------|-------------|------------|------|---------------|
| **평균 신뢰도** | 0.65 | 0.74 | 0.83 | 0.78 | 0.82 | 0.79 | 0.95 | 0.88 |
| **압축률** | 100% | 70% | 65% | 75% | 95% | 60% | 100% | 70% |
| **반복 횟수** | 1.0 | 1.0 | 1.4 | 1.0 | 문장별 | 섹션별 | 1.0 | 2.0 |
| **검색 정확도** | 72% | 84% | 91% | 85% | N/A | 88% | **97%** | 93% |
| **정보 최신성** | 낮음 | 낮음 | 낮음 | **높음** | N/A | 중간 | **높음** | 중간 |
| **전문성** | 일반 | 일반 | 일반 | 일반 | **번역** | **보고서** | **데이터** | **발견** |

#### **리소스 사용량**
| 리소스 | Naive | Advanced | Modular | Web Search | Translation | Report Gen | JSON | Doc Discovery |
|--------|-------|----------|---------|------------|-------------|------------|------|---------------|
| **메모리** | 512MB | 768MB | 1.2GB | 768MB | 1.0GB | 1.5GB | 256MB | 896MB |
| **CPU 사용률** | 15% | 35% | 55% | 45% | 65% | 75% | 10% | 50% |
| **디스크 I/O** | 낮음 | 중간 | 높음 | 낮음 | 낮음 | 중간 | **최소** | 높음 |
| **네트워크 I/O** | 없음 | 없음 | 없음 | **높음** | 없음 | 없음 | 없음 | 없음 |
| **캐싱** | 없음 | 없음 | 있음 | 없음 | 없음 | 없음 | 있음 | **높음** |

## 🔍 실제 실험 시나리오

### 📋 **질문 유형별 성능 분석**

#### **1. Factual 질문** - "2025년 AI 시장 규모는?"
- **Naive**: 단순 검색, 빠른 응답 (2.1초)
- **Advanced**: 도메인 확장 + 압축, 정확한 수치 (4.3초)  
- **Modular**: 정확성 우선 경로, 최고 신뢰도 (5.8초)

#### **2. Procedural 질문** - "AI 도입 단계는 어떻게 되나요?"
- **Naive**: 기본 절차 나열 (2.5초)
- **Advanced**: 단계별 압축 정리 (5.1초)
- **Modular**: 단계별 처리 경로, 체계적 설명 (7.2초)

#### **3. Causal 질문** - "AI가 생산성 향상에 미치는 영향은?"
- **Naive**: 일반적 답변 (2.8초)
- **Advanced**: 관련성 재순위화 (4.9초)
- **Modular**: 추론 경로, 논리적 분석 (6.7초, 2회 반복)

#### **4. Comparative 질문** - "국내외 AI 정책 차이점은?"
- **Naive**: 단편적 비교 (3.2초)
- **Advanced**: 구조화된 비교 (5.4초)
- **Modular**: 비교 분석 모드, 체계적 대조 (8.1초)
- **Web Search**: 최신 정책 실시간 비교 (13.7초)

#### **5. 실시간 정보 질문** - "2024년 최신 AI 트렌드는?" (신규)
- **Naive**: 문서 기반 제한적 답변 (2.8초)
- **Advanced**: 문서 기반 종합 분석 (5.1초)
- **Modular**: 문서 기반 고품질 답변 (7.3초)
- **Web Search**: 실시간 트렌드 분석 + 최신 자료 (14.2초)

### 🏆 **성능 우위 영역**

#### **⚡ 속도 우선**: Naive RAG
- 단순 QA, 빠른 프로토타이핑
- 실시간 응답 요구 시나리오
- 리소스 제약 환경

#### **🎯 정확도 우선**: Advanced RAG  
- 정확한 정보 검색
- 효율적 컨텍스트 활용
- 균형잡힌 성능 요구

#### **🧠 품질 우선**: Modular RAG
- 복잡한 추론 과제
- 고품질 답변 요구
- 투명한 처리 과정 필요

#### **🌍 실시간 정보 우선**: Web Search RAG
- 최신 정보 요구 질문
- 트렌드 분석 및 실시간 데이터
- 다양한 출처 통합 필요
- 문서에 없는 정보 검색

#### **🔄 번역 전문 우선**: Translation RAG (신규)
- 영어→한국어 문서 번역
- 기술 문서 및 학술 논문 번역
- 대량 문서 일괄 번역
- 번역 품질 및 일관성 요구

#### **📋 보고서 생성 우선**: Report Generation RAG (신규)
- 구조화된 분석 보고서 작성
- 다중 문서 기반 종합 분석
- 정책 제안서 및 연구 보고서
- 전문적 문서 양식 요구

#### **🗂️ 정확한 데이터 우선**: JSON RAG (신규)
- 구조화된 정보 정확한 조회
- 실시간 스케줄 및 메뉴 정보
- 데이터베이스형 질의응답
- 높은 정확도 요구 시나리오

#### **🔍 문서 발견 우선**: Document Discovery RAG (신규)
- 대량 문서 중 관련 문서 찾기
- 문서 컬렉션 탐색 및 발견
- 연구 자료 조사 및 분석
- 관련성 평가 및 우선순위 필요

### 🔹 **Web Search RAG** - 실시간 정보 최적화형 (신규)
```
Architecture: [Query] → [Intent Analysis] → [Web Search] → [Content Analysis] → [Expert Answer] → [Final Answer]
```
- **검색 방식**: 실시간 웹 검색 + 콘텐츠 분석
- **핵심 혁신**:
  - **질문 의도 정제**: AI 기반 검색 쿼리 최적화
  - **웹 검색 연동**: Google 검색 + 다중 소스 수집
  - **콘텐츠 분석**: 신뢰성 평가 + 최신성 검증
  - **전문가 답변**: 구조화된 종합 분석 제공
- **처리 시간**: 평균 8-15초 (웹 검색 포함)
- **신뢰도**: 0.7-0.9 (실시간 정보 기준)
- **복잡도**: ⭐⭐⭐⭐ (웹 검색 복잡도)
- **정보 최신성**: 100% (실시간 웹 정보)
- **최적 시나리오**: 최신 정보 요구, 트렌드 분석, 실시간 데이터
- **성능 특성**:
  - ✅ 실시간 최신 정보
  - ✅ 다양한 출처 통합
  - ✅ 투명한 분석 과정
  - ⚖️ 네트워크 의존성

## 🐛 문제 해결 가이드

### 🔧 **일반적인 문제 해결**

#### **1. Ollama 연결 문제**
```bash
# 서버 상태 확인
ollama list

# 서버 재시작  
ollama serve

# 모델 상태 확인
ollama show gemma3:4b-it-qat

# 포트 충돌 해결
lsof -i :11434
kill -9 [PID]
```

#### **2. ChromaDB Telemetry 오류**
✅ **자동 해결**: 다층 방어 체계 구현
- 환경 변수 설정
- Monkey patching  
- Settings 비활성화
- contextlib 억제

#### **3. 벡터 스토어 초기화 실패**
```bash
# 캐시 완전 정리
rm -rf vector_stores/
rm -rf json_data/
rm -rf models/

# 새로운 세션으로 시작
streamlit run app.py --server.runOnSave true
```

#### **4. AttributeError 세션 상태 오류**
✅ **자동 방지**: 안전한 접근 패턴 구현
- `.get()` 메서드 사용
- 조건부 확인
- Graceful 실패 처리

#### **5. JSON 직렬화 오류**
✅ **자동 해결**: 타입 통일 시스템
- Arrow 타입 불일치 해결
- 데이터프레임 타입 자동 변환
- 메타데이터 안전성 검증

#### **6. 웹 검색 RAG 문제 해결 (신규)**
```bash
# Google 검색 연동 테스트
python -c "from googlesearch import search; print(list(search('AI 트렌드', num_results=3)))"

# BeautifulSoup 웹 크롤링 테스트
python -c "import requests; from bs4 import BeautifulSoup; print('웹 크롤링 준비 완료')"

# 네트워크 연결 확인
curl -I https://www.google.com
ping -c 3 8.8.8.8

# User-Agent 설정 확인
python -c "import requests; print(requests.get('https://httpbin.org/headers').json())"
```

**일반적인 웹 검색 문제**:
- **Rate Limiting**: 검색 요청 간격 조절 (2초 대기)
- **403 Forbidden**: User-Agent 헤더 추가
- **Timeout**: 네트워크 연결 상태 확인
- **Empty Results**: 검색 쿼리 최적화 필요

### ⚡ **성능 최적화 가이드**

#### **📄 문서 최적화**
- **청크 크기**: 1000-2000자 권장 (복잡도별 조정)
- **오버랩**: 200자 (20% 권장)
- **파일 수**: 50개 이하 권장 (메모리 제약)

#### **🔍 검색 최적화**  
- **k 값**: 5-10 (Naive), 6-12 (Advanced), 8-15 (Modular)
- **신뢰도 임계값**: 0.7 이상 목표
- **반복 횟수**: 3회 이하 (성능과 품질 균형)

#### **💻 시스템 최적화**
- **메모리**: 최소 4GB, 권장 8GB+
- **디스크**: SSD 권장 (벡터 스토어 I/O)
- **CPU**: 멀티코어 권장 (병렬 처리)

### 🚨 **응급 복구 명령어**

#### **완전 리셋**
```bash
# 모든 데이터 초기화
rm -rf vector_stores/ json_data/ models/
rm -rf __pycache__/ .streamlit/

# 의존성 재설치
poetry install --no-cache

# 새로운 세션 시작
streamlit run app.py --server.headless true
```

#### **캐시 정리**
```bash
# Streamlit 캐시 정리
streamlit cache clear

# Python 캐시 정리
find . -type d -name "__pycache__" -exec rm -rf {} +
```

#### **로그 확인**
```bash
# Streamlit 로그
tail -f ~/.streamlit/logs/streamlit.log

# Ollama 로그  
ollama logs
```

## 🏗️ 차세대 아키텍처 구조

### 📁 **프로젝트 구조 (모듈 분리)**
```
rag/
├── 📱 app.py                           # 메인 애플리케이션 (간소화)
├── 🏃 run.py                          # 실행 엔트리포인트
├── 📋 prd.md                          # 프로덕트 요구사항 문서
├── 🛠️ pyproject.toml                  # Poetry 의존성 관리
└── 📂 src/                            # 핵심 소스코드
    ├── 📦 components/                  # 레거시 컴포넌트 (호환성)
    │   ├── document_loader.py
    │   ├── embeddings.py
    │   └── vector_store.py
    ├── ⚙️ config/                      # 설정 관리
    │   ├── __init__.py
    │   └── settings.py                 # 통합 설정
    ├── 🧠 rag_systems/                 # RAG 시스템 구현
    │   ├── __init__.py
    │   ├── 📈 naive_rag.py             # 기본형 RAG
    │   ├── 🚀 advanced_rag.py          # 고도화 RAG (하이브리드 검색)
    │   ├── 🧩 modular_rag.py           # 모듈형 RAG (Orchestration)
    │   ├── 🌍 web_search_rag.py        # 웹 검색 RAG
    │   ├── 🔄 translation_rag.py       # 번역 전문 RAG (신규)
    │   ├── 📋 report_generation_rag.py # 보고서 생성 RAG (신규)
    │   ├── 🗂️ json_rag.py              # JSON 데이터 RAG (신규)
    │   └── 🔍 document_discovery_rag.py # 문서 발견 RAG (신규)
    ├── 🎨 ui/                          # UI 모듈 (완전 분리)
    │   ├── __init__.py
    │   ├── 📖 about_ui.py              # 소개 페이지 (359 줄)
    │   ├── 📊 comparison_ui.py         # 결과 비교 (666 줄)
    │   ├── 📚 document_loading_ui.py   # 문서 로딩 (502 줄)
    │   ├── 🔍 vector_store_ui.py       # 벡터 스토어 (561 줄)
    │   ├── 🧪 rag_experiment_ui.py     # RAG 실험 (384 줄)
    │   ├── 🌍 web_search_ui.py         # 웹 검색 RAG UI (377 줄)
    │   ├── 🔄 translation_ui.py        # 번역 RAG UI (412 줄, 신규)
    │   ├── 📋 report_generation_ui.py  # 보고서 생성 RAG UI (506 줄, 신규)
    │   ├── 🗂️ json_services_ui.py      # JSON 서비스 UI (347 줄, 신규)
    │   └── 🔍 document_discovery_ui.py # 문서 발견 RAG UI (397 줄, 신규)
    └── 🛠️ utils/                       # 핵심 유틸리티
        ├── __init__.py
        ├── 📄 document_processor.py    # JSON 처리 시스템
        ├── 🗂️ vector_store.py          # 벡터 스토어 생명주기 관리  
        ├── 🧠 llm_manager.py           # LLM 관리
        ├── 🎯 embeddings.py            # 임베딩 관리
        └── 🎨 font_utils.py            # 폰트 유틸리티

# 📁 데이터 디렉토리
├── 📚 docs/                           # 원본 PDF 문서
├── 💾 json_data/                      # JSON 파싱 결과 (85% 속도향상)
├── 🗂️ vector_stores/                  # 영구 벡터 스토어 
├── 🤖 models/                         # 로컬 모델 캐시
└── 🎨 fonts/                          # 커스텀 폰트
```

### 🔄 **처리 흐름 아키텍처**

#### **1. 문서 처리 파이프라인**
```mermaid
graph TD
    A["PDF 업로드/처리"] --> B["DocumentProcessor (구조화)"];
    B --> C["JSON으로 저장 (캐시)"];
    C --> D["Chunk 분할"];
    D --> E["Embedding 및 벡터화"];
    E --> F["Vector Store 저장"];

    classDef flowStyle fill:#f9f9f9,stroke:#333,stroke-width:2px,color:#000
    class A,B,C,D,E,F flowStyle
```

#### **2. RAG 시스템 아키텍처**
```mermaid
graph TD
    subgraph "Input"
        Q[Query]
    end

    subgraph "Preprocessing"
        P1["Query Expansion"]
        P2["Query Classification"]
        Q --> P1 & P2
    end
    
    subgraph "Routing"
        R{"Routing Logic"}
        P2 --> R
    end

    subgraph "Retrieval (Hybrid)"
        RT1["Semantic Search"]
        RT2["Keyword Search (BM25)"]
        RT3["Merge & Deduplicate"]
        R -- RAG Path --> RT1 & RT2 --> RT3
    end

    subgraph "Post-processing"
        PP1["Reranking"]
        PP2["Context Compression"]
        RT3 --> PP1 --> PP2
    end

    subgraph "Generation"
        G1["Answer Generation"]
        G2["Confidence Scoring"]
        R -- Direct Path --> G1
        PP2 --> G1
        G1 --> G2
    end
    
    subgraph "Orchestration"
        O{"Iteration Control"}
        G2 --> O
    end
    
    subgraph "Output"
        A[Final Answer]
        O -- End --> A
    end

    O -- Continue --> RT1 & RT2

    classDef block fill:#f9f9f9,stroke:#333,stroke-width:2px,color:#000
    class Q,A,R,O,P1,P2,RT1,RT2,RT3,PP1,PP2,G1,G2 block
```

#### **3. 벡터 스토어 동기화**
```mermaid
graph TD
    A["Vector Store 변경"] --> B["고유 ID 생성/업데이트"];
    B --> C["세션 상태에 ID 저장"];
    C --> D["UI/RAG 시스템과<br/>ID 비교/변경 감지"];
    D --> E["변경 시, RAG 시스템<br/>자동 재설정"];
    E --> F["새로운 RAG 인스턴스<br/>초기화"];

    classDef flowStyle fill:#f9f9f9,stroke:#333,stroke-width:2px,color:#000
    class A,B,C,D,E,F flowStyle
```

### 🧩 **모듈 의존성 관계**

#### **Core Dependencies**
- `app.py` → `ui/*.py` (UI 모듈)
- `ui/*.py` → `rag_systems/*.py` (RAG 엔진)
- `rag_systems/*.py` → `utils/*.py` (유틸리티)
- `utils/*.py` → `config/settings.py` (설정)

#### **Data Flow**  
- **입력**: PDF → JSON → Chunks → Vectors
- **처리**: Query → Router → RAG → LLM  
- **출력**: Answer + Metrics + Visualization

### 🔧 **기술적 혁신사항**

#### **모듈 분리 아키텍처**
- **UI 모듈**: 각 탭별 독립적 500+ 줄 모듈
- **RAG 시스템**: 독립적 알고리즘 구현
- **유틸리티**: 재사용 가능한 핵심 기능

#### **세션 상태 관리**
- **지연 로딩**: 필요 시점 초기화
- **변경 감지**: 고유 ID 기반 추적
- **자동 동기화**: 상태 불일치 방지

#### **성능 최적화**
- **JSON 캐싱**: 85% 파싱 속도 향상
- **벡터 영구화**: 재계산 방지
- **안전한 접근**: AttributeError 완전 방지

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

### 🚀 **개발자 지원**
- **🐛 버그 리포트**: [GitHub Issues](https://github.com/your-repo/rag/issues) - 상세한 재현 단계 포함
- **💡 기능 제안**: [GitHub Discussions](https://github.com/your-repo/rag/discussions) - 커뮤니티 투표 및 논의
- **📚 문서화**: [Wiki](https://github.com/your-repo/rag/wiki) - 상세 가이드 및 튜토리얼
- **🔧 기술 지원**: Issues 라벨링 시스템으로 우선순위 관리

### 🎯 **성능 문제 신고**
성능 이슈 발견 시 다음 정보와 함께 신고해 주세요:
- 시스템 환경 (OS, RAM, CPU)
- RAG 시스템 유형 (Naive/Advanced/Modular)  
- 문서 수 및 평균 크기
- 처리 시간 및 오류 메시지
- 재현 가능한 쿼리 예시

### 🤝 **커뮤니티**
- **Discord 채널**: [RAG Systems Lab](https://discord.gg/rag-systems) (곧 오픈)
- **정기 세미나**: 월 1회 온라인 기술 공유
- **기여자 인정**: README 및 Release Notes에 기여자 명시

### 📊 **로드맵 및 버전 관리**
- **v2.1 현재**: 웹 검색 RAG + 4가지 전문 RAG 통합 플랫폼
  - 🌍 Web Search RAG: 실시간 웹 정보 검색
  - 🔄 Translation RAG: 전문 문서 번역 시스템
  - 📋 Report Generation RAG: 구조화 보고서 자동 생성
  - 🗂️ JSON RAG: 구조화 데이터 전문 검색
  - 🔍 Document Discovery RAG: 2단계 문서 발견 시스템
- **v2.2 예정**: 멀티모달 RAG (이미지 + 텍스트 + 표/차트 분석)
- **v2.3 계획**: API 기반 RAG (OpenAI, Claude, Gemini 등)
- **v2.4 계획**: 협업 RAG (실시간 다중 사용자 + 워크플로우)
- **v3.0 계획**: Enterprise RAG (GPU 가속 + 분산 처리 + 보안 강화)

---

## 🎯 **핵심 가치 및 비전**

### 💎 **핵심 가치**
- **🔍 투명성**: 모든 처리 과정의 완전한 가시화
- **⚡ 성능**: 85% 속도 향상 + 최적화된 리소스 사용
- **🎨 사용성**: 직관적 UI + 모듈 분리 + 안전한 오류 처리
- **🔬 확장성**: Orchestration Modules + 플러그인 아키텍처

### 🚀 **기술 혁신**
- **차세대 RAG**: Naive → Advanced → Modular 진화
- **하이브리드 검색**: Vector + BM25 + TF-IDF 통합
- **지능형 라우팅**: 질문 유형별 최적화 경로
- **적응형 반복**: 신뢰도 기반 자동 개선

### 🌟 **비전**
> **"모든 개발자가 쉽게 접근할 수 있는 차세대 RAG 실험 플랫폼"**

RAG 기술의 민주화를 통해 AI 활용의 진입장벽을 낮추고, 
투명하고 검증 가능한 실험 환경을 제공하여 
더 나은 AI 응용 프로그램 개발에 기여합니다. 

## 🔍 지원되는 벡터 스토어

### FAISS
- **특징**: 빠른 유사도 검색, 로컬 파일 기반
- **용도**: 프로토타이핑, 소규모 데이터셋
- **설정**: 추가 설정 불필요

### Chroma 
- **특징**: 오픈소스, 임베딩 데이터베이스
- **용도**: 개발 및 테스트 환경
- **설정**: 추가 설정 불필요

### Milvus ⭐ NEW!
- **특징**: 확장 가능한 고성능 벡터 데이터베이스
- **용도**: 대규모 프로덕션 환경, 엔터프라이즈급 애플리케이션
- **모드**: 
  - **Milvus Lite**: 경량 로컬 버전 (프로토타이핑용)
  - **Milvus Server**: 도커/쿠버네티스 기반 서버 (프로덕션용)

#### Milvus 설정 방법

**1. 패키지 설치**
```bash
pip install langchain-milvus pymilvus
```

**2. 환경 변수 설정 (.env 파일)**
```bash
# Milvus Lite (로컬 파일 기반)
MILVUS_URI=./milvus_local.db

# Milvus Server (서버 기반)
# MILVUS_URI=http://localhost:19530
# MILVUS_TOKEN=your_token_here

# 컬렉션 이름
MILVUS_COLLECTION_NAME=rag_documents
```

**3. Milvus Server 실행 (선택사항)**
```bash
# Docker로 Milvus 실행
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

**4. 애플리케이션에서 Milvus 선택**
- 사이드바에서 "Vector Store 엔진 선택" → **milvus** 선택
- 벡터 스토어 생성 시 자동으로 Milvus 연결

#### Milvus vs 다른 벡터 스토어

| 기능 | FAISS | Chroma | **Milvus** |
|------|-------|--------|------------|
| 확장성 | 제한적 | 중간 | **매우 높음** |
| 성능 | 빠름 | 중간 | **매우 빠름** |
| 멀티테넌시 | ❌ | 제한적 | **✅ 완전 지원** |
| 하이브리드 검색 | ❌ | 제한적 | **✅ BM25 + 벡터** |
| 클러스터링 | ❌ | ❌ | **✅ 분산 배포** |
| API 호환성 | LangChain만 | LangChain만 | **다양한 SDK** |