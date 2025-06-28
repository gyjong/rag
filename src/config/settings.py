"""
애플리케이션 설정 관리
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 문서 및 폰트 경로
DOCS_DIR = PROJECT_ROOT / "docs"
FONTS_DIR = PROJECT_ROOT / "fonts"
FONT_PATH = FONTS_DIR / "Paperlogy.ttf"

# API 및 모델 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b-it-qat")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")

# RAG 설정
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Streamlit 설정
PAGE_TITLE = "RAG 비교 실험 플랫폼"
PAGE_ICON = "🔬"
LAYOUT = "wide"

# RAG 유형별 설정
RAG_CONFIGS = {
    "naive": {
        "name": "Naive RAG",
        "description": "기본적인 검색-생성 파이프라인",
        "color": "#90EE90",  # Light Green
        "features": [
            "단순한 문서 분할",
            "벡터 유사도 검색",
            "직접적인 답변 생성"
        ]
    },
    "advanced": {
        "name": "Advanced RAG", 
        "description": "최적화된 검색-생성 파이프라인",
        "color": "#FFA500",  # Orange
        "features": [
            "쿼리 전처리 및 확장",
            "하이브리드 검색",
            "재순위화 및 필터링",
            "답변 후처리"
        ]
    },
    "modular": {
        "name": "Modular RAG",
        "description": "모듈화된 유연한 RAG 시스템", 
        "color": "#FF6B6B",  # Red
        "features": [
            "모듈화된 아키텍처",
            "동적 라우팅",
            "다중 검색 전략",
            "적응형 생성"
        ]
    }
}

# 평가 메트릭
EVALUATION_METRICS = [
    "응답 품질",
    "관련성",
    "정확성", 
    "완전성",
    "응답 시간"
]

# Context7 설정
CONTEXT7_CONFIG = {
    "library_id": "/upstash/context7",
    "max_tokens": 10000
} 