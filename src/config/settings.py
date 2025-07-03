"""
RAG 애플리케이션 통합 설정 관리
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent.parent

# ====== 애플리케이션 기본 설정 ======
APP_TITLE = "RAG Systems Comparison Tool"
APP_DESCRIPTION = "단계별 Naive RAG, Advanced RAG, Modular RAG 비교 실험 애플리케이션"

# ====== 경로 설정 ======
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_FOLDER = DOCS_DIR  # 호환성을 위한 별칭
FONTS_DIR = PROJECT_ROOT / "fonts"
FONT_PATH = FONTS_DIR / "Paperlogy.ttf"
MODELS_FOLDER = PROJECT_ROOT / "models"  # 모델 저장 경로
JSON_OUTPUT_FOLDER = PROJECT_ROOT / "json_data"
VECTOR_STORES_FOLDER = PROJECT_ROOT / "vector_stores"

# ====== LLM 설정 ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b-it-qat")
DEFAULT_LLM_MODEL = "gemma3:12b-it-qat"

# 사용 가능한 LLM 모델 목록
AVAILABLE_LLM_MODELS = {
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "GPT-4o mini": "gpt-4o-mini",
    "Gemma 3 (1B)": "gemma3:1b",
    "Gemma 3 (1B-QAT)": "gemma3:1b-it-qat",
    "Gemma 3 (4B)": "gemma3:4b",
    "Gemma 3 (4B-QAT)": "gemma3:4b-it-qat",
    "Gemma 3 (12B)": "gemma3:12b",
    "Gemma 3 (12B-QAT)": "gemma3:12b-it-qat",
    "Gemma 3 (27B)": "gemma3:27b",
    "Gemma 3 (27B-QAT)": "gemma3:27b-it-qat"
}

# ====== 임베딩 설정 ======
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")

# ====== 벡터 스토어 설정 ======
# Milvus 설정
MILVUS_URI = os.getenv("MILVUS_URI", "./milvus_local.db")  # Lite 버전 기본값
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")  # 서버 모드용
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "rag_documents")

# 기본 벡터 스토어 설정
VECTOR_STORE_TYPE = "chroma"
COLLECTION_NAME = "rag_documents"

# ====== 문서 처리 및 검색 설정 ======
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))  # config.py 값으로 통일
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))  # config.py 값으로 통일
TOP_K = int(os.getenv("TOP_K", "5"))
DEFAULT_K = 5
SIMILARITY_THRESHOLD = 0.7

# ====== Advanced RAG 설정 ======
RERANK_TOP_K = 3
QUERY_EXPANSION_COUNT = 3

# ====== Modular RAG 설정 ======
MAX_ITERATIONS = 3
CONFIDENCE_THRESHOLD = 0.8

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

# ====== 번역 설정 ======
SUPPORTED_SOURCE_LANGUAGES = [
    "English",
    "Korean",
    "Japanese",
    "Chinese",
    "French",
    "German",
    "Spanish",
    "Italian",
    "Portuguese",
    "Dutch",
    "Russian"
]

SUPPORTED_TARGET_LANGUAGES = [
    "Korean",
    "English",
    "Japanese",
    "Chinese",
    "French",
    "German",
    "Spanish",
    "Italian",
    "Portuguese",
    "Dutch",
    "Russian"
]

DEFAULT_SOURCE_LANGUAGE = "English"
DEFAULT_TARGET_LANGUAGE = "Korean"

SUPPORTED_TRANSLATION_FILE_TYPES = ["txt", "pdf", "docx", "md"]


# ========== LangFuse 설정 ==========
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

# API KEY 입력
langfuse = Langfuse(
    public_key="pk-lf-8436e2a2-1b1f-4f23-acfc-d13ba578f470",
    secret_key="sk-lf-2f5292cd-0b57-4324-8158-7dc1167a7abb",
    host="http://localhost:3000"
)

langfuse_handler = CallbackHandler()