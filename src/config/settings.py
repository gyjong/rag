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
RERANK_TOP_K = 3                      # 재순위화 후 선택할 상위 문서 수
QUERY_EXPANSION_COUNT = 5             # 쿼리 확장 시 추가할 키워드 수
CONTEXT_COMPRESSION_MAX_LENGTH = 3000 # 컨텍스트 압축 최대 길이 (문자 수)
                                      # 이 값이 3000인 이유:
                                      # - LLM 컨텍스트 윈도우 크기 고려 (대략 500-800 토큰)
                                      # - 성능과 정확도의 균형
                                      # - 메모리 효율성 및 응답 시간 최적화

# Advanced RAG 도메인 키워드 맵 (쿼리 확장 및 컨텍스트 압축에 사용)
ADVANCED_RAG_DOMAIN_KEYWORD_MAP = {
    "AI": (["ai", "인공지능", "artificial intelligence", "머신러닝", "machine learning", "딥러닝", "deep learning"],
           ["생성형 AI", "generative ai", "신경망", "neural network", "자동화", "automation", "알고리즘", "algorithm", "데이터 분석", "data analysis", "예측 모델", "predictive modeling", "chatgpt", "claude", "llm", "대화형", "conversational", "지능형", "intelligent"]),
    
    "Business": (["업무", "work", "직장", "business", "비즈니스", "회사", "기업", "조직"],
                 ["생산성", "productivity", "효율성", "efficiency", "업무 프로세스", "work process", "디지털 전환", "digital transformation", "혁신", "innovation", "경영", "management", "운영", "operation", "서비스", "service", "고객", "customer"]),
    
    "Manufacturing": (["제조", "manufacturing", "생산", "production", "공장", "factory", "자율제조", "autonomous manufacturing"],
                      ["스마트 팩토리", "smart factory", "스마트 제조", "smart manufacturing", "공정", "process", "품질관리", "quality control", "생산성", "productivity", "자동화", "automation", "로봇", "robot", "설비", "equipment", "라인", "line"]),
    
    "Government": (["정부", "government", "국가", "nation", "공공", "public", "행정", "administration", "부처", "ministry"],
                   ["정부기관", "government agency", "행정기관", "administrative agency", "공공기관", "public agency", "중앙정부", "central government", "지방정부", "local government", "부처", "ministry", "청", "office", "위원회", "committee", "기관", "agency", "공무원", "civil servant", "행정서비스", "administrative service", "공공서비스", "public service", "정부지원", "government support", "공공투자", "public investment", "행정절차", "administrative procedure", "공공예산", "public budget"]),
    
    "Economics": (["경제", "economy", "경제성", "economics", "금융", "finance", "투자", "investment", "자금", "funding"],
                  ["경제성장", "economic growth", "시장경제", "market economy", "경쟁력", "competitiveness", "수익성", "profitability", "비용", "cost", "예산", "budget", "재정", "fiscal", "투자유치", "investment attraction"]),
    
    "Education": (["교육", "education", "학습", "learning", "훈련", "training", "인재", "talent", "역량", "capability"],
                  ["인재양성", "talent development", "교육과정", "curriculum", "스킬", "skill", "역량강화", "capacity building", "전문성", "expertise", "지식", "knowledge", "학습모델", "learning model", "교육프로그램", "education program"]),
    
    "Global": (["국제", "international", "글로벌", "global", "해외", "overseas", "세계", "world", "협력", "cooperation"],
               ["국제협력", "international cooperation", "글로벌 트렌드", "global trend", "해외진출", "overseas expansion", "국제경쟁", "international competition", "글로벌 표준", "global standard", "세계시장", "global market", "국가간", "inter-country"]),
    
    "Data": (["데이터", "data", "정보", "information", "지식", "knowledge", "db", "database", "빅데이터", "big data"],
             ["데이터베이스", "database", "정보화", "informatization", "지식관리", "knowledge management", "데이터 처리", "data processing", "정보시스템", "information system", "데이터 활용", "data utilization", "데이터 분석", "data analytics"]),
    
    "Security": (["보안", "security", "안전", "safety", "위험", "risk", "보호", "protection", "신뢰", "trust"],
                 ["사이버보안", "cybersecurity", "정보보안", "information security", "안전성", "safety", "신뢰성", "reliability", "위험관리", "risk management", "보안체계", "security system", "데이터 보호", "data protection"]),
    
    "Trend": (["트렌드", "trend", "동향", "전망", "미래", "future", "변화", "change", "패러다임", "paradigm"],
              ["시장 동향", "market trend", "기술 동향", "technology trend", "발전 방향", "development direction", "혁신", "innovation", "진화", "evolution", "패러다임 변화", "paradigm shift", "미래예측", "future forecast"]),
    
    "Industry": (["산업", "industry", "시장", "market", "분야", "field", "영역", "domain", "섹터", "sector"],
                 ["산업계", "industrial sector", "시장 분석", "market analysis", "경쟁", "competition", "성장", "growth", "전략", "strategy", "산업정책", "industrial policy", "생태계", "ecosystem", "밸류체인", "value chain"]),
    
    "Analysis": (["분석", "analysis", "연구", "research", "조사", "survey", "평가", "evaluation", "검토", "review"],
                 ["데이터 분석", "data analysis", "통계", "statistics", "조사 결과", "survey results", "연구 보고서", "research report", "실증분석", "empirical analysis", "정량분석", "quantitative analysis", "정성분석", "qualitative analysis"]),
    
    "Strategy": (["전략", "strategy", "방안", "plan", "계획", "planning", "도입", "implementation", "추진", "promotion"],
                 ["전략계획", "strategic planning", "실행 계획", "execution plan", "로드맵", "roadmap", "단계별 접근", "step-by-step approach", "성공 사례", "success case", "추진방안", "implementation plan", "정책수단", "policy instrument"]),
    
    "Performance": (["성능", "performance", "품질", "quality", "효율성", "efficiency", "효과성", "effectiveness", "생산성", "productivity"],
                    ["최적화", "optimization", "개선", "improvement", "측정", "measurement", "평가", "evaluation", "벤치마크", "benchmark", "성과", "achievement", "지표", "indicator", "KPI"]),
    
    "Impact": (["영향", "impact", "효과", "effect", "파급효과", "ripple effect", "변화", "change", "결과", "result"],
               ["성과", "outcome", "개선 효과", "improvement effect", "변화 분석", "change analysis", "영향 평가", "impact assessment", "사회적 영향", "social impact", "경제적 효과", "economic effect"]),
    
    "Technology": (["기술", "technology", "테크", "tech", "솔루션", "solution", "플랫폼", "platform", "시스템", "system"],
                   ["첨단기술", "advanced technology", "핵심기술", "core technology", "기술개발", "technology development", "기술혁신", "technology innovation", "디지털", "digital", "ICT", "클라우드", "cloud", "IoT", "블록체인", "blockchain"]),
    
    "Temporal": (["현재", "current", "미래", "future", "과거", "past", "비교", "compare", "시기", "timing", "기간", "period"],
                 ["시계열 분석", "time series analysis", "트렌드 비교", "trend comparison", "변화 추이", "change trend", "예측", "prediction", "전망", "outlook", "장기", "long-term", "단기", "short-term", "중장기", "mid to long-term"]),
    
    "Legal": (["법", "law", "규제", "regulation", "정책", "policy", "제도", "system", "법률", "legal", "규정", "rule"],
              ["법령", "legislation", "조항", "clause", "시행", "enforcement", "적용", "application", "준수", "compliance", "규제체계", "regulatory framework", "법적근거", "legal basis", "제도개선", "institutional improvement"]),

    "Policy": (["정책", "policy", "정책방향", "policy direction", "국가전략", "national strategy", "공공정책", "public policy", "거버넌스", "governance"],
               ["정책수립", "policy making", "정책결정", "policy decision", "정책실행", "policy implementation", "정책평가", "policy evaluation", "정책개선", "policy improvement", "정책변화", "policy change", "정책효과", "policy effect", "정책목표", "policy objective", "정책수단", "policy instrument", "정책과제", "policy task", "정책의제", "policy agenda", "정책우선순위", "policy priority", "정책연구", "policy research", "정책개발", "policy development", "정책혁신", "policy innovation"]),
}

# TF-IDF 불용어 설정
KOREAN_STOP_WORDS = [
    # 조사
    '은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '으로', '로', '에서', '에게', '한테', 
    '부터', '까지', '만', '도', '라도', '조차', '마저', '보다', '처럼', '같이', '마다', '마냥',
    
    # 어미 (일반적인 것들)
    '다', '습니다', '어요', '아요', '에요', '이에요', '예요', '죠', '지요', '네요', '어요', '아요',
    
    # 접속사/부사
    '그리고', '그런데', '하지만', '그러나', '또한', '또는', '혹은', '즉', '다만', '단지', '오직',
    '비록', '설령', '가령', '만약', '만일', '아마', '정말', '참', '너무', '아주', '매우', '상당히',
    
    # 대명사
    '이', '그', '저', '이것', '그것', '저것', '여기', '거기', '저기', '이곳', '그곳', '저곳',
    '누구', '무엇', '언제', '어디', '어떻게', '왜', '어느', '어떤',
    
    # 일반 불용어
    '등', '및', '또는', '즉', '단', '다만', '때문에', '위해', '위한', '대한', '관한', '통해', '따라', 
    '의해', '에서', '에게', '으로', '로서', '로써', '라고', '이라고', '한다', '된다', '있다', '없다',
    '같다', '다르다', '크다', '작다', '많다', '적다', '좋다', '나쁘다', '높다', '낮다',
    
    # 숫자 및 기타
    '첫', '둘', '셋', '넷', '다섯', '여섯', '일곱', '여덟', '아홉', '열', '하나', '둘', '셋',
    '년', '월', '일', '시', '분', '초', '번째', '개', '명', '회', '차례', '정도', '약간', '조금',
    
    # 감탄사 및 기타
    '아', '어', '오', '음', '으음', '에', '어머', '아이고', '이런', '저런', '그런', '이렇게', '저렇게', '그렇게'
]

# 영어와 한국어 불용어를 결합한 전체 불용어 리스트
COMBINED_STOP_WORDS = None  # 런타임에 생성됨

# ====== Modular RAG 설정 ======
MAX_ITERATIONS = 5
CONFIDENCE_THRESHOLD = 0.7

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

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

langfuse_handler = CallbackHandler()

# ========== RAGAS 평가 설정 ==========
# RAGAS 평가에 사용할 메트릭들
RAGAS_EVALUATION_METRICS = [
    "answer_relevancy",
    "faithfulness", 
    "context_recall",
    "context_precision"
]

# RAGAS 평가 기본 설정
RAGAS_SAMPLE_SIZE = 10      # 평가용 샘플 데이터셋 기본 크기
RAGAS_TIMEOUT = 300         # 평가 타임아웃 (초)
RAGAS_BATCH_SIZE = 5        # 배치 처리 크기
RAGAS_RANDOM_SEED = 42      # 재현 가능한 결과를 위한 시드

# RAGAS 평가 대상 RAG 시스템
RAGAS_AVAILABLE_MODELS = [
    "naive",
    "advanced", 
    "modular"
]

# RAGAS 평가 결과 저장 경로
RAGAS_RESULTS_DIR = PROJECT_ROOT / "ragas_dataset"
RAGAS_RESULTS_FILE = RAGAS_RESULTS_DIR / "evaluation_results.json"