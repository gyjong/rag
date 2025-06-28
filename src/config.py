"""Configuration settings for the RAG comparison application."""

import os
from pathlib import Path

# Application settings
APP_TITLE = "RAG Systems Comparison Tool"
APP_DESCRIPTION = "단계별 Naive RAG, Advanced RAG, Modular RAG 비교 실험 애플리케이션"

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DOCS_FOLDER = PROJECT_ROOT / "docs"
FONTS_FOLDER = PROJECT_ROOT / "fonts"
FONT_PATH = FONTS_FOLDER / "Paperlogy.ttf"
JSON_OUTPUT_FOLDER = PROJECT_ROOT / "json_data"

# LLM Configuration
DEFAULT_LLM_MODEL = "gemma3:12b-it-qat"
OLLAMA_BASE_URL = "http://localhost:11434"

AVAILABLE_LLM_MODELS = {
    "Gemma 3 (1B)": "gemma3:1b",
    "Gemma 3 (1B-QAT)": "gemma3:1b-it-qat",
    "Gemma 3 (4B)": "gemma3:4b", 
    "Gemma 3 (4B-QAT)": "gemma3:4b-it-qat",
    "Gemma 3 (12B)": "gemma3:12b",
    "Gemma 3 (12B-QAT)": "gemma3:12b-it-qat",
    "Gemma 3 (27B)": "gemma3:27b",
    "Gemma 3 (27B-QAT)": "gemma3:27b-it-qat"
}

# Embedding Configuration
EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
MODELS_FOLDER = PROJECT_ROOT / "models"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector Store Configuration
VECTOR_STORE_TYPE = "chroma"
COLLECTION_NAME = "rag_documents"

# Retrieval Configuration
DEFAULT_K = 5
SIMILARITY_THRESHOLD = 0.7

# Advanced RAG Configuration
RERANK_TOP_K = 3
QUERY_EXPANSION_COUNT = 3

# Modular RAG Configuration
MAX_ITERATIONS = 3
CONFIDENCE_THRESHOLD = 0.8 