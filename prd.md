streamlit 기반으로 첨부의 naive rag, advanced rag, modular rag 를 직접 단계별로 실험하고 비교 경험 가능한 application 생성해줘.

1. 패키지 관리: poetry
2. font: fonts/Paperlogy.ttf
3. source documents folder: docs
4. LLM: ollama gemma3:4b-it-qat (from langchain_ollama import ChatOllama)
5. embedding model: huggingface의 multilingual-e5-large-instruct
5. 참조 문서 tool: content7
6. framework: langchain, langgraph

# 모델 설정
EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
DEFAULT_LLM_MODEL = "gemma3:12b-it-qat"
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