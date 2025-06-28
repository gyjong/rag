"""
임베딩 생성 및 관리 컴포넌트
"""
import streamlit as st
from typing import List, Optional
from pathlib import Path
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

from src.config.settings import EMBEDDING_MODEL
from src.config import MODELS_FOLDER


class CustomEmbeddings(Embeddings):
    """커스텀 임베딩 클래스"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        # Use HuggingFace embeddings instead of SentenceTransformer
        cache_folder = str(MODELS_FOLDER / "embeddings")
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_folder,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 리스트에 대한 임베딩 생성"""
        return self.model.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리에 대한 임베딩 생성"""
        return self.model.embed_query(text)


class EmbeddingManager:
    """임베딩 관리 클래스"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.embeddings = self._load_embedding_model()
    
    @st.cache_resource
    def _load_embedding_model(_self) -> Embeddings:
        """임베딩 모델 로딩 (캐시됨)"""
        try:
            # Set cache directory to local models folder
            cache_folder = str(MODELS_FOLDER / "embeddings")
            MODELS_FOLDER.mkdir(exist_ok=True)
            
            # HuggingFace Embeddings 사용
            embeddings = HuggingFaceEmbeddings(
                model_name=_self.model_name,
                cache_folder=cache_folder,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            return embeddings
        except Exception as e:
            st.error(f"임베딩 모델 로딩 실패: {str(e)}")
            # 폴백으로 커스텀 임베딩 사용
            return CustomEmbeddings(_self.model_name)
    
    def get_embeddings(self) -> Embeddings:
        """임베딩 객체 반환"""
        return self.embeddings
    
    def compute_similarity(self, query_embedding: List[float], 
                          doc_embeddings: List[List[float]]) -> List[float]:
        """쿼리와 문서들 간의 유사도 계산"""
        query_vec = np.array(query_embedding)
        doc_vecs = np.array(doc_embeddings)
        
        # 코사인 유사도 계산
        similarities = np.dot(doc_vecs, query_vec) / (
            np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec)
        )
        
        return similarities.tolist()
    
    def get_model_info(self) -> dict:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "dimension": self._get_embedding_dimension(),
            "max_sequence_length": self._get_max_sequence_length()
        }
    
    def _get_embedding_dimension(self) -> int:
        """임베딩 차원 수 반환"""
        try:
            test_embedding = self.embeddings.embed_query("test")
            return len(test_embedding)
        except:
            return 384  # 기본값
    
    def _get_max_sequence_length(self) -> int:
        """최대 시퀀스 길이 반환"""
        try:
            if hasattr(self.embeddings, 'client') and hasattr(self.embeddings.client, 'max_seq_length'):
                return self.embeddings.client.max_seq_length
            return 512  # 기본값
        except:
            return 512


class HybridEmbeddings:
    """하이브리드 임베딩 (밀집 + 희소)"""
    
    def __init__(self, dense_model: str = EMBEDDING_MODEL):
        self.dense_embeddings = EmbeddingManager(dense_model)
        self.sparse_embeddings = self._init_sparse_embeddings()
    
    def _init_sparse_embeddings(self):
        """희소 임베딩 초기화 (TF-IDF 등)"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def fit_sparse(self, texts: List[str]):
        """희소 임베딩 학습"""
        self.sparse_embeddings.fit(texts)
    
    def get_dense_embeddings(self, texts: List[str]) -> List[List[float]]:
        """밀집 임베딩 생성"""
        return self.dense_embeddings.get_embeddings().embed_documents(texts)
    
    def get_sparse_embeddings(self, texts: List[str]) -> np.ndarray:
        """희소 임베딩 생성"""
        return self.sparse_embeddings.transform(texts).toarray()
    
    def get_hybrid_scores(self, query: str, texts: List[str], 
                         alpha: float = 0.7) -> List[float]:
        """하이브리드 점수 계산 (alpha: 밀집 가중치)"""
        # 밀집 임베딩 유사도
        query_dense = self.dense_embeddings.get_embeddings().embed_query(query)
        doc_dense = self.get_dense_embeddings(texts)
        dense_scores = self.dense_embeddings.compute_similarity(query_dense, doc_dense)
        
        # 희소 임베딩 유사도
        query_sparse = self.sparse_embeddings.transform([query]).toarray()[0]
        doc_sparse = self.get_sparse_embeddings(texts)
        sparse_scores = np.dot(doc_sparse, query_sparse) / (
            np.linalg.norm(doc_sparse, axis=1) * np.linalg.norm(query_sparse)
        )
        
        # 하이브리드 점수
        hybrid_scores = alpha * np.array(dense_scores) + (1 - alpha) * sparse_scores
        return hybrid_scores.tolist()


def get_embedding_manager(model_name: Optional[str] = None) -> EmbeddingManager:
    """임베딩 매니저 팩토리 함수"""
    if model_name is None:
        model_name = EMBEDDING_MODEL
    
    return EmbeddingManager(model_name)


def display_embedding_info(embedding_manager: EmbeddingManager):
    """임베딩 모델 정보 표시"""
    info = embedding_manager.get_model_info()
    
    st.sidebar.markdown("### 🔧 임베딩 모델 정보")
    st.sidebar.markdown(f"**모델**: {info['model_name']}")
    st.sidebar.markdown(f"**차원**: {info['dimension']}")
    st.sidebar.markdown(f"**최대 길이**: {info['max_sequence_length']}") 