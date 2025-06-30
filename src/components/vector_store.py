"""
벡터 스토어 관리 컴포넌트
"""
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStore

try:
    from langchain_milvus import Milvus
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    Milvus = None

from src.components.embeddings import EmbeddingManager

try:
    from src.config.settings import TOP_K, MILVUS_URI, MILVUS_TOKEN, MILVUS_COLLECTION_NAME
except ImportError:
    # 기본값 사용
    TOP_K = 5
    MILVUS_URI = "./milvus_local.db"
    MILVUS_TOKEN = ""
    MILVUS_COLLECTION_NAME = "rag_documents"


class VectorStoreManager:
    """벡터 스토어 관리 클래스"""
    
    def __init__(self, embedding_manager: EmbeddingManager, store_type: str = "faiss"):
        self.embedding_manager = embedding_manager
        self.store_type = store_type
        self.vector_store: Optional[VectorStore] = None
        self.documents: List[Document] = []
    
    def create_vector_store(self, documents: List[Document]) -> VectorStore:
        """벡터 스토어 생성"""
        if not documents:
            raise ValueError("문서가 없습니다.")
        
        self.documents = documents
        embeddings = self.embedding_manager.get_embeddings()
        
        try:
            if self.store_type == "faiss":
                self.vector_store = FAISS.from_documents(documents, embeddings)
            elif self.store_type == "chroma":
                self.vector_store = Chroma.from_documents(documents, embeddings)
            elif self.store_type == "milvus":
                if not MILVUS_AVAILABLE:
                    raise ImportError(
                        "Milvus 사용을 위해 langchain-milvus와 pymilvus를 설치해주세요: "
                        "pip install langchain-milvus pymilvus"
                    )
                
                # Milvus 연결 인자 설정
                connection_args = {"uri": MILVUS_URI}
                if MILVUS_TOKEN:
                    connection_args["token"] = MILVUS_TOKEN
                
                self.vector_store = Milvus.from_documents(
                    documents,
                    embeddings,
                    collection_name=MILVUS_COLLECTION_NAME,
                    connection_args=connection_args,
                    drop_old=False  # 기존 컬렉션 유지
                )
            else:
                raise ValueError(f"지원하지 않는 벡터 스토어 타입: {self.store_type}")
            
            return self.vector_store
            
        except Exception as e:
            st.error(f"벡터 스토어 생성 실패: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = TOP_K) -> List[Document]:
        """유사도 검색"""
        if not self.vector_store:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다.")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = TOP_K) -> List[Tuple[Document, float]]:
        """점수와 함께 유사도 검색"""
        if not self.vector_store:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다.")
        
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def get_retriever(self, search_type: str = "similarity", search_kwargs: Optional[Dict] = None):
        """리트리버 반환"""
        if not self.vector_store:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다.")
        
        if search_kwargs is None:
            search_kwargs = {"k": TOP_K}
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def add_documents(self, documents: List[Document]):
        """문서 추가"""
        if not self.vector_store:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다.")
        
        self.vector_store.add_documents(documents)
        self.documents.extend(documents)
    
    def get_store_info(self) -> Dict[str, Any]:
        """벡터 스토어 정보 반환"""
        info = {
            "store_type": self.store_type,
            "document_count": len(self.documents),
            "embedding_model": self.embedding_manager.model_name
        }
        
        if self.vector_store and hasattr(self.vector_store, 'index'):
            if hasattr(self.vector_store.index, 'ntotal'):
                info["vector_count"] = self.vector_store.index.ntotal
        
        return info


class HybridVectorStore:
    """하이브리드 벡터 스토어 (밀집 + 희소)"""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.dense_store: Optional[VectorStore] = None
        self.sparse_index = None
        self.documents: List[Document] = []
        self._init_sparse_index()
    
    def _init_sparse_index(self):
        """희소 인덱스 초기화"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def create_hybrid_store(self, documents: List[Document]) -> None:
        """하이브리드 스토어 생성"""
        self.documents = documents
        
        # 밀집 벡터 스토어 생성
        embeddings = self.embedding_manager.get_embeddings()
        self.dense_store = FAISS.from_documents(documents, embeddings)
        
        # 희소 인덱스 생성
        texts = [doc.page_content for doc in documents]
        self.sparse_index = self.tfidf_vectorizer.fit_transform(texts)
    
    def hybrid_search(self, query: str, k: int = TOP_K, alpha: float = 0.7) -> List[Tuple[Document, float]]:
        """하이브리드 검색 (alpha: 밀집 벡터 가중치)"""
        if not self.dense_store or self.sparse_index is None:
            raise ValueError("하이브리드 스토어가 초기화되지 않았습니다.")
        
        # 밀집 벡터 검색
        dense_results = self.dense_store.similarity_search_with_score(query, k=k*2)
        
        # 희소 벡터 검색
        query_sparse = self.tfidf_vectorizer.transform([query])
        sparse_scores = cosine_similarity(query_sparse, self.sparse_index).flatten()
        
        # 결과 결합
        combined_results = []
        
        for doc, dense_score in dense_results:
            # 문서 인덱스 찾기
            doc_idx = next((i for i, d in enumerate(self.documents) 
                           if d.page_content == doc.page_content), -1)
            
            if doc_idx >= 0:
                sparse_score = sparse_scores[doc_idx]
                # 점수 정규화 (0-1 범위)
                normalized_dense = 1 / (1 + dense_score) if dense_score > 0 else 0
                normalized_sparse = sparse_score
                
                # 하이브리드 점수 계산
                hybrid_score = alpha * normalized_dense + (1 - alpha) * normalized_sparse
                combined_results.append((doc, hybrid_score))
        
        # 점수 기준 정렬
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        return combined_results[:k]


class MultiModalVectorStore:
    """멀티모달 벡터 스토어 (향후 확장용)"""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.text_store: Optional[VectorStore] = None
        # 향후 이미지, 테이블 등 다른 모달리티 추가 가능
    
    def create_text_store(self, documents: List[Document]) -> VectorStore:
        """텍스트 벡터 스토어 생성"""
        embeddings = self.embedding_manager.get_embeddings()
        self.text_store = FAISS.from_documents(documents, embeddings)
        return self.text_store
    
    # TODO: 이미지, 테이블 등 다른 모달리티 지원 추가


def create_vector_store(documents: List[Document], 
                       embedding_manager: EmbeddingManager,
                       store_type: str = "faiss") -> VectorStoreManager:
    """벡터 스토어 팩토리 함수"""
    manager = VectorStoreManager(embedding_manager, store_type)
    manager.create_vector_store(documents)
    return manager


@st.cache_data
def get_cached_vector_store_info(store_type: str, doc_count: int) -> Dict[str, Any]:
    """캐시된 벡터 스토어 정보"""
    return {
        "store_type": store_type,
        "document_count": doc_count,
        "status": "ready"
    }


def display_vector_store_info(vector_store_manager: VectorStoreManager):
    """벡터 스토어 정보 표시"""
    info = vector_store_manager.get_store_info()
    
    st.sidebar.markdown("### 📊 벡터 스토어 정보")
    st.sidebar.markdown(f"**타입**: {info['store_type']}")
    st.sidebar.markdown(f"**문서 수**: {info['document_count']}")
    
    if "vector_count" in info:
        st.sidebar.markdown(f"**벡터 수**: {info['vector_count']}") 