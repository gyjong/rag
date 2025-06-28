"""Vector store utilities for the RAG application."""

from typing import List, Optional, Dict, Any
import streamlit as st
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
import chromadb
import tempfile
import shutil
from pathlib import Path


class VectorStoreManager:
    """Manages vector stores for the RAG systems."""
    
    def __init__(self, embeddings: Embeddings, collection_name: str = "rag_documents"):
        """Initialize the vector store manager.
        
        Args:
            embeddings: Embeddings instance
            collection_name: Name of the vector store collection
        """
        self.embeddings = embeddings
        self.collection_name = collection_name
        self._vector_store = None
        self._temp_dir = None
        
    def create_vector_store(self, documents: List[Document]) -> VectorStore:
        """Create a vector store from documents.
        
        Args:
            documents: List of documents to index
            
        Returns:
            VectorStore instance
        """
        if not documents:
            raise ValueError("문서가 제공되지 않았습니다.")
            
        # Create temporary directory for ChromaDB
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp()
            
        with st.spinner(f"벡터 스토어 생성 중... ({len(documents)}개 문서)"):
            try:
                self._vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    persist_directory=self._temp_dir
                )
                st.success(f"벡터 스토어가 성공적으로 생성되었습니다. ({len(documents)}개 문서 인덱싱)")
                
            except Exception as e:
                st.error(f"벡터 스토어 생성 실패: {str(e)}")
                raise
                
        return self._vector_store
    
    def get_vector_store(self) -> Optional[VectorStore]:
        """Get the current vector store.
        
        Returns:
            VectorStore instance or None if not created
        """
        return self._vector_store
    
    def similarity_search(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """Perform similarity search.
        
        Args:
            query: Search query
            k: Number of documents to return
            **kwargs: Additional search parameters
            
        Returns:
            List of similar documents
        """
        if self._vector_store is None:
            raise ValueError("벡터 스토어가 생성되지 않았습니다.")
            
        return self._vector_store.similarity_search(query, k=k, **kwargs)
    
    def similarity_search_with_score(self, query: str, k: int = 5, **kwargs) -> List[tuple]:
        """Perform similarity search with scores.
        
        Args:
            query: Search query
            k: Number of documents to return
            **kwargs: Additional search parameters
            
        Returns:
            List of (document, score) tuples
        """
        if self._vector_store is None:
            raise ValueError("벡터 스토어가 생성되지 않았습니다.")
            
        return self._vector_store.similarity_search_with_score(query, k=k, **kwargs)
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        if self._vector_store is None:
            raise ValueError("벡터 스토어가 생성되지 않았습니다.")
            
        with st.spinner(f"문서 추가 중... ({len(documents)}개)"):
            self._vector_store.add_documents(documents)
            st.success(f"{len(documents)}개 문서가 추가되었습니다.")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if self._vector_store is None:
            return {"status": "벡터 스토어가 생성되지 않았습니다."}
            
        try:
            # Get collection info from ChromaDB
            collection = self._vector_store._collection
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "status": "활성화됨"
            }
        except Exception as e:
            return {"status": f"통계 조회 실패: {str(e)}"}
    
    def cleanup(self) -> None:
        """Clean up temporary resources."""
        if self._temp_dir and Path(self._temp_dir).exists():
            try:
                shutil.rmtree(self._temp_dir)
                self._temp_dir = None
            except Exception as e:
                st.warning(f"임시 디렉토리 정리 실패: {str(e)}")
    
    def __del__(self):
        """Destructor to clean up resources."""
        self.cleanup() 