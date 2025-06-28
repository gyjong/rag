"""Embedding utilities for the RAG application."""

from typing import List, Optional
import streamlit as st
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
import os


class EmbeddingManager:
    """Manages embedding models for the RAG systems."""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct", models_folder: Path = None):
        """Initialize the embedding manager.
        
        Args:
            model_name: Name of the HuggingFace embedding model
            models_folder: Local folder to store models
        """
        self.model_name = model_name
        self.models_folder = models_folder or Path("models")
        self.models_folder.mkdir(exist_ok=True)
        self._embeddings = None
        
    @st.cache_resource
    def get_embeddings(_self) -> Embeddings:
        """Get the embeddings model (cached).
        
        Returns:
            Embeddings instance
        """
        if _self._embeddings is None:
            with st.spinner(f"임베딩 모델 로딩 중: {_self.model_name}"):
                # Set cache directory to local models folder
                cache_folder = str(_self.models_folder / "embeddings")
                
                _self._embeddings = HuggingFaceEmbeddings(
                    model_name=_self.model_name,
                    cache_folder=cache_folder,
                    model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
                    encode_kwargs={'normalize_embeddings': True}
                )
        return _self._embeddings
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.get_embeddings()
        return embeddings.embed_documents(texts)
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector for the query
        """
        embeddings = self.get_embeddings()
        return embeddings.embed_query(query)
    
    def get_model_info(self) -> dict:
        """Get information about the embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "model_type": "HuggingFaceEmbeddings",
            "dimension": 1024,  # multilingual-e5-large-instruct dimension
            "max_sequence_length": 512,
            "cache_folder": str(self.models_folder / "embeddings")
        } 