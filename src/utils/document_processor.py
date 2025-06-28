"""Document processing utilities for the RAG application."""

import os
from pathlib import Path
from typing import List, Optional

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class DocumentProcessor:
    """Handles document loading and processing for RAG systems."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_documents_from_folder(self, folder_path: Path) -> List[Document]:
        """Load all PDF documents from a folder.
        
        Args:
            folder_path: Path to the folder containing PDF files
            
        Returns:
            List of loaded documents
        """
        documents = []
        
        if not folder_path.exists():
            st.error(f"문서 폴더가 존재하지 않습니다: {folder_path}")
            return documents
            
        pdf_files = list(folder_path.glob("*.pdf"))
        
        if not pdf_files:
            st.warning(f"PDF 파일이 없습니다: {folder_path}")
            return documents
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, pdf_file in enumerate(pdf_files):
            try:
                status_text.text(f"문서 로딩 중: {pdf_file.name}")
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                
                # Add metadata
                for doc in docs:
                    doc.metadata.update({
                        "source": pdf_file.name,
                        "file_path": str(pdf_file),
                        "doc_type": "pdf"
                    })
                
                documents.extend(docs)
                progress_bar.progress((i + 1) / len(pdf_files))
                
            except Exception as e:
                st.error(f"문서 로딩 실패 {pdf_file.name}: {str(e)}")
                continue
                
        status_text.text(f"총 {len(documents)} 페이지 로딩 완료")
        progress_bar.empty()
        status_text.empty()
        
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        if not documents:
            return []
            
        with st.spinner("문서 분할 중..."):
            chunks = self.text_splitter.split_documents(documents)
            
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk.page_content)
            })
            
        st.success(f"문서가 {len(chunks)}개의 청크로 분할되었습니다.")
        return chunks

    def get_document_stats(self, documents: List[Document]) -> dict:
        """Get statistics about the loaded documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Dictionary with document statistics
        """
        if not documents:
            return {}
            
        total_chars = sum(len(doc.page_content) for doc in documents)
        sources = set(doc.metadata.get("source", "Unknown") for doc in documents)
        
        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "average_chars_per_doc": total_chars / len(documents) if documents else 0,
            "unique_sources": len(sources),
            "sources": list(sources)
        } 