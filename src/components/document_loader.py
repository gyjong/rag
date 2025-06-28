"""
문서 로딩 및 전처리 컴포넌트
"""
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from src.config.settings import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


class DocumentLoader:
    """문서 로딩 및 전처리를 담당하는 클래스"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
    
    def get_available_documents(self) -> List[str]:
        """사용 가능한 문서 목록 반환"""
        if not DOCS_DIR.exists():
            return []
        
        pdf_files = list(DOCS_DIR.glob("*.pdf"))
        return [f.name for f in pdf_files]
    
    def load_document(self, filename: str) -> List[Document]:
        """단일 문서 로딩"""
        file_path = DOCS_DIR / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"문서를 찾을 수 없습니다: {filename}")
        
        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            # 메타데이터 추가
            for doc in documents:
                doc.metadata.update({
                    "filename": filename,
                    "source_type": "pdf",
                    "chunk_id": f"{filename}_{documents.index(doc)}"
                })
            
            return documents
            
        except Exception as e:
            st.error(f"문서 로딩 중 오류 발생: {str(e)}")
            return []
    
    def load_multiple_documents(self, filenames: List[str]) -> List[Document]:
        """여러 문서 로딩"""
        all_documents = []
        
        for filename in filenames:
            documents = self.load_document(filename)
            all_documents.extend(documents)
        
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """문서를 청크로 분할"""
        return self.text_splitter.split_documents(documents)
    
    def get_document_info(self, filename: str) -> Dict[str, Any]:
        """문서 정보 조회"""
        file_path = DOCS_DIR / filename
        
        if not file_path.exists():
            return {}
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                info = {
                    "filename": filename,
                    "pages": len(pdf_reader.pages),
                    "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                    "title": pdf_reader.metadata.get('/Title', 'Unknown') if pdf_reader.metadata else 'Unknown'
                }
                
                # 첫 페이지 텍스트 미리보기
                if pdf_reader.pages:
                    first_page_text = pdf_reader.pages[0].extract_text()
                    info["preview"] = first_page_text[:200] + "..." if len(first_page_text) > 200 else first_page_text
                
                return info
                
        except Exception as e:
            st.error(f"문서 정보 조회 중 오류 발생: {str(e)}")
            return {}
    
    def process_documents_for_rag(self, filenames: List[str]) -> List[Document]:
        """RAG를 위한 문서 전처리 파이프라인"""
        # 1. 문서 로딩
        documents = self.load_multiple_documents(filenames)
        
        if not documents:
            return []
        
        # 2. 문서 분할
        chunks = self.split_documents(documents)
        
        # 3. 청크 품질 검증 및 필터링
        filtered_chunks = self._filter_chunks(chunks)
        
        return filtered_chunks
    
    def _filter_chunks(self, chunks: List[Document]) -> List[Document]:
        """청크 품질 필터링"""
        filtered = []
        
        for chunk in chunks:
            text = chunk.page_content.strip()
            
            # 빈 청크 제거
            if not text:
                continue
            
            # 너무 짧은 청크 제거 (50자 미만)
            if len(text) < 50:
                continue
            
            # 의미 없는 텍스트 패턴 제거
            if self._is_meaningless_chunk(text):
                continue
            
            filtered.append(chunk)
        
        return filtered
    
    def _is_meaningless_chunk(self, text: str) -> bool:
        """의미 없는 청크 판별"""
        # 숫자나 특수문자만 있는 경우
        if len(text.strip().replace(' ', '').replace('\n', '')) < 10:
            return True
        
        # 반복되는 문자가 대부분인 경우
        unique_chars = set(text.lower().replace(' ', '').replace('\n', ''))
        if len(unique_chars) < 5:
            return True
        
        return False


@st.cache_data
def get_cached_document_info(filename: str) -> Dict[str, Any]:
    """캐시된 문서 정보 조회"""
    loader = DocumentLoader()
    return loader.get_document_info(filename)


@st.cache_data
def get_cached_documents(filenames: List[str]) -> List[Document]:
    """캐시된 문서 로딩"""
    loader = DocumentLoader()
    return loader.process_documents_for_rag(filenames) 