"""Document processing utilities for the RAG application."""

import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    Docx2txtLoader
)
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
                
                # Add enhanced metadata
                for page_num, doc in enumerate(docs):
                    doc.metadata.update({
                        "source": pdf_file.name,
                        "file_path": str(pdf_file),
                        "doc_type": "pdf",
                        "page_number": page_num + 1,
                        "total_pages": len(docs),
                        "file_size_mb": round(pdf_file.stat().st_size / (1024 * 1024), 2),
                        "loaded_at": datetime.now().isoformat()
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

    def load_documents_from_file(self, file_path: Path) -> List[Document]:
        """Load a single PDF document from file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of loaded documents from the file
        """
        documents = []
        
        if not file_path.exists():
            raise FileNotFoundError(f"파일이 존재하지 않습니다: {file_path}")
            
        if file_path.suffix.lower() != '.pdf':
            raise ValueError(f"PDF 파일이 아닙니다: {file_path}")
            
        try:
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            
            # Add enhanced metadata
            for page_num, doc in enumerate(docs):
                doc.metadata.update({
                    "source": file_path.name,
                    "file_path": str(file_path),
                    "doc_type": "pdf",
                    "page_number": page_num + 1,
                    "total_pages": len(docs),
                    "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                    "loaded_at": datetime.now().isoformat()
                })
            
            documents.extend(docs)
            
        except Exception as e:
            raise Exception(f"문서 로딩 실패 {file_path.name}: {str(e)}")
                
        return documents

    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """Load documents from multiple files of various formats.
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            List of loaded documents
        """
        documents = []
        
        for file_path_str in file_paths:
            file_path = Path(file_path_str)
            
            if not file_path.exists():
                st.warning(f"파일이 존재하지 않습니다: {file_path}")
                continue
                
            try:
                file_extension = file_path.suffix.lower()
                
                if file_extension == '.pdf':
                    loader = PyPDFLoader(str(file_path))
                    docs = loader.load()
                    doc_type = "pdf"
                    
                elif file_extension == '.txt':
                    # Try different encodings for text files
                    docs = None
                    doc_type = "txt"
                    for encoding in ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr']:
                        try:
                            loader = TextLoader(str(file_path), encoding=encoding)
                            docs = loader.load()
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if docs is None:
                        # If all encodings fail, read with errors='ignore'
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        docs = [Document(page_content=content, metadata={})]
                    
                elif file_extension == '.docx':
                    loader = Docx2txtLoader(str(file_path))
                    docs = loader.load()
                    doc_type = "docx"
                    
                elif file_extension == '.md':
                    # Handle markdown files as text files
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        docs = [Document(page_content=content, metadata={})]
                        doc_type = "markdown"
                    except UnicodeDecodeError:
                        # Try with different encodings
                        for encoding in ['utf-8-sig', 'cp949', 'euc-kr']:
                            try:
                                with open(file_path, 'r', encoding=encoding) as f:
                                    content = f.read()
                                docs = [Document(page_content=content, metadata={})]
                                doc_type = "markdown"
                                break
                            except UnicodeDecodeError:
                                continue
                        else:
                            # If all fail, read with errors='ignore'
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            docs = [Document(page_content=content, metadata={})]
                            doc_type = "markdown"
                    
                else:
                    st.warning(f"지원하지 않는 파일 형식: {file_extension} ({file_path.name})")
                    continue
                
                # Add enhanced metadata
                for i, doc in enumerate(docs):
                    doc.metadata.update({
                        "source": file_path.name,
                        "file_path": str(file_path),
                        "doc_type": doc_type,
                        "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                        "loaded_at": datetime.now().isoformat()
                    })
                    
                    # Add page numbers for PDFs
                    if doc_type == "pdf":
                        doc.metadata.update({
                            "page_number": i + 1,
                            "total_pages": len(docs)
                        })
                
                documents.extend(docs)
                
            except Exception as e:
                st.error(f"문서 로딩 실패 {file_path.name}: {str(e)}")
                continue
                
        return documents

    def save_documents_to_json(self, documents: List[Document], output_path: Path) -> bool:
        """Save loaded documents to JSON format.
        
        Args:
            documents: List of documents to save
            output_path: Path to save the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create structured data
            json_data = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_documents": len(documents),
                    "total_characters": sum(len(doc.page_content) for doc in documents),
                    "chunk_config": {
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap
                    }
                },
                "documents": []
            }
            
            # Convert documents to JSON format
            for i, doc in enumerate(documents):
                doc_data = {
                    "id": i,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "content_stats": {
                        "character_count": len(doc.page_content),
                        "word_count": len(doc.page_content.split()),
                        "line_count": len(doc.page_content.split('\n'))
                    }
                }
                json_data["documents"].append(doc_data)
            
            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            st.success(f"✅ 문서 데이터가 JSON으로 저장되었습니다: {output_path}")
            return True
            
        except Exception as e:
            st.error(f"❌ JSON 저장 실패: {str(e)}")
            return False

    def load_documents_from_json(self, json_path: Path) -> List[Document]:
        """Load documents from JSON format.
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            List of loaded documents
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            documents = []
            for doc_data in json_data.get("documents", []):
                doc = Document(
                    page_content=doc_data["content"],
                    metadata=doc_data["metadata"]
                )
                documents.append(doc)
            
            st.success(f"✅ JSON에서 {len(documents)}개 문서를 로딩했습니다")
            return documents
            
        except Exception as e:
            st.error(f"❌ JSON 로딩 실패: {str(e)}")
            return []

    def split_documents(self, documents: List[Document], merge_pages: bool) -> List[Document]:
        """Split documents into smaller chunks.
        
        Args:
            documents: List of documents to split.
            merge_pages: If True, combines pages of a single file before chunking.
                         If False, chunks each page individually.
                   
        Returns:
            List of chunked documents
        """
        if not documents:
            return []

        if merge_pages:
            # Group documents by file path to process them as single entities
            docs_by_file: Dict[str, List[Document]] = {}
            for doc in documents:
                file_path = doc.metadata.get("file_path")
                if not file_path:
                    # Handle documents without a file path, treat them as a single group
                    file_path = "unknown_source"
                
                if file_path not in docs_by_file:
                    docs_by_file[file_path] = []
                docs_by_file[file_path].append(doc)

            all_chunks = []
            for file_path, docs in docs_by_file.items():
                if not docs:
                    continue

                # For multi-page documents (like PDFs), combine page content
                if len(docs) > 1 and all(d.metadata.get("page_number") is not None for d in docs):
                    # Sort pages by page number to maintain logical order
                    docs.sort(key=lambda d: d.metadata.get("page_number", 0))
                    
                    full_text = "\n".join([doc.page_content for doc in docs])
                    
                    # Use metadata from the first page, but remove page-specific info
                    combined_metadata = docs[0].metadata.copy()
                    combined_metadata.pop("page_number", None)
                    combined_metadata.pop("page", None)
                    combined_metadata.pop("page_label", None)
                    
                    combined_doc = Document(page_content=full_text, metadata=combined_metadata)
                    
                    file_chunks = self.text_splitter.split_documents([combined_doc])
                else:
                    # For single-page docs or docs not from PDFs, split them directly
                    file_chunks = self.text_splitter.split_documents(docs)

                all_chunks.extend(file_chunks)
        else:
            # Original method: split each document (page) individually
            all_chunks = self.text_splitter.split_documents(documents)
            
        # Add chunk-specific metadata to all generated chunks
        for i, chunk in enumerate(all_chunks):
            # Page number tracking is complex/unreliable after merging, so it's omitted
            # in the merged case. For the non-merged case, it's in original metadata.
            chunk.metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk.page_content),
                "word_count": len(chunk.page_content.split()),
                "chunked_at": datetime.now().isoformat()
            })
            
        return all_chunks

    def save_chunks_to_json(self, chunks: List[Document], output_path: Path) -> bool:
        """Save chunked documents to a JSON file.
        
        Args:
            chunks: List of chunks to save
            output_path: Path to save the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create structured chunk data
            json_data = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_chunks": len(chunks),
                    "total_characters": sum(len(chunk.page_content) for chunk in chunks),
                    "average_chunk_size": sum(len(chunk.page_content) for chunk in chunks) / len(chunks) if chunks else 0,
                    "chunk_config": {
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap
                    }
                },
                "chunks": []
            }
            
            # Convert chunks to JSON format
            for chunk in chunks:
                chunk_data = {
                    "id": chunk.metadata.get("chunk_id", 0),
                    "content": chunk.page_content,
                    "metadata": chunk.metadata,
                    "content_stats": {
                        "character_count": len(chunk.page_content),
                        "word_count": len(chunk.page_content.split()),
                        "line_count": len(chunk.page_content.split('\n'))
                    }
                }
                json_data["chunks"].append(chunk_data)
            
            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            st.success(f"✅ 청크 데이터가 JSON으로 저장되었습니다: {output_path}")
            return True
            
        except Exception as e:
            st.error(f"❌ JSON 저장 실패: {str(e)}")
            return False

    def load_chunks_from_json(self, json_path: Path) -> List[Document]:
        """Load document chunks from JSON format.
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            List of loaded chunks
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            chunks = []
            for chunk_data in json_data.get("chunks", []):
                chunk = Document(
                    page_content=chunk_data["content"],
                    metadata=chunk_data["metadata"]
                )
                chunks.append(chunk)
            
            st.success(f"✅ JSON에서 {len(chunks)}개 청크를 로딩했습니다")
            return chunks
            
        except Exception as e:
            st.error(f"❌ JSON 로딩 실패: {str(e)}")
            return []

    def get_json_info(self, json_path: Path) -> Optional[Dict[str, Any]]:
        """Get information about a JSON file.
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            Dictionary with JSON file information
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            metadata = json_data.get("metadata", {})
            
            # Determine if it's documents or chunks
            data_type = "chunks" if "chunks" in json_data else "documents"
            data_list = json_data.get(data_type, [])
            
            return {
                "file_path": str(json_path),
                "file_size_mb": round(json_path.stat().st_size / (1024 * 1024), 2),
                "data_type": data_type,
                "created_at": metadata.get("created_at", "Unknown"),
                "total_items": len(data_list),
                "total_characters": metadata.get("total_characters", 0),
                "chunk_config": metadata.get("chunk_config", {}),
                "sample_content": data_list[0]["content"][:200] + "..." if data_list else ""
            }
            
        except Exception as e:
            st.error(f"❌ JSON 정보 조회 실패: {str(e)}")
            return None

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