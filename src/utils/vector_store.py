"""Vector store utilities for the RAG application."""

import os
import sys
import warnings
import contextlib
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
import streamlit as st
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.embeddings import Embeddings

try:
    from langchain_milvus import Milvus
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    Milvus = None
import chromadb
from chromadb.config import Settings
import tempfile
import shutil
from pathlib import Path
import pickle

# Comprehensive ChromaDB telemetry disabling
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLE"] = "False"
os.environ["POSTHOG_DISABLED"] = "True"

# Suppress ChromaDB warnings
warnings.filterwarnings("ignore", category=UserWarning, module="chromadb")

# Monkey patch to disable telemetry completely
try:
    import chromadb.telemetry.posthog as posthog_module
    # Replace posthog capture method with no-op
    original_capture = getattr(posthog_module, 'capture', None)
    if original_capture:
        def dummy_capture(*args, **kwargs):
            pass
        posthog_module.capture = dummy_capture
except (ImportError, AttributeError):
    pass


@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class VectorStoreManager:
    """Manages vector stores for the RAG systems."""

    def __init__(self, embeddings: Embeddings, vector_store_type: str = "chroma", collection_name: str = "rag_documents"):
        """Initialize the vector store manager.

        Args:
            embeddings: Embeddings instance
            vector_store_type: Type of vector store ("chroma" or "faiss")
            collection_name: Name of the vector store collection
        """
        self.embeddings = embeddings
        self.vector_store_type = vector_store_type.lower()
        self.collection_name = collection_name
        self._vector_store = None
        self._temp_dir = None
        self._persist_directory = None
        self._metadata = {}

    def create_vector_store(self, documents: List[Document]) -> VectorStore:
        """Create a vector store from documents.

        Args:
            documents: List of documents to index

        Returns:
            VectorStore instance
        """
        if not documents:
            raise ValueError("문서가 제공되지 않았습니다.")

        with st.spinner(f"벡터 스토어 생성 중... ({len(documents)}개 문서)"):
            try:
                if self.vector_store_type == "chroma":
                    self._vector_store = self._create_chroma_store(documents)
                elif self.vector_store_type == "faiss":
                    self._vector_store = self._create_faiss_store(documents)
                elif self.vector_store_type == "milvus":
                    self._vector_store = self._create_milvus_store(documents)
                else:
                    raise ValueError(f"지원하지 않는 벡터 스토어 타입: {self.vector_store_type}")

                # Set basic metadata when creating vector store
                self._metadata = {
                    "created_at": datetime.now().isoformat(),
                    "vector_store_type": self.vector_store_type,
                    "collection_name": self.collection_name,
                    "document_count": len(documents),
                    "store_name": f"vectorstore_{self.vector_store_type}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                }

                st.success(f"✅ 벡터 스토어가 성공적으로 생성되었습니다. ({len(documents)}개 문서 인덱싱)")

            except Exception as e:
                st.error(f"❌ 벡터 스토어 생성 실패: {str(e)}")
                raise

        return self._vector_store

    def _create_chroma_store(self, documents: List[Document]) -> VectorStore:
        """Create ChromaDB vector store."""
        # Create temporary directory for ChromaDB
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp()

        # Create ChromaDB settings with all telemetry disabled
        chroma_settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True
        )

        # Suppress ChromaDB output and telemetry messages
        with suppress_output():
            # Create ChromaDB client with comprehensive telemetry disabling
            client = chromadb.PersistentClient(
                path=self._temp_dir,
                settings=chroma_settings
            )

            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self._temp_dir,
                client=client
            )

        return vector_store

    def _create_faiss_store(self, documents: List[Document]) -> VectorStore:
        """Create FAISS vector store."""
        return FAISS.from_documents(documents, self.embeddings)

    def _create_milvus_store(self, documents: List[Document]) -> VectorStore:
        """Create Milvus vector store."""
        if not MILVUS_AVAILABLE:
            raise ImportError(
                "Milvus 사용을 위해 langchain-milvus와 pymilvus를 설치해주세요: "
                "pip install langchain-milvus pymilvus"
            )

        # Milvus 연결 설정
        try:
            from src.config.settings import MILVUS_URI, MILVUS_TOKEN
        except ImportError:
            # 기본값 사용
            MILVUS_URI = "./milvus_local.db"
            MILVUS_TOKEN = ""

        connection_args = {"uri": MILVUS_URI}
        if MILVUS_TOKEN:
            connection_args["token"] = MILVUS_TOKEN

        if not os.path.exists(MILVUS_URI):
            Milvus(
                embedding_function=self.embeddings,
                connection_args={"uri": MILVUS_URI},
                index_params={"index_type": "FLAT", "metric_type": "L2"},
            )

        return Milvus.from_documents(
            documents,
            self.embeddings,
            collection_name=self.collection_name,
            connection_args=connection_args,
            drop_old=False,  # 기존 컬렉션 유지
            # 메타데이터 동적 필드 활성화 (선택사항)
            enable_dynamic_field=True
        )

    def save_vector_store(self, save_path: Path, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save the vector store to disk.

        Args:
            save_path: Path to save the vector store
            metadata: Additional metadata to save

        Returns:
            True if successful, False otherwise
        """
        if self._vector_store is None:
            st.error("❌ 저장할 벡터 스토어가 없습니다.")
            return False

        try:
            save_path.mkdir(parents=True, exist_ok=True)

            # Prepare metadata
            save_metadata = {
                "created_at": datetime.now().isoformat(),
                "vector_store_type": self.vector_store_type,
                "collection_name": self.collection_name,
                "embedding_model": getattr(self.embeddings, 'model_name', 'unknown'),
                **(metadata or {})
            }

            if self.vector_store_type == "chroma":
                # Save ChromaDB
                if self._temp_dir:
                    chroma_path = save_path / "chroma_db"
                    if chroma_path.exists():
                        shutil.rmtree(chroma_path)
                    shutil.copytree(self._temp_dir, chroma_path)
                    save_metadata["chroma_path"] = str(chroma_path)

            elif self.vector_store_type == "faiss":
                # Save FAISS
                faiss_path = save_path / "faiss_index"
                self._vector_store.save_local(str(faiss_path))
                save_metadata["faiss_path"] = str(faiss_path)

            elif self.vector_store_type == "milvus":
                # Milvus는 서버 기반이므로 연결 정보만 저장
                save_metadata["milvus_collection"] = self.collection_name
                save_metadata["milvus_connection"] = "Server-based storage"

            # Save metadata
            metadata_path = save_path / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(save_metadata, f, ensure_ascii=False, indent=2)

            # Get collection stats for metadata
            stats = self.get_collection_stats()
            save_metadata.update(stats)

            # Update metadata file with stats
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(save_metadata, f, ensure_ascii=False, indent=2)

            # Store metadata for later access
            self._metadata = save_metadata

            return True

        except Exception as e:
            return False

    def load_vector_store(self, load_path: Path) -> bool:
        """Load a vector store from disk.

        Args:
            load_path: Path to load the vector store from

        Returns:
            True if successful, False otherwise
        """
        try:
            metadata_path = load_path / "metadata.json"
            if not metadata_path.exists():
                st.error("❌ 메타데이터 파일을 찾을 수 없습니다.")
                return False

            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            vector_store_type = metadata.get("vector_store_type", "chroma")
            collection_name = metadata.get("collection_name", "rag_documents")

            if vector_store_type == "chroma":
                chroma_path = load_path / "chroma_db"
                if not chroma_path.exists():
                    st.error("❌ ChromaDB 파일을 찾을 수 없습니다.")
                    return False

                # Create new temp directory and copy ChromaDB
                if self._temp_dir:
                    shutil.rmtree(self._temp_dir)
                self._temp_dir = tempfile.mkdtemp()
                shutil.copytree(chroma_path, self._temp_dir, dirs_exist_ok=True)

                # Load ChromaDB
                chroma_settings = Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )

                with suppress_output():
                    client = chromadb.PersistentClient(
                        path=self._temp_dir,
                        settings=chroma_settings
                    )

                    self._vector_store = Chroma(
                        collection_name=collection_name,
                        embedding_function=self.embeddings,
                        persist_directory=self._temp_dir,
                        client=client
                    )

            elif vector_store_type == "faiss":
                faiss_path = load_path / "faiss_index"
                if not faiss_path.exists():
                    st.error("❌ FAISS 인덱스 파일을 찾을 수 없습니다.")
                    return False

                # Load FAISS
                self._vector_store = FAISS.load_local(
                    str(faiss_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )

            elif vector_store_type == "milvus":
                if not MILVUS_AVAILABLE:
                    st.error("❌ Milvus 사용을 위해 langchain-milvus와 pymilvus를 설치해주세요.")
                    return False

                # Milvus는 서버 기반이므로 연결만 확인
                try:
                    from src.config.settings import MILVUS_URI, MILVUS_TOKEN
                except ImportError:
                    # 기본값 사용
                    MILVUS_URI = "./milvus_local.db"
                    MILVUS_TOKEN = ""

                connection_args = {"uri": MILVUS_URI}
                if MILVUS_TOKEN:
                    connection_args["token"] = MILVUS_TOKEN

                try:
                    # 기존 컬렉션에 연결
                    self._vector_store = Milvus(
                        embedding_function=self.embeddings,
                        collection_name=collection_name,
                        connection_args=connection_args
                    )
                    st.info("ℹ️ Milvus 컬렉션에 연결되었습니다.")
                except Exception as e:
                    st.error(f"❌ Milvus 연결 실패: {str(e)}")
                    return False

            else:
                st.error(f"❌ 지원하지 않는 벡터 스토어 타입: {vector_store_type}")
                return False

            # Update instance variables
            self.vector_store_type = vector_store_type
            self.collection_name = collection_name

            # Store metadata for later access
            self._metadata = metadata

            if vector_store_type == "milvus":
                st.success(f"✅ 벡터 스토어가 로딩되었습니다: {self.collection_name}")
            else:
                st.success(f"✅ 벡터 스토어가 로딩되었습니다: {load_path}")

            return True

        except Exception as e:
            st.error(f"❌ 벡터 스토어 로딩 실패: {str(e)}")
            return False

    @staticmethod
    def get_vector_store_info(store_path: Path) -> Optional[Dict[str, Any]]:
        """Get information about a saved vector store.

        Args:
            store_path: Path to the vector store

        Returns:
            Dictionary with vector store information or None if failed
        """
        try:
            metadata_path = store_path / "metadata.json"
            if not metadata_path.exists():
                return None

            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Add file size info
            total_size = 0
            for file_path in store_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size

            metadata["file_size_mb"] = round(total_size / (1024 * 1024), 1)
            metadata["store_path"] = str(store_path)
            metadata["store_name"] = store_path.name

            return metadata

        except Exception as e:
            return {"error": str(e), "store_path": str(store_path)}

    @staticmethod
    def list_saved_vector_stores(base_path: Path) -> List[Dict[str, Any]]:
        """List all saved vector stores.

        Args:
            base_path: Base directory to search for vector stores

        Returns:
            List of vector store information dictionaries
        """
        stores = []
        if not base_path.exists():
            return stores

        for store_dir in base_path.iterdir():
            if store_dir.is_dir():
                info = VectorStoreManager.get_vector_store_info(store_dir)
                if info:
                    stores.append(info)

        # Sort by creation date (newest first)
        stores.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return stores

    @staticmethod
    def delete_vector_store(store_path: Path) -> bool:
        """Delete a saved vector store.

        Args:
            store_path: Path to the vector store to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            if store_path.exists():
                shutil.rmtree(store_path)
                return True
            return False
        except Exception as e:
            st.error(f"❌ 벡터 스토어 삭제 실패: {str(e)}")
            return False

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
            st.success(f"✅ {len(documents)}개 문서가 추가되었습니다.")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection.

        Returns:
            Dictionary with collection statistics
        """
        if self._vector_store is None:
            return {"status": "❌ 벡터 스토어가 생성되지 않았습니다."}

        try:
            if self.vector_store_type == "chroma":
                # ChromaDB specific stats
                collection = self._vector_store._collection
                count = collection.count()

                return {
                    "collection_name": self.collection_name,
                    "document_count": count,
                    "status": "✅ 활성화됨",
                    "embedding_dimension": "unknown",
                    "telemetry_status": "✅ 완전 비활성화"
                }

            elif self.vector_store_type == "faiss":
                # FAISS specific stats
                try:
                    # First, try to get from metadata (most reliable)
                    metadata_count = self._metadata.get('document_count')

                    if metadata_count is not None:
                        doc_count = metadata_count
                        status = "✅ 활성화됨"
                    else:
                        # Fallback: try to get from FAISS index
                        if hasattr(self._vector_store, 'index') and self._vector_store.index:
                            index_size = self._vector_store.index.ntotal
                            doc_count = f"{index_size} 벡터"
                            status = "✅ 활성화됨 (인덱스 기반)"
                        else:
                            # Last resort: test with similarity search
                            try:
                                test_docs = self._vector_store.similarity_search("test", k=1)
                                doc_count = "활성" if test_docs else "비어있음"
                                status = "✅ 활성화됨 (검색 테스트)"
                            except:
                                doc_count = "N/A"
                                status = "⚠️ 상태 확인 불가"

                    return {
                        "collection_name": self.collection_name,
                        "document_count": doc_count,
                        "status": status,
                        "embedding_dimension": "unknown",
                        "telemetry_status": "N/A (FAISS)"
                    }

                except Exception as faiss_error:
                    # Complete fallback
                    return {
                        "collection_name": self.collection_name,
                        "document_count": self._metadata.get('document_count', 'N/A'),
                        "status": f"⚠️ FAISS 오류: {str(faiss_error)[:50]}",
                        "embedding_dimension": "unknown",
                        "telemetry_status": "N/A (FAISS)"
                    }

            elif self.vector_store_type == "milvus":
                # Milvus specific stats
                try:
                    # Try to get collection info from Milvus
                    if hasattr(self._vector_store, 'col') and self._vector_store.col:
                        try:
                            # Get collection statistics
                            num_entities = self._vector_store.col.num_entities
                            doc_count = num_entities
                            status = "✅ 활성화됨"
                        except:
                            # Fallback to metadata
                            doc_count = self._metadata.get('document_count', 'N/A')
                            status = "✅ 활성화됨 (메타데이터 기반)"
                    else:
                        # Use metadata
                        doc_count = self._metadata.get('document_count', 'N/A')
                        status = "✅ 활성화됨 (서버 기반)"

                    # Try to get embedding dimension
                    embedding_dim = "unknown"
                    try:
                        if hasattr(self._vector_store, 'col') and self._vector_store.col:
                            schema = self._vector_store.col.schema
                            for field in schema.fields:
                                if field.dtype.name == 'FLOAT_VECTOR':
                                    embedding_dim = field.params.get('dim', 'unknown')
                                    break
                    except:
                        pass

                    return {
                        "collection_name": self.collection_name,
                        "document_count": doc_count,
                        "status": status,
                        "embedding_dimension": embedding_dim,
                        "telemetry_status": "✅ Milvus 연결됨"
                    }

                except Exception as milvus_error:
                    return {
                        "collection_name": self.collection_name,
                        "document_count": self._metadata.get('document_count', 'N/A'),
                        "status": f"⚠️ Milvus 통계 조회 오류: {str(milvus_error)[:50]}",
                        "embedding_dimension": "unknown",
                        "telemetry_status": "⚠️ 연결 문제"
                    }
            else:
                return {
                    "collection_name": self.collection_name,
                    "document_count": "N/A",
                    "status": f"⚠️ 알 수 없는 타입: {self.vector_store_type}",
                    "embedding_dimension": "unknown",
                    "telemetry_status": "N/A"
                }

        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "document_count": self._metadata.get('document_count', 'N/A'),
                "status": f"❌ 통계 조회 실패: {str(e)}",
                "embedding_dimension": "unknown",
                "telemetry_status": "N/A"
            }

    def cleanup(self) -> None:
        """Clean up temporary resources."""
        if self._temp_dir and Path(self._temp_dir).exists():
            try:
                shutil.rmtree(self._temp_dir)
                self._temp_dir = None
            except Exception as e:
                st.warning(f"⚠️ 임시 디렉토리 정리 실패: {str(e)}")

    def __del__(self):
        """Destructor to clean up resources."""
        self.cleanup()