"""Vector store UI module for the RAG application."""

import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import json

from ..config import *
from ..utils.embeddings import EmbeddingManager
from ..utils.vector_store import VectorStoreManager


class VectorStoreUI:
    """UI components for vector store creation and management."""

    @staticmethod
    def display_vector_store_tab():
        """Enhanced vector store tab with creation/loading/management functionality."""
        st.header("🔍 벡터 스토어 생성 및 관리")

        # Create vector stores output folder if it doesn't exist
        VECTOR_STORES_FOLDER.mkdir(exist_ok=True)

        # Vector store source selection
        st.subheader("🎯 벡터 스토어 소스 선택")

        tab1, tab2, tab3 = st.tabs(["🚀 새 벡터 스토어 생성", "📥 기존 벡터 스토어 로딩", "🗂️ 벡터 스토어 관리"])

        with tab1:
            VectorStoreUI._display_vector_store_creation()

        with tab2:
            VectorStoreUI._display_vector_store_loading()

        with tab3:
            VectorStoreUI._display_vector_store_management()
        
        # Display search test once at the main level if vector store is loaded
        VectorStoreUI._display_search_test()

    @staticmethod
    def _display_vector_store_creation():
        """Display vector store creation tab."""
        st.write("### 🔍 현재 로딩된 문서에서 새 벡터 스토어 생성")

        if not st.session_state.documents_loaded:
            st.warning("⚠️ 먼저 문서를 로딩해주세요.")
            st.info("**📚 문서 로딩** 탭에서 PDF를 로딩하거나 JSON에서 문서를 불러와주세요.")
        else:
            # Display current document info
            chunks = st.session_state.document_chunks
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🧩 총 청크", len(chunks))
            with col2:
                avg_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
                st.metric("📏 평균 크기", f"{avg_size:.0f}")
            with col3:
                sources = set(chunk.metadata.get('source', 'Unknown') for chunk in chunks)
                st.metric("📚 소스 수", len(sources))
            with col4:
                total_chars = sum(len(chunk.page_content) for chunk in chunks)
                st.metric("📝 총 문자", f"{total_chars:,}")

            st.info(f"🤖 임베딩 모델: {EMBEDDING_MODEL}")

            # Vector store creation options
            VectorStoreUI._display_creation_options(chunks, sources, total_chars, avg_size)

    @staticmethod
    def _display_creation_options(chunks, sources, total_chars, avg_size):
        """Display vector store creation options."""
        st.write("### ⚙️ 벡터 스토어 설정")

        # Vector store type (from sidebar setting)
        vector_store_type = st.session_state.get("vector_store_type", "chroma")

        # Generate default collection name (only once per session)
        vector_store_key = f"{vector_store_type}_{len(chunks)}chunks"
        
        if f"default_collection_name_{vector_store_key}" not in st.session_state:
            collection_name_generated = f"vectorstore_{vector_store_type}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            st.session_state[f"default_collection_name_{vector_store_key}"] = collection_name_generated
        
        # Get default name from session state
        collection_name_generated = st.session_state[f"default_collection_name_{vector_store_key}"]

        # Initialize store_name to avoid UnboundLocalError
        store_name = None

        if vector_store_type == "milvus":
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**벡터 스토어 타입:** {vector_store_type.upper()}")

            with col2:
                save_vector_store = st.checkbox("💾 벡터 스토어 저장", value=True)

            col1, col2 = st.columns(2)
            with col2:
                if save_vector_store:
                    store_name = st.text_input(
                        "컬렉션 이름:",
                        value=collection_name_generated,
                        key=f"milvus_collection_name_{vector_store_key}",
                        help="컬렉션 이름을 수정하면 변경 사항이 자동으로 저장됩니다"
                    )
                else:
                    st.write(f"**컬렉션 이름:** {collection_name_generated}")
                    store_name = collection_name_generated
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**벡터 스토어 타입:** {vector_store_type.upper()}")
                st.write(f"**컬렉션 이름:** {COLLECTION_NAME}")

            with col2:
                # Save options
                save_vector_store = st.checkbox("💾 벡터 스토어 저장", value=True)
                if save_vector_store:
                    store_name = st.text_input(
                        "벡터 스토어 이름:",
                        value=collection_name_generated,
                        key=f"vector_store_name_{vector_store_key}",
                        help="벡터 스토어 이름을 수정하면 변경 사항이 자동으로 저장됩니다"
                    )
                else:
                    store_name = COLLECTION_NAME

        # Create vector store button
        if st.button("🚀 벡터 스토어 생성 시작", type="primary"):
            VectorStoreUI._create_vector_store(
                chunks, vector_store_type,
                save_vector_store, store_name if save_vector_store else None,
                total_chars, sources, avg_size
            )

    @staticmethod
    def _create_vector_store(chunks, vector_store_type, save_vector_store, store_name, total_chars, sources, avg_size):
        """Create vector store."""
        try:
            # Initialize embedding manager with models folder
            embedding_manager = EmbeddingManager(EMBEDDING_MODEL, MODELS_FOLDER)
            embeddings = embedding_manager.get_embeddings()

            # Display embedding model info
            embed_info = embedding_manager.get_model_info()
            with st.expander("🤖 임베딩 모델 정보"):
                for key, value in embed_info.items():
                    st.write(f"• **{key}**: {value}")

            # Create vector store manager
            vector_store_manager = VectorStoreManager(
                embeddings,
                vector_store_type=vector_store_type,
                collection_name=store_name if save_vector_store else COLLECTION_NAME
            )

            # Create vector store
            vector_store = vector_store_manager.create_vector_store(chunks)

            # Store in session state
            st.session_state.vector_store_manager = vector_store_manager
            st.session_state.embedding_manager = embedding_manager
            st.session_state.vector_store_created = True

            # Store vector store metadata in session state for consistency
            st.session_state.vector_store_metadata = getattr(vector_store_manager, '_metadata', {})
            st.session_state.vector_store_source = 'created'

            # Generate unique vector store ID for change detection
            import time
            st.session_state.vector_store_id = f"created_{int(time.time())}_{len(chunks)}docs"

            # Reset RAG systems and experiments when vector store changes
            if "rag_systems" in st.session_state:
                st.session_state.rag_systems = {}
            if "experiment_results" in st.session_state:
                st.session_state.experiment_results = []
            else:
                st.session_state.experiment_results = []

            # Display collection stats
            stats = vector_store_manager.get_collection_stats()
            st.write("### 📊 벡터 스토어 통계")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 문서 수", stats.get("document_count", "N/A"))
            with col2:
                st.metric("🔧 상태", stats.get("status", "N/A"))
            with col3:
                st.metric("🚫 텔레메트리", stats.get("telemetry_status", "N/A"))

            # Save vector store if requested
            if save_vector_store and store_name:
                if vector_store_type == "milvus":
                    st.info("ℹ️ Milvus는 서버 기반 스토리지입니다. 컬렉션 정보만 저장됩니다.")

                store_path = VECTOR_STORES_FOLDER / store_name

                # Prepare metadata
                metadata = {
                    "document_count": len(chunks),
                    "total_characters": total_chars,
                    "source_count": len(sources),
                    "avg_chunk_size": avg_size,
                    "embedding_model": EMBEDDING_MODEL,
                    "chunk_size": st.session_state.get("chunk_size", CHUNK_SIZE),
                    "chunk_overlap": st.session_state.get("chunk_overlap", CHUNK_OVERLAP)
                }

                success = vector_store_manager.save_vector_store(store_path, metadata)
                if success:
                    if vector_store_type == "milvus":
                        st.success(f"✅ 벡터 스토어가 저장되었습니다: {store_name}")
                    else:
                        st.success(f"✅ 벡터 스토어가 저장되었습니다: {store_path}")
                else:
                    st.error(f"❌ 벡터 스토어 저장 실패: {str(e)}")
            st.success(f"✅ 벡터 스토어 생성 완료: {store_name}")

        except Exception as e:
            st.error(f"❌ 벡터 스토어 생성 실패: {str(e)}")

    @staticmethod
    def _display_vector_store_loading():
        """Display vector store loading tab."""
        st.write("### 📥 기존 벡터 스토어에서 로딩")

        # List available vector stores
        saved_stores = VectorStoreManager.list_saved_vector_stores(VECTOR_STORES_FOLDER)

        if saved_stores:
            st.write(f"📁 **사용 가능한 벡터 스토어 ({len(saved_stores)}개):**")

            # Create detailed vector store table
            VectorStoreUI._display_vector_stores_table(saved_stores)

            # Vector store selection and loading
            VectorStoreUI._display_vector_store_selection(saved_stores)
        else:
            VectorStoreUI._display_manual_loading_option()

    @staticmethod
    def _display_vector_stores_table(saved_stores):
        """Display vector stores in a table format."""
        store_data = []
        for i, store in enumerate(saved_stores):
            store_data.append({
                "이름": store['store_name'],
                "타입": store.get('vector_store_type', 'unknown').upper(),
                "문서 수": store.get('document_count', 'N/A'),
                "크기 (MB)": f"{store.get('file_size_mb', 0):.1f}",
                "임베딩 모델": store.get('embedding_model', 'N/A'),
                "컬렉션": store.get('collection_name', 'N/A'),
                "생성일": store.get('created_at', 'N/A')[:10] if store.get('created_at') else 'N/A',
                "생성시간": store.get('created_at', 'N/A')[11:16] if store.get('created_at') and len(store.get('created_at', '')) > 16 else 'N/A'
            })

        if store_data:
            df_stores = pd.DataFrame(store_data)

            # Display the table
            st.write("### 📊 벡터 스토어 목록")
            st.dataframe(df_stores, use_container_width=True)

    @staticmethod
    def _display_vector_store_selection(saved_stores):
        """Display vector store selection and detailed info."""
        st.write("### 🎯 로딩할 벡터 스토어 선택")
        store_options = [f"{store['store_name']} ({store.get('vector_store_type', 'unknown').upper()}) - {store.get('document_count', 'N/A')}개 문서" for store in saved_stores]
        selected_store_idx = st.selectbox(
            "로딩할 벡터 스토어 선택:",
            options=range(len(store_options)),
            format_func=lambda x: store_options[x],
            key="vector_store_selector"
        )

        if selected_store_idx is not None:
            selected_store = saved_stores[selected_store_idx]
            store_path = Path(selected_store["store_path"])

            # Display vector store info
            VectorStoreUI._display_selected_store_info(selected_store, store_path)

            # Load vector store button
            if st.button("📥 벡터 스토어 로딩", type="primary"):
                VectorStoreUI._load_vector_store(selected_store, store_path)

    @staticmethod
    def _display_selected_store_info(selected_store, store_path):
        """Display selected vector store information."""
        st.write("### 📋 벡터 스토어 정보")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 파일 크기", f"{selected_store.get('file_size_mb', 0):.1f} MB")
        with col2:
            st.metric("🔧 타입", selected_store.get('vector_store_type', 'unknown').upper())
        with col3:
            st.metric("📄 문서 수", selected_store.get('document_count', 'N/A'))

        # Additional detailed info
        with st.expander("🔍 상세 정보 및 통계"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**📅 생성 정보:**")
                st.write(f"• 생성 시간: {selected_store.get('created_at', 'N/A')[:19]}")
                st.write(f"• 컬렉션 이름: {selected_store.get('collection_name', 'N/A')}")
                if selected_store.get('vector_store_type', 'unknown').upper() != "MILVUS":
                    st.write(f"• 저장 경로: {store_path}")

                st.write("**🤖 모델 정보:**")
                st.write(f"• 임베딩 모델: {selected_store.get('embedding_model', 'N/A')}")
                st.write(f"• 벡터 스토어 타입: {selected_store.get('vector_store_type', 'unknown').upper()}")

            with col2:
                st.write("**📊 데이터 통계:**")
                if selected_store.get('total_characters'):
                    st.write(f"• 총 문자 수: {selected_store['total_characters']:,}")
                if selected_store.get('avg_chunk_size'):
                    st.write(f"• 평균 청크 크기: {selected_store['avg_chunk_size']:.0f}")
                if selected_store.get('source_count'):
                    st.write(f"• 소스 파일 수: {selected_store['source_count']}")

                st.write("**⚙️ 청크 설정:**")
                if selected_store.get('chunk_size'):
                    st.write(f"• 청크 크기: {selected_store['chunk_size']}")
                    st.write(f"• 청크 오버랩: {selected_store.get('chunk_overlap', 0)}")

        # Storage path info
        if selected_store.get('vector_store_type', 'unknown').upper() != "MILVUS":
            st.info(f"📂 **저장 위치:** `{store_path}`")

    @staticmethod
    def _load_vector_store(selected_store, store_path):
        """Load vector store."""
        try:
            vector_store_type = selected_store.get('vector_store_type', 'chroma')

            embedding_manager = EmbeddingManager(EMBEDDING_MODEL, MODELS_FOLDER)
            embeddings = embedding_manager.get_embeddings()

            vector_store_manager = VectorStoreManager(
                embeddings,
                vector_store_type=vector_store_type,
                collection_name=selected_store.get('collection_name', COLLECTION_NAME)
            )

            vector_store_manager.load_vector_store(store_path)

            st.session_state.vector_store_manager = vector_store_manager
            st.session_state.embedding_manager = embedding_manager
            st.session_state.vector_store_created = True

            # 메타데이터를 세션 상태에 저장하여 일관성 유지
            st.session_state.vector_store_metadata = getattr(vector_store_manager, '_metadata', {})
            st.session_state.vector_store_source = 'loaded'

            # 벡터 스토어 ID 생성
            import time
            st.session_state.vector_store_id = f"loaded_{int(time.time())}_{selected_store['store_name']}"

            # RAG 시스템 및 실험 결과 초기화
            if "rag_systems" in st.session_state:
                st.session_state.rag_systems = {}
            if "experiment_results" in st.session_state:
                st.session_state.experiment_results = []

            st.success(f"✅ 벡터 스토어 '{selected_store['store_name']}' 로딩 완료!")

        except Exception as e:
            st.error(f"❌ 벡터 스토어 로딩 실패: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

    @staticmethod
    def _display_search_test():
        """Display search test functionality if vector store is loaded."""
        if st.session_state.get("vector_store_created", False):
            st.markdown("---")
            st.subheader("🔍 벡터 스토어 검색 테스트")

            # Use form to handle both Enter key and button click
            with st.form(key="vector_search_form"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    test_query = st.text_input("테스트 검색어를 입력하세요:", placeholder="예: AI 트렌드", key="vs_search_query")
                with col2:
                    test_k = st.slider("검색 문서 수:", 1, 10, st.session_state.get("top_k", DEFAULT_K), key="vs_search_k")

                # Form submit button (works for both Enter key and button click)
                search_submitted = st.form_submit_button("🔍 검색 테스트", type="primary", use_container_width=True)

            # Execute search when form is submitted
            if search_submitted and test_query:
                vector_store_manager = st.session_state.vector_store_manager
                try:
                    docs_with_score = vector_store_manager.similarity_search_with_score(test_query, k=test_k)

                    st.write(f"### 📊 검색 결과 ({len(docs_with_score)}개)")

                    for i, (doc, score) in enumerate(docs_with_score):
                        with st.expander(f"📄 문서 {i+1}: {doc.metadata.get('source', 'Unknown')} (점수: {score:.3f})"):
                            st.write(f"**유사도 점수:** {score:.4f}")
                            st.write(f"**출처:** {doc.metadata.get('source', 'Unknown')}")
                            st.write(f"**페이지:** {doc.metadata.get('page_number', 'N/A')}")
                            st.write("**내용:**")
                            content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                            st.text(content_preview)

                except Exception as e:
                    st.error(f"❌ 검색 테스트 실패: {str(e)}")
            elif search_submitted and not test_query:
                st.warning("⚠️ 검색어를 입력해주세요.")

            # Current vector store status
            with st.expander("📊 현재 벡터 스토어 상태"):
                vector_store_manager = st.session_state.vector_store_manager
                stats = vector_store_manager.get_collection_stats()
                for key, value in stats.items():
                    st.write(f"• **{key}**: {value}")

    @staticmethod
    def _display_manual_loading_option():
        """Display manual loading option."""
        st.info("📭 저장된 벡터 스토어가 없습니다. 먼저 새 벡터 스토어를 생성해주세요.")

        # Manual path option as fallback
        st.write("**🔧 수동 로딩 (고급 사용자용):**")
        manual_path = st.text_input(
            "벡터 스토어 폴더 경로를 직접 입력:",
            placeholder="예: /Users/kenny/GitHub/rag/vector_stores/vectorstore_chroma_20250628_1832"
        )

        if manual_path and st.button("🔍 수동 경로에서 로딩"):
            VectorStoreUI._load_manual_path(manual_path)

    @staticmethod
    def _load_manual_path(manual_path):
        """Load vector store from manual path."""
        manual_store_path = Path(manual_path)
        if manual_store_path.exists() and manual_store_path.is_dir():
            metadata_path = manual_store_path / "metadata.json"
            if metadata_path.exists():
                try:
                    # Load metadata
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    st.success(f"✅ 메타데이터 발견: {metadata.get('vector_store_type', 'unknown')} 타입")

                    # Initialize and load
                    embedding_manager = EmbeddingManager(EMBEDDING_MODEL, MODELS_FOLDER)
                    embeddings = embedding_manager.get_embeddings()

                    vector_store_manager = VectorStoreManager(
                        embeddings,
                        vector_store_type=metadata.get('vector_store_type', 'chroma'),
                        collection_name=metadata.get('collection_name', COLLECTION_NAME)
                    )

                    success = vector_store_manager.load_vector_store(manual_store_path)
                    if success:
                        st.session_state.vector_store_manager = vector_store_manager
                        st.session_state.embedding_manager = embedding_manager
                        st.session_state.vector_store_created = True

                        # Store vector store metadata in session state for consistency
                        st.session_state.vector_store_metadata = getattr(vector_store_manager, '_metadata', {})
                        st.session_state.vector_store_source = 'manual_loaded'

                        # Generate unique vector store ID for change detection
                        import time
                        st.session_state.vector_store_id = f"manual_{int(time.time())}_{manual_store_path.name}"

                        # Reset RAG systems and experiments when vector store changes
                        if "rag_systems" in st.session_state:
                            st.session_state.rag_systems = {}
                        if "experiment_results" in st.session_state:
                            st.session_state.experiment_results = []
                        else:
                            st.session_state.experiment_results = []

                        st.info("🔄 **새로운 벡터 스토어가 수동 로딩되었습니다.** RAG 실험이 재설정됩니다.")

                except Exception as e:
                    st.error(f"❌ 수동 로딩 실패: {str(e)}")
            else:
                st.error("❌ metadata.json 파일을 찾을 수 없습니다.")
        else:
            st.error("❌ 지정된 경로가 존재하지 않습니다.")

    @staticmethod
    def _display_vector_store_management():
        """Display vector store management tab."""
        st.write("### 🗂️ 벡터 스토어 파일 관리")

        saved_stores = VectorStoreManager.list_saved_vector_stores(VECTOR_STORES_FOLDER)

        if saved_stores:
            st.write(f"📁 **벡터 스토어 목록 ({len(saved_stores)}개):**")

            # Create detailed vector store list
            store_data = []
            for store in saved_stores:
                store_data.append({
                    "이름": store['store_name'],
                    "타입": store.get('vector_store_type', 'unknown').upper(),
                    "문서 수": store.get('document_count', 'N/A'),
                    "크기 (MB)": store.get('file_size_mb', 0),
                    "생성일": store.get('created_at', 'N/A')[:10]  # Date only
                })

            if store_data:
                df_stores = pd.DataFrame(store_data)
                st.dataframe(df_stores, use_container_width=True)

                # Vector store operations
                VectorStoreUI._display_management_operations(saved_stores)
        else:
            st.info("📭 관리할 벡터 스토어가 없습니다.")

    @staticmethod
    def _display_management_operations(saved_stores):
        """Display vector store management operations."""
        st.write("### 🔧 파일 작업")
        selected_stores = st.multiselect(
            "작업할 벡터 스토어 선택:",
            options=[store['store_name'] for store in saved_stores]
        )

        if selected_stores:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("👁️ 선택 스토어 미리보기"):
                    for store_name in selected_stores:
                        # Find store info
                        store_info = next((s for s in saved_stores if s['store_name'] == store_name), None)
                        if store_info:
                            with st.expander(f"🔍 {store_name}"):
                                st.write(f"**타입:** {store_info.get('vector_store_type', 'unknown').upper()}")
                                st.write(f"**문서 수:** {store_info.get('document_count', 'N/A')}")
                                st.write(f"**컬렉션:** {store_info.get('collection_name', 'N/A')}")
                                st.write(f"**임베딩 모델:** {store_info.get('embedding_model', 'N/A')}")
                                st.write(f"**크기:** {store_info.get('file_size_mb', 0):.1f} MB")

            with col2:
                if st.button("🗑️ 선택된 벡터 스토어 삭제", type="secondary"):
                    for store_name in selected_stores:
                        store_path = VECTOR_STORES_FOLDER / store_name
                        success = VectorStoreManager.delete_vector_store(store_path)
                        if success:
                            st.success(f"✅ {store_name} 삭제 완료")
                        else:
                            st.error(f"❌ {store_name} 삭제 실패")
                    st.rerun()