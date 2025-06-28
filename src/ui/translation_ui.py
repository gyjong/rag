"""Translation UI for document translation functionality."""

import streamlit as st
import io
from typing import Dict, Any, Optional
from datetime import datetime

from ..rag_systems.translation_rag import TranslationRAG
from ..utils.llm_manager import LLMManager
from ..config import (
    SUPPORTED_SOURCE_LANGUAGES, 
    SUPPORTED_TARGET_LANGUAGES,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    SUPPORTED_TRANSLATION_FILE_TYPES
)


class TranslationUI:
    """UI class for document translation."""
    
    @staticmethod
    def display_translation_tab():
        """Display the translation tab interface."""
        st.markdown("## 🌐 문서 번역")
        st.markdown("---")
        
        # Check LLM availability
        if not TranslationUI._check_llm_availability():
            return
        
        # Initialize translation system
        translation_rag = TranslationUI._get_translation_system()
        if translation_rag is None:
            return
        
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            TranslationUI._display_upload_section(translation_rag)
        
        with col2:
            TranslationUI._display_settings_section()
        
        # Translation results section
        if "translation_result" in st.session_state and st.session_state.translation_result:
            TranslationUI._display_results_section(st.session_state.translation_result)
    
    @staticmethod
    def _check_llm_availability() -> bool:
        """Check if LLM is available.
        
        Returns:
            True if LLM is available, False otherwise
        """
        try:
            llm_manager = LLMManager(
                st.session_state.get("selected_llm_model", "gemma3:12b-it-qat"),
                "http://localhost:11434",
                st.session_state.get("llm_temperature", 0.1)
            )
            
            model_info = llm_manager.get_model_info()
            
            if not model_info["connection_status"]:
                st.error("❌ Ollama 서버에 연결할 수 없습니다. 먼저 Ollama 서버를 실행해주세요.")
                st.code("ollama serve")
                return False
            
            if not model_info["model_available"]:
                st.error(f"❌ 모델 '{llm_manager.model_name}'을 찾을 수 없습니다.")
                st.code(f"ollama pull {llm_manager.model_name}")
                return False
            
            return True
            
        except Exception as e:
            st.error(f"LLM 확인 중 오류: {str(e)}")
            return False
    
    @staticmethod
    def _get_translation_system() -> Optional[TranslationRAG]:
        """Get or create translation system.
        
        Returns:
            TranslationRAG instance or None if failed
        """
        try:
            if "translation_rag" not in st.session_state:
                llm_manager = LLMManager(
                    st.session_state.get("selected_llm_model", "gemma3:12b-it-qat"),
                    "http://localhost:11434",
                    st.session_state.get("llm_temperature", 0.1)
                )
                st.session_state.translation_rag = TranslationRAG(llm_manager)
            
            return st.session_state.translation_rag
            
        except Exception as e:
            st.error(f"번역 시스템 초기화 오류: {str(e)}")
            return None
    
    @staticmethod
    def _display_upload_section(translation_rag: TranslationRAG):
        """Display file upload and input section.
        
        Args:
            translation_rag: Translation RAG system instance
        """
        st.subheader("📁 문서 입력")
        
        # File upload
        uploaded_file = st.file_uploader(
            "번역할 문서를 업로드하세요",
            type=SUPPORTED_TRANSLATION_FILE_TYPES,
            help=f"지원 형식: {', '.join(ext.upper() for ext in SUPPORTED_TRANSLATION_FILE_TYPES)}"
        )
        
        # Text input alternative
        st.markdown("**또는 직접 텍스트 입력:**")
        manual_text = st.text_area(
            "번역할 텍스트를 입력하세요",
            height=150,
            placeholder="여기에 영어 텍스트를 입력하세요..."
        )
        
        # Process input
        text_to_translate = None
        
        if uploaded_file is not None:
            with st.spinner("파일 처리 중..."):
                text_to_translate = translation_rag.process_uploaded_file(uploaded_file)
                if text_to_translate:
                    st.success(f"✅ 파일 '{uploaded_file.name}' 처리 완료")
                    with st.expander("📄 추출된 텍스트 미리보기"):
                        st.text_area("", text_to_translate[:1000] + "..." if len(text_to_translate) > 1000 else text_to_translate, height=100, disabled=True)
                else:
                    st.error("❌ 파일 처리에 실패했습니다.")
        
        elif manual_text.strip():
            text_to_translate = manual_text.strip()
            st.info("✅ 수동 입력 텍스트 준비 완료")
        
        # Store text in session state
        st.session_state.text_to_translate = text_to_translate
        
        # Translation button
        if text_to_translate:
            st.markdown("---")
            if st.button("🚀 번역 시작", type="primary", use_container_width=True):
                TranslationUI._perform_translation(translation_rag, text_to_translate)
    
    @staticmethod
    def _display_settings_section():
        """Display translation settings section."""
        st.subheader("⚙️ 번역 설정")
        
        # Language selection
        col1, col2 = st.columns(2)
        
        with col1:
            source_lang = st.selectbox(
                "원본 언어",
                options=SUPPORTED_SOURCE_LANGUAGES,
                index=SUPPORTED_SOURCE_LANGUAGES.index(DEFAULT_SOURCE_LANGUAGE) if DEFAULT_SOURCE_LANGUAGE in SUPPORTED_SOURCE_LANGUAGES else 0,
                help="번역할 문서의 원본 언어를 선택하세요"
            )
        
        with col2:
            target_lang = st.selectbox(
                "대상 언어",
                options=SUPPORTED_TARGET_LANGUAGES,
                index=SUPPORTED_TARGET_LANGUAGES.index(DEFAULT_TARGET_LANGUAGE) if DEFAULT_TARGET_LANGUAGE in SUPPORTED_TARGET_LANGUAGES else 0,
                help="번역할 대상 언어를 선택하세요"
            )
        
        # Store settings in session state
        st.session_state.translation_source_lang = source_lang
        st.session_state.translation_target_lang = target_lang
        
        # Translation options
        st.markdown("**번역 옵션:**")
        
        preserve_formatting = st.checkbox(
            "서식 보존",
            value=True,
            help="원본 문서의 줄 바꿈과 문단 구조를 유지합니다"
        )
        
        show_sentence_pairs = st.checkbox(
            "문장별 비교 표시",
            value=True,
            help="번역 결과에 원문과 번역문을 문장별로 비교 표시합니다"
        )
        
        # Store options in session state
        st.session_state.preserve_formatting = preserve_formatting
        st.session_state.show_sentence_pairs = show_sentence_pairs
        
        # Translation info
        st.markdown("---")
        st.markdown("**ℹ️ 번역 정보:**")
        st.info(f"""
        - **원본 언어:** {source_lang}
        - **대상 언어:** {target_lang}
        - **LLM 모델:** {st.session_state.get('selected_llm_model', 'Unknown')}
        - **Temperature:** {st.session_state.get('llm_temperature', 0.1)}
        """)
    
    @staticmethod
    def _perform_translation(translation_rag: TranslationRAG, text: str):
        """Perform document translation.
        
        Args:
            translation_rag: Translation RAG system instance
            text: Text to translate
        """
        source_lang = st.session_state.get("translation_source_lang", "English")
        target_lang = st.session_state.get("translation_target_lang", "Korean")
        
        with st.spinner(f"{source_lang}에서 {target_lang}으로 번역 중..."):
            try:
                result = translation_rag.translate_document(
                    text=text,
                    source_lang=source_lang,
                    target_lang=target_lang
                )
                
                if result.get("success", False):
                    st.session_state.translation_result = result
                    st.success("✅ 번역이 완료되었습니다!")
                    st.rerun()
                else:
                    st.error(f"❌ 번역 실패: {result.get('error', '알 수 없는 오류')}")
                    
            except Exception as e:
                st.error(f"❌ 번역 중 오류가 발생했습니다: {str(e)}")
    
    @staticmethod
    def _display_results_section(result: Dict[str, Any]):
        """Display translation results.
        
        Args:
            result: Translation result dictionary
        """
        st.markdown("## 📋 번역 결과")
        st.markdown("---")
        
        # Translation statistics
        stats = TranslationUI._get_translation_stats(result)
        
        # Display stats in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 문장 수", stats.get("total_sentences", 0))
        
        with col2:
            st.metric("번역 문장 수", stats.get("translated_sentences", 0))
        
        with col3:
            st.metric("원문 단어 수", stats.get("original_word_count", 0))
        
        with col4:
            st.metric("번역 단어 수", stats.get("translated_word_count", 0))
        
        # Main translation result
        st.subheader("📄 번역된 문서")
        
        # Display translated text
        translated_text = result.get("translated_text", "")
        st.text_area(
            "번역 결과",
            translated_text,
            height=300,
            disabled=True
        )
        
        # Download button
        TranslationUI._display_download_section(result)
        
        # Sentence-by-sentence comparison
        if st.session_state.get("show_sentence_pairs", True):
            TranslationUI._display_sentence_comparison(result)
    
    @staticmethod
    def _get_translation_stats(result: Dict[str, Any]) -> Dict[str, Any]:
        """Get translation statistics.
        
        Args:
            result: Translation result dictionary
            
        Returns:
            Statistics dictionary
        """
        if "translation_rag" not in st.session_state:
            return {}
        
        translation_rag = st.session_state.translation_rag
        return translation_rag.get_translation_stats(result)
    
    @staticmethod
    def _display_download_section(result: Dict[str, Any]):
        """Display download options.
        
        Args:
            result: Translation result dictionary
        """
        st.subheader("💾 다운로드 옵션")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download translated text only
            translated_text = result.get("translated_text", "")
            st.download_button(
                label="📄 번역문만 다운로드",
                data=translated_text,
                file_name=f"translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # Download full report
            if "translation_rag" in st.session_state:
                translation_rag = st.session_state.translation_rag
                export_text = translation_rag.export_translation_result(result)
                
                st.download_button(
                    label="📊 상세 리포트 다운로드",
                    data=export_text,
                    file_name=f"translation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
    
    @staticmethod
    def _display_sentence_comparison(result: Dict[str, Any]):
        """Display sentence-by-sentence comparison.
        
        Args:
            result: Translation result dictionary
        """
        st.subheader("🔍 문장별 번역 비교")
        
        sentence_pairs = result.get("sentence_pairs", [])
        
        if not sentence_pairs:
            st.warning("문장별 비교 데이터가 없습니다.")
            return
        
        # Filter out skipped sentences
        filtered_pairs = [pair for pair in sentence_pairs if not pair.get("skipped", False)]
        
        if not filtered_pairs:
            st.warning("번역된 문장이 없습니다.")
            return
        
        # Pagination for large documents
        items_per_page = 10
        total_pages = (len(filtered_pairs) + items_per_page - 1) // items_per_page
        
        if total_pages > 1:
            page = st.selectbox(
                "페이지 선택",
                options=list(range(1, total_pages + 1)),
                index=0,
                format_func=lambda x: f"페이지 {x} ({(x-1)*items_per_page + 1}-{min(x*items_per_page, len(filtered_pairs))})"
            )
        else:
            page = 1
        
        # Calculate slice
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(filtered_pairs))
        page_pairs = filtered_pairs[start_idx:end_idx]
        
        # Display pairs
        for i, pair in enumerate(page_pairs, start=start_idx + 1):
            with st.expander(f"문장 {i}: {pair.get('original', '')[:50]}..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**원문:**")
                    st.text_area(
                        "원문",
                        pair.get("original", ""),
                        height=100,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"original_{i}"
                    )
                
                with col2:
                    st.markdown("**번역:**")
                    st.text_area(
                        "번역",
                        pair.get("translated", ""),
                        height=100,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"translated_{i}"
                    )
        
        # Clear results button
        st.markdown("---")
        if st.button("🗑️ 번역 결과 지우기", type="secondary"):
            if "translation_result" in st.session_state:
                del st.session_state.translation_result
            st.rerun() 