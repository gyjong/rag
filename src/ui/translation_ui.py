"""Translation UI for document translation functionality."""

import streamlit as st
import io
from typing import Dict, Any, Optional
from datetime import datetime
import tempfile
import os
from pathlib import Path

# TranslationRAG는 더 이상 직접 사용하지 않음
# from ..rag_systems.translation_rag import TranslationRAG 
from ..graphs.translation_graph import run_translation_graph
from ..utils.llm_manager import LLMManager
from ..utils.document_processor import DocumentProcessor # 직접 사용
from ..config import (
    SUPPORTED_SOURCE_LANGUAGES, 
    SUPPORTED_TARGET_LANGUAGES,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    SUPPORTED_TRANSLATION_FILE_TYPES
)


class TranslationUI:
    """UI class for document translation."""
    
    # document_processor를 클래스 변수로 초기화
    document_processor = DocumentProcessor()

    @staticmethod
    def display_translation_tab():
        """Display the translation tab interface."""
        st.markdown("## 🌐 문서 번역")
        st.markdown("---")
        
        if not TranslationUI._check_llm_availability():
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            TranslationUI._display_upload_section()
        
        with col2:
            TranslationUI._display_settings_section()
        
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
                st.session_state.get("selected_llm_model", "gemma:7b"),
                st.session_state.get("ollama_base_url", "http://localhost:11434"),
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
    def _get_translation_system() -> Optional[Any]:
        # 이 메서드는 더 이상 필요하지 않지만, 다른 곳에서 호출할 경우를 대비해 None을 반환하도록 남겨둘 수 있습니다.
        # 혹은 관련 호출부를 모두 제거했다면 이 메서드 자체를 삭제해도 됩니다.
        # 지금은 호출부가 없으므로 삭제하는 방향으로 진행하겠습니다.
        pass
    
    @staticmethod
    def _display_upload_section():
        """Display file upload and input section."""
        st.subheader("📁 문서 입력")
        
        document_title = st.text_input("문서 제목 (선택사항)", value="", placeholder="문서 제목을 입력하면 번역 품질이 향상됩니다")
        uploaded_file = st.file_uploader("번역할 문서를 업로드하세요", type=SUPPORTED_TRANSLATION_FILE_TYPES)
        manual_text = st.text_area("또는 직접 텍스트 입력:", height=150, placeholder="여기에 영어 텍스트를 입력하세요...")
        
        text_to_translate = None
        if uploaded_file is not None:
            with st.spinner("파일 처리 중..."):
                text_to_translate = TranslationUI._process_uploaded_file(uploaded_file)
                if text_to_translate:
                    st.success(f"✅ 파일 '{uploaded_file.name}' 처리 완료")
                    with st.expander("📄 추출된 텍스트 미리보기"):
                        st.text_area("", text_to_translate[:1000] + "...", height=100, disabled=True)
                else:
                    st.error("❌ 파일 처리에 실패했습니다.")
        elif manual_text.strip():
            text_to_translate = manual_text.strip()
        
        st.session_state.text_to_translate = text_to_translate
        st.session_state.document_title = document_title.strip() if document_title.strip() else None
        
        if text_to_translate:
            if st.button("🚀 번역 시작", type="primary", use_container_width=True):
                TranslationUI._perform_translation(text_to_translate)
    
    @staticmethod
    def _display_settings_section():
        """Display translation settings section."""
        st.subheader("⚙️ 번역 설정")
        
        # Create language mapping for display
        lang_display = {
            "English": "영어 (English)",
            "Korean": "한국어 (Korean)", 
            "Japanese": "일본어 (Japanese)",
            "Chinese": "중국어 (Chinese)",
            "French": "프랑스어 (French)",
            "German": "독일어 (German)",
            "Spanish": "스페인어 (Spanish)",
            "Italian": "이탈리아어 (Italian)",
            "Portuguese": "포르투갈어 (Portuguese)",
            "Dutch": "네덜란드어 (Dutch)",
            "Russian": "러시아어 (Russian)"
        }
        
        # Language selection
        col1, col2 = st.columns(2)
        
        with col1:
            source_lang = st.selectbox(
                "원본 언어",
                options=SUPPORTED_SOURCE_LANGUAGES,
                index=SUPPORTED_SOURCE_LANGUAGES.index(DEFAULT_SOURCE_LANGUAGE) if DEFAULT_SOURCE_LANGUAGE in SUPPORTED_SOURCE_LANGUAGES else 0,
                help="번역할 문서의 원본 언어를 선택하세요",
                format_func=lambda x: lang_display.get(x, x)
            )
        
        with col2:
            target_lang = st.selectbox(
                "대상 언어",
                options=SUPPORTED_TARGET_LANGUAGES,
                index=SUPPORTED_TARGET_LANGUAGES.index(DEFAULT_TARGET_LANGUAGE) if DEFAULT_TARGET_LANGUAGE in SUPPORTED_TARGET_LANGUAGES else 0,
                help="번역할 대상 언어를 선택하세요",
                format_func=lambda x: lang_display.get(x, x)
            )
        
        # Display selected languages clearly
        source_display = lang_display.get(source_lang, source_lang)
        target_display = lang_display.get(target_lang, target_lang)
        
        st.success(f"🔄 **{source_display}** → **{target_display}** 번역")
        
        # Translation mode selection
        st.markdown("**번역 방식:**")
        translation_mode = st.radio(
            "번역 방식을 선택하세요",
            options=["단락 기반 (추천)", "문장 기반 (레거시)"],
            index=0,
            help="단락 기반: 빠르고 자연스러운 번역, 문장 기반: 정교한 문장별 번역"
        )
        
        use_paragraph_mode = translation_mode == "단락 기반 (추천)"
        
        # Store settings in session state
        st.session_state.translation_source_lang = source_lang
        st.session_state.translation_target_lang = target_lang
        st.session_state.use_paragraph_mode = use_paragraph_mode
        
        # Translation options
        st.markdown("**번역 옵션:**")
        
        preserve_formatting = st.checkbox(
            "서식 보존",
            value=True,
            help="원본 문서의 줄 바꿈과 문단 구조를 유지합니다"
        )
        
        if use_paragraph_mode:
            show_markdown = st.checkbox(
                "마크다운 형식으로 정리",
                value=True,
                help="번역 결과를 마크다운 형식으로 정리하여 보기 좋게 표시합니다"
            )
            
            show_comparison = st.checkbox(
                "단락별 비교 표시",
                value=True,
                help="번역 결과에 원문과 번역문을 단락별로 비교 표시합니다"
            )
        else:
            show_markdown = False
            show_comparison = st.checkbox(
                "문장별 비교 표시",
                value=True,
                help="번역 결과에 원문과 번역문을 문장별로 비교 표시합니다"
            )
        
        # Store options in session state
        st.session_state.preserve_formatting = preserve_formatting
        st.session_state.show_markdown = show_markdown
        st.session_state.show_comparison = show_comparison
        
        # Translation info
        st.markdown("---")
        st.markdown("**ℹ️ 번역 정보:**")
        
        info_text = f"""
        - **원본 언어:** {source_display}
        - **대상 언어:** {target_display}
        - **번역 방식:** {translation_mode}
        - **LLM 모델:** {st.session_state.get('selected_llm_model', 'Unknown')}
        - **Temperature:** {st.session_state.get('llm_temperature', 0.1)}
        """
        
        if use_paragraph_mode:
            info_text += "\n        - **마크다운 정리:** " + ("활성화" if show_markdown else "비활성화")
        
        st.info(info_text)
    
    @staticmethod
    def _perform_translation(text: str):
        """Perform document translation using the graph."""
        source_lang = st.session_state.get("translation_source_lang", "English")
        target_lang = st.session_state.get("translation_target_lang", "Korean")
        use_paragraph_mode = st.session_state.get("use_paragraph_mode", True)
        document_title = st.session_state.get("document_title", None)
        llm_model = st.session_state.get("selected_llm_model", "gemma:7b")
        llm_temperature = st.session_state.get("llm_temperature", 0.1)
        
        mode_text = "단락 기반" if use_paragraph_mode else "문장 기반"
        
        with st.spinner(f"{source_lang}에서 {target_lang}으로 번역 중... ({mode_text})"):
            try:
                result = run_translation_graph(
                    text=text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    use_paragraph_mode=use_paragraph_mode,
                    document_title=document_title,
                    llm_model=llm_model,
                    temperature=llm_temperature
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
        
        # Language information display
        source_lang = result.get("source_language", "Unknown")
        target_lang = result.get("target_language", "Unknown") 
        translation_mode = result.get("translation_mode", "unknown")
        
        # Create language mapping for display
        lang_display = {
            "English": "영어 (English)",
            "Korean": "한국어 (Korean)", 
            "Japanese": "일본어 (Japanese)",
            "Chinese": "중국어 (Chinese)",
            "French": "프랑스어 (French)",
            "German": "독일어 (German)",
            "Spanish": "스페인어 (Spanish)",
            "Italian": "이탈리아어 (Italian)",
            "Portuguese": "포르투갈어 (Portuguese)",
            "Dutch": "네덜란드어 (Dutch)",
            "Russian": "러시아어 (Russian)"
        }
        
        source_display = lang_display.get(source_lang, source_lang)
        target_display = lang_display.get(target_lang, target_lang)
        mode_display = "단락 기반" if translation_mode == "paragraph" else "문장 기반"
        
        # Prominent language information
        st.info(f"""
        🌐 **번역 정보**
        - **원본 언어:** {source_display}
        - **대상 언어:** {target_display}  
        - **번역 방식:** {mode_display}
        - **문서 제목:** {result.get('document_title', '제목 없음')}
        """)
        
        st.markdown("---")
        
        # Translation statistics
        stats = TranslationUI._get_translation_stats(result)
        
        # Display stats based on translation mode
        if translation_mode == "paragraph":
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("총 단락 수", stats.get("total_paragraphs", 0))
            
            with col2:
                st.metric("번역 단락 수", stats.get("translated_paragraphs", 0))
            
            with col3:
                st.metric("원문 단어 수", stats.get("original_word_count", 0))
            
            with col4:
                st.metric("번역 단어 수", stats.get("translated_word_count", 0))
        else:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("총 문장 수", stats.get("total_sentences", 0))
            
            with col2:
                st.metric("번역 문장 수", stats.get("translated_sentences", 0))
            
            with col3:
                st.metric("원문 단어 수", stats.get("original_word_count", 0))
            
            with col4:
                st.metric("번역 단어 수", stats.get("translated_word_count", 0))
        
        # Display translation results
        markdown_content = result.get("markdown_content", "")
        show_markdown = st.session_state.get("show_markdown", True)
        
        if markdown_content and show_markdown:
            # Display markdown formatted result
            st.subheader(f"📄 번역된 문서 ({target_display}, 마크다운 형식)")
            st.markdown(markdown_content)
            
            # Also show raw translated text in expander
            with st.expander(f"📝 원본 번역 텍스트 보기 ({target_display})"):
                translated_text = result.get("translated_text", "")
                st.text_area(
                    f"번역 결과 ({target_display})",
                    translated_text,
                    height=300,
                    disabled=True
                )
        else:
            # Display raw translated text
            st.subheader(f"📄 번역된 문서 ({target_display})")
            translated_text = result.get("translated_text", "")
            st.text_area(
                f"번역 결과 ({target_display})",
                translated_text,
                height=300,
                disabled=True
            )
        
        # Download button
        TranslationUI._display_download_section(result)
        
        # Comparison section
        if st.session_state.get("show_comparison", True):
            if translation_mode == "paragraph":
                TranslationUI._display_paragraph_comparison(result)
            else:
                TranslationUI._display_sentence_comparison(result)
    
    @staticmethod
    def _get_translation_stats(result: Dict[str, Any]) -> Dict[str, Any]:
        """Get translation statistics.
        
        Args:
            result: Translation result dictionary
            
        Returns:
            Statistics dictionary
        """
        if not result.get("success", False): return {}
        
        mode = result.get('translation_mode', 'unknown')
        pairs_key = 'paragraph_pairs' if mode == 'paragraph' else 'sentence_pairs'
        pairs = result.get(pairs_key, [])
        total = len(pairs)
        skipped = sum(1 for p in pairs if p.get('skipped'))
        
        stats = {
            "translation_mode": "단락 단위" if mode == "paragraph" else "문장 단위",
            "total_units": total,
            "translated_units": total - skipped,
            "original_char_count": len(result.get('original_text', '')),
            "translated_char_count": len(result.get('translated_text', '')),
            "original_word_count": len(result.get('original_text', '').split()),
            "translated_word_count": len(result.get('translated_text', '').split()),
            "has_markdown": bool(result.get('markdown_content'))
        }
        if mode == 'paragraph':
            stats["total_paragraphs"] = total
            stats["translated_paragraphs"] = total - skipped
        else:
            stats["total_sentences"] = total
            stats["translated_sentences"] = total - skipped
            
        return stats
    
    @staticmethod
    def _display_download_section(result: Dict[str, Any]):
        """Display download options.
        
        Args:
            result: Translation result dictionary
        """
        st.subheader("💾 다운로드 옵션")
        
        source_lang = result.get('source_language', 'Unknown')
        target_lang = result.get('target_language', 'Unknown')
        lang_codes = {"English": "en", "Korean": "ko", "Japanese": "ja", "Chinese": "zh"}
        source_code = lang_codes.get(source_lang, 'unk')
        target_code = lang_codes.get(target_lang, 'unk')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        cols = st.columns(3)
        with cols[0]:
            st.download_button(label="📄 번역문 다운로드 (.txt)", data=result.get("translated_text", ""),
                               file_name=f"translated_{source_code}_{target_code}_{timestamp}.txt", mime="text/plain", use_container_width=True)
        
        with cols[1]:
            if result.get("markdown_content"):
                st.download_button(label="📝 마크다운 다운로드 (.md)", data=result.get("markdown_content", ""),
                                   file_name=f"markdown_{source_code}_{target_code}_{timestamp}.md", mime="text/markdown", use_container_width=True)
        
        with cols[2]:
            report_text = TranslationUI._create_export_report(result)
            st.download_button(label="📊 상세 리포트 (.md)", data=report_text,
                               file_name=f"report_{source_code}_{target_code}_{timestamp}.md", mime="text/markdown", use_container_width=True)
    
    @staticmethod
    def _create_export_report(result: Dict[str, Any]) -> str:
        """Create a detailed report string for export."""
        if not result.get("success", False): return "번역 결과를 내보낼 수 없습니다."
        
        lang_display = {"English": "영어", "Korean": "한국어"}
        source_display = lang_display.get(result.get('source_language'), result.get('source_language'))
        target_display = lang_display.get(result.get('target_language'), result.get('target_language'))
        mode_display = "단락 기반" if result.get('translation_mode') == "paragraph" else "문장 기반"
        
        return f"""# 문서 번역 결과
## 번역 정보
- **원본 언어:** {source_display}
- **대상 언어:** {target_display}
- **번역 방식:** {mode_display}
- **문서 제목:** {result.get('document_title', '제목 없음')}
- **번역 시간:** {result.get('timestamp', 'Unknown')}
---
{result.get('markdown_content', result.get('translated_text', ''))}
"""
    
    @staticmethod
    def _display_paragraph_comparison(result: Dict[str, Any]):
        """Display paragraph-by-paragraph comparison.
        
        Args:
            result: Translation result dictionary
        """
        st.subheader("🔍 단락별 번역 비교")
        
        paragraph_pairs = result.get("paragraph_pairs", [])
        
        if not paragraph_pairs:
            st.warning("단락별 비교 데이터가 없습니다.")
            return
        
        # Filter out skipped paragraphs
        filtered_pairs = [pair for pair in paragraph_pairs if not pair.get("skipped", False)]
        
        if not filtered_pairs:
            st.warning("번역된 단락이 없습니다.")
            return
        
        # Pagination for large documents
        items_per_page = 5  # Show fewer paragraphs per page since they're longer
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
            with st.expander(f"단락 {i}: {pair.get('original', '')[:80]}..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**원문:**")
                    st.text_area(
                        "원문",
                        pair.get("original", ""),
                        height=200,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"original_para_{i}"
                    )
                
                with col2:
                    st.markdown("**번역:**")
                    st.text_area(
                        "번역",
                        pair.get("translated", ""),
                        height=200,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"translated_para_{i}"
                    )
        
        # Clear results button
        st.markdown("---")
        if st.button("🗑️ 번역 결과 지우기", type="secondary"):
            if "translation_result" in st.session_state:
                del st.session_state.translation_result
            st.rerun()
    
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
                        key=f"original_sent_{i}"
                    )
                
                with col2:
                    st.markdown("**번역:**")
                    st.text_area(
                        "번역",
                        pair.get("translated", ""),
                        height=100,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"translated_sent_{i}"
                    )
        
        # Clear results button
        st.markdown("---")
        if st.button("🗑️ 번역 결과 지우기", type="secondary"):
            if "translation_result" in st.session_state:
                del st.session_state.translation_result
            st.rerun()

    @staticmethod
    def _process_uploaded_file(uploaded_file) -> Optional[str]:
        """Process uploaded file and extract text."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            documents = TranslationUI.document_processor.load_documents([tmp_path])
            os.unlink(tmp_path)
            
            if documents:
                return '\\n\\n'.join([doc.page_content for doc in documents])
            return None
        except Exception as e:
            st.error(f"파일 처리 오류: {str(e)}")
            return None 