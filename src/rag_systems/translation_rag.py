"""Translation RAG system for translating documents from English to Korean."""

import streamlit as st
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import tempfile
import os
from pathlib import Path

from ..utils.llm_manager import LLMManager
from ..utils.document_processor import DocumentProcessor
from langchain_core.messages import SystemMessage, HumanMessage


class TranslationRAG:
    """RAG system specialized for document translation."""
    
    def __init__(self, llm_manager: LLMManager):
        """Initialize the Translation RAG system.
        
        Args:
            llm_manager: LLM manager instance
        """
        self.llm_manager = llm_manager
        self.document_processor = DocumentProcessor()
        
    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into logical paragraphs for translation.
        
        Args:
            text: Input text to split
            
        Returns:
            List of paragraphs
        """
        # Split by double newlines first (common paragraph separator)
        paragraphs = re.split(r'\n\s*\n', text.strip())
        
        # Further split very long paragraphs (over 1000 characters)
        final_paragraphs = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If paragraph is too long, split by sentence groups
            if len(paragraph) > 1000:
                # Split into sentence groups of about 500-800 characters
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) > 800 and current_chunk:
                        final_paragraphs.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk += (" " if current_chunk else "") + sentence
                
                if current_chunk.strip():
                    final_paragraphs.append(current_chunk.strip())
            else:
                final_paragraphs.append(paragraph)
        
        return final_paragraphs
    
    def translate_paragraph(self, paragraph: str, source_lang: str = "English", 
                           target_lang: str = "Korean", context: str = "") -> str:
        """Translate a paragraph with context awareness.
        
        Args:
            paragraph: Paragraph to translate
            source_lang: Source language
            target_lang: Target language
            context: Additional context for translation
            
        Returns:
            Translated paragraph
        """
        context_prompt = f"\n\nContext: {context}" if context else ""
        
        # Create explicit language mapping for better prompt clarity
        lang_map = {
            "English": "영어",
            "Korean": "한국어", 
            "Japanese": "일본어",
            "Chinese": "중국어",
            "French": "프랑스어",
            "German": "독일어",
            "Spanish": "스페인어",
            "Italian": "이탈리아어",
            "Portuguese": "포르투갈어",
            "Dutch": "네덜란드어",
            "Russian": "러시아어"
        }
        
        source_native = lang_map.get(source_lang, source_lang)
        target_native = lang_map.get(target_lang, target_lang)
        
        # Special handling for Korean translation
        korean_emphasis = ""
        if target_lang == "Korean":
            korean_emphasis = """
**중요한 한국어 번역 지침:**
- 반드시 한국어로만 번역하세요
- 영어나 다른 언어를 섞지 마세요
- 자연스러운 한국어 문체를 사용하세요
- 존댓말보다는 평어체를 사용하세요
- 한국어 맞춤법과 띄어쓰기를 정확히 지켜주세요
"""
        
        prompt = f"""You are a professional translator. Please translate the following text from {source_lang} to {target_lang}.

IMPORTANT TRANSLATION REQUIREMENTS:
- Source language: {source_lang} ({source_native})
- Target language: {target_lang} ({target_native})
- Translate ONLY to {target_lang}
- Do NOT include the original text
- Do NOT add explanations or comments
- Maintain the original meaning, tone, and style
- Ensure natural and fluent {target_lang} expression
- Keep technical terminology accurate
- Preserve formatting elements like lists, numbers, and special characters
- Maintain paragraph structure{korean_emphasis}{context_prompt}

Text to translate:
{paragraph}

Translated text in {target_lang}:"""
        
        try:
            translation = self.llm_manager.generate_response(prompt)
            return translation.strip()
        except Exception as e:
            st.error(f"번역 오류 (단락: {paragraph[:50]}...): {str(e)}")
            return f"[번역 오류: {paragraph}]"
    
    def organize_and_format_translation(self, translated_text: str, 
                                      original_title: str = None,
                                      target_lang: str = "Korean") -> str:
        """Organize translated content and format as markdown.
        
        Args:
            translated_text: Translated content
            original_title: Original document title
            target_lang: Target language
            
        Returns:
            Markdown formatted translation
        """
        # Create language mapping for better prompt clarity
        lang_map = {
            "English": "영어",
            "Korean": "한국어", 
            "Japanese": "일본어",
            "Chinese": "중국어",
            "French": "프랑스어",
            "German": "독일어",
            "Spanish": "스페인어",
            "Italian": "이탈리아어",
            "Portuguese": "포르투갈어",
            "Dutch": "네덜란드어",
            "Russian": "러시아어"
        }
        
        target_native = lang_map.get(target_lang, target_lang)
        
        # Special handling for Korean formatting
        korean_emphasis = ""
        if target_lang == "Korean":
            korean_emphasis = """
**한국어 정리 지침:**
- 모든 내용을 한국어로 유지하세요
- 한국어 맞춤법과 띄어쓰기를 정확히 지켜주세요
- 자연스러운 한국어 표현을 사용하세요
- 제목과 소제목도 한국어로 작성하세요
"""
        
        prompt = f"""Please organize and format the following {target_lang} ({target_native}) translated text into a well-structured markdown document.

IMPORTANT FORMATTING REQUIREMENTS:
- The text is already translated to {target_lang}
- Create appropriate headings (# ## ###) based on content structure
- Add proper paragraph breaks for readability
- Format lists, quotes, and code blocks appropriately
- Ensure logical flow and coherent organization
- Add a title if not present
- Maintain all original content while improving structure
- Keep all text in {target_lang}{korean_emphasis}

Original title: {original_title or '문서'}

{target_lang} text to organize:
{translated_text}

Please provide the organized markdown format in {target_lang}:"""
        
        try:
            organized_content = self.llm_manager.generate_response(prompt)
            return organized_content.strip()
        except Exception as e:
            st.error(f"내용 정리 오류: {str(e)}")
            # Fallback: simple markdown formatting
            lines = translated_text.split('\n')
            markdown_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    markdown_lines.append('')
                    continue
                    
                # Simple heuristic for headers
                if len(line) < 50 and not line.endswith('.') and not line.endswith(','):
                    markdown_lines.append(f"## {line}")
                else:
                    markdown_lines.append(line)
            
            return '\n'.join(markdown_lines)
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for translation.
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting using regex
        # This handles common sentence endings with periods, exclamation marks, and question marks
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
        
        # Clean up sentences and remove empty ones
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Further split by newlines for better handling of lists and paragraphs
        final_sentences = []
        for sentence in sentences:
            if '\n' in sentence:
                # Split by newlines but keep the structure
                parts = sentence.split('\n')
                for part in parts:
                    if part.strip():
                        final_sentences.append(part.strip())
            else:
                final_sentences.append(sentence)
        
        return final_sentences
    
    def translate_sentence(self, sentence: str, source_lang: str = "English", 
                          target_lang: str = "Korean") -> str:
        """Translate a single sentence.
        
        Args:
            sentence: Sentence to translate
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Translated sentence
        """
        # Create explicit language mapping for better prompt clarity
        lang_map = {
            "English": "영어",
            "Korean": "한국어", 
            "Japanese": "일본어",
            "Chinese": "중국어",
            "French": "프랑스어",
            "German": "독일어",
            "Spanish": "스페인어",
            "Italian": "이탈리아어",
            "Portuguese": "포르투갈어",
            "Dutch": "네덜란드어",
            "Russian": "러시아어"
        }
        
        source_native = lang_map.get(source_lang, source_lang)
        target_native = lang_map.get(target_lang, target_lang)
        
        # Special handling for Korean translation
        korean_emphasis = ""
        if target_lang == "Korean":
            korean_emphasis = """
**중요한 한국어 번역 지침:**
- 반드시 한국어로만 번역하세요
- 영어나 다른 언어를 섞지 마세요
- 자연스러운 한국어 문체를 사용하세요
- 존댓말보다는 평어체를 사용하세요
- 한국어 맞춤법과 띄어쓰기를 정확히 지켜주세요
"""
        
        prompt = f"""You are a professional translator. Please translate the following text from {source_lang} to {target_lang}.

IMPORTANT TRANSLATION REQUIREMENTS:
- Source language: {source_lang} ({source_native})
- Target language: {target_lang} ({target_native})
- Translate ONLY to {target_lang}
- Do NOT include the original text
- Do NOT add explanations or comments
- Keep the original meaning and tone
- If the text contains technical terminology, maintain accuracy{korean_emphasis}

Text to translate: {sentence}

Translated text in {target_lang}:"""
        
        try:
            translation = self.llm_manager.generate_response(prompt)
            return translation.strip()
        except Exception as e:
            st.error(f"번역 오류 (문장: {sentence[:50]}...): {str(e)}")
            return f"[번역 오류: {sentence}]"
    
    def translate_document(self, text: str, source_lang: str = "English", 
                          target_lang: str = "Korean", 
                          use_paragraph_mode: bool = True,
                          document_title: str = None,
                          progress_callback=None) -> Dict[str, Any]:
        """Translate an entire document.
        
        Args:
            text: Document text to translate
            source_lang: Source language
            target_lang: Target language
            use_paragraph_mode: Whether to use paragraph-based translation
            document_title: Title of the document
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing translation results
        """
        if use_paragraph_mode:
            return self._translate_document_by_paragraphs(
                text, source_lang, target_lang, document_title, progress_callback
            )
        else:
            return self._translate_document_by_sentences(
                text, source_lang, target_lang, progress_callback
            )
    
    def _translate_document_by_paragraphs(self, text: str, source_lang: str, 
                                        target_lang: str, document_title: str,
                                        progress_callback=None) -> Dict[str, Any]:
        """Translate document using paragraph-based approach.
        
        Args:
            text: Document text to translate
            source_lang: Source language
            target_lang: Target language
            document_title: Title of the document
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing translation results
        """
        # Split document into paragraphs
        paragraphs = self.split_into_paragraphs(text)
        
        if not paragraphs:
            return {
                "success": False,
                "error": "문서에서 번역할 내용을 찾을 수 없습니다.",
                "original_text": text,
                "translated_text": "",
                "markdown_content": ""
            }
        
        st.info(f"📄 총 {len(paragraphs)}개 단락을 번역합니다...")
        
        translations = []
        translated_paragraphs = []
        
        # Create progress bar
        if progress_callback is None:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Generate context for better translation
        document_context = f"문서 제목: {document_title}" if document_title else ""
        if len(text) > 500:
            # Create context from first paragraph
            first_para = paragraphs[0][:200] + "..." if len(paragraphs[0]) > 200 else paragraphs[0]
            document_context += f"\n문서 개요: {first_para}"
        
        # Translate each paragraph
        for i, paragraph in enumerate(paragraphs):
            if progress_callback:
                progress_callback(i, len(paragraphs), paragraph)
            else:
                progress = (i + 1) / len(paragraphs)
                progress_bar.progress(progress)
                status_text.text(f"번역 중... ({i+1}/{len(paragraphs)}): {paragraph[:50]}...")
            
            # Skip very short paragraphs
            if len(paragraph.strip()) <= 10:
                translations.append(paragraph)
                translated_paragraphs.append({
                    "original": paragraph,
                    "translated": paragraph,
                    "skipped": True
                })
                continue
                
            translation = self.translate_paragraph(
                paragraph, source_lang, target_lang, document_context
            )
            translations.append(translation)
            translated_paragraphs.append({
                "original": paragraph,
                "translated": translation,
                "skipped": False
            })
        
        # Complete progress bar
        if progress_callback is None:
            progress_bar.progress(1.0)
            status_text.text("번역 완료! 내용을 정리하고 있습니다...")
        
        # Combine translations
        translated_text = '\n\n'.join(translations)
        
        # Organize and format as markdown
        try:
            markdown_content = self.organize_and_format_translation(
                translated_text, document_title, target_lang
            )
        except Exception as e:
            st.warning(f"마크다운 정리 중 오류 발생: {str(e)}")
            markdown_content = translated_text
        
        if progress_callback is None:
            status_text.text("번역 및 정리 완료!")
        
        return {
            "success": True,
            "original_text": text,
            "translated_text": translated_text,
            "markdown_content": markdown_content,
            "source_language": source_lang,
            "target_language": target_lang,
            "paragraphs": paragraphs,
            "translations": translations,
            "paragraph_pairs": translated_paragraphs,
            "total_paragraphs": len(paragraphs),
            "translation_mode": "paragraph",
            "document_title": document_title,
            "timestamp": datetime.now().isoformat()
        }
    
    def _translate_document_by_sentences(self, text: str, source_lang: str, 
                                       target_lang: str, progress_callback=None) -> Dict[str, Any]:
        """Translate document using sentence-based approach (legacy).
        
        Args:
            text: Document text to translate
            source_lang: Source language
            target_lang: Target language
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing translation results
        """
        # Split document into sentences
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return {
                "success": False,
                "error": "문서에서 번역할 문장을 찾을 수 없습니다.",
                "original_text": text,
                "translated_text": "",
                "sentences": [],
                "translations": []
            }
        
        st.info(f"📝 총 {len(sentences)}개 문장을 번역합니다...")
        
        translations = []
        translated_sentences = []
        
        # Create progress bar
        if progress_callback is None:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Translate each sentence
        for i, sentence in enumerate(sentences):
            if progress_callback:
                progress_callback(i, len(sentences), sentence)
            else:
                progress = (i + 1) / len(sentences)
                progress_bar.progress(progress)
                status_text.text(f"번역 중... ({i+1}/{len(sentences)}): {sentence[:50]}...")
            
            # Skip very short sentences or single characters
            if len(sentence.strip()) <= 2:
                translations.append(sentence)
                translated_sentences.append({
                    "original": sentence,
                    "translated": sentence,
                    "skipped": True
                })
                continue
                
            translation = self.translate_sentence(sentence, source_lang, target_lang)
            translations.append(translation)
            translated_sentences.append({
                "original": sentence,
                "translated": translation,
                "skipped": False
            })
        
        # Complete progress bar
        if progress_callback is None:
            progress_bar.progress(1.0)
            status_text.text("번역 완료!")
        
        # Combine translations
        translated_text = '\n'.join(translations)
        
        return {
            "success": True,
            "original_text": text,
            "translated_text": translated_text,
            "source_language": source_lang,
            "target_language": target_lang,
            "sentences": sentences,
            "translations": translations,
            "sentence_pairs": translated_sentences,
            "total_sentences": len(sentences),
            "translation_mode": "sentence",
            "timestamp": datetime.now().isoformat()
        }
    
    def process_uploaded_file(self, uploaded_file) -> Optional[str]:
        """Process uploaded file and extract text.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Extracted text or None if failed
        """
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Process document
            documents = self.document_processor.load_documents([tmp_path])
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            if documents:
                # Combine all document content
                text = '\n\n'.join([doc.page_content for doc in documents])
                return text
            else:
                return None
                
        except Exception as e:
            st.error(f"파일 처리 오류: {str(e)}")
            return None
    
    def export_translation_result(self, result: Dict[str, Any]) -> str:
        """Export translation result as formatted text.
        
        Args:
            result: Translation result dictionary
            
        Returns:
            Formatted text for export
        """
        if not result.get("success", False):
            return "번역 결과를 내보낼 수 없습니다."
        
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
        
        source_lang = result.get('source_language', 'Unknown')
        target_lang = result.get('target_language', 'Unknown')
        source_display = lang_display.get(source_lang, source_lang)
        target_display = lang_display.get(target_lang, target_lang)
        translation_mode = result.get('translation_mode', 'Unknown')
        mode_display = "단락 기반" if translation_mode == "paragraph" else "문장 기반"
        
        # Use markdown content if available
        if result.get("markdown_content"):
            export_text = f"""# 문서 번역 결과

## 번역 정보
- **원본 언어:** {source_display}
- **대상 언어:** {target_display}
- **번역 방식:** {mode_display}
- **문서 제목:** {result.get('document_title', '제목 없음')}
- **번역 시간:** {result.get('timestamp', 'Unknown')}

---

{result.get('markdown_content', '')}
"""
        else:
            # Fallback to legacy format
            export_text = f"""# 문서 번역 결과

## 번역 정보
- **원본 언어:** {source_display}
- **대상 언어:** {target_display}
- **번역 방식:** {mode_display}
- **총 항목 수:** {result.get('total_sentences', result.get('total_paragraphs', 0))}
- **번역 시간:** {result.get('timestamp', 'Unknown')}

## 번역된 문서

{result.get('translated_text', '')}
"""
        
        return export_text
    
    def get_translation_stats(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about the translation.
        
        Args:
            result: Translation result dictionary
            
        Returns:
            Translation statistics
        """
        if not result.get("success", False):
            return {}
        
        translation_mode = result.get('translation_mode', 'unknown')
        
        if translation_mode == 'paragraph':
            paragraph_pairs = result.get('paragraph_pairs', [])
            total_units = len(paragraph_pairs)
            skipped_units = sum(1 for pair in paragraph_pairs if pair.get('skipped', False))
            translated_units = total_units - skipped_units
            
            stats = {
                "translation_mode": "단락 단위",
                "total_paragraphs": total_units,
                "translated_paragraphs": translated_units,
                "skipped_paragraphs": skipped_units,
            }
        else:
            sentence_pairs = result.get('sentence_pairs', [])
            total_units = len(sentence_pairs)
            skipped_units = sum(1 for pair in sentence_pairs if pair.get('skipped', False))
            translated_units = total_units - skipped_units
            
            stats = {
                "translation_mode": "문장 단위",
                "total_sentences": total_units,
                "translated_sentences": translated_units,
                "skipped_sentences": skipped_units,
            }
        
        original_text = result.get('original_text', '')
        translated_text = result.get('translated_text', '')
        
        stats.update({
            "original_char_count": len(original_text),
            "translated_char_count": len(translated_text),
            "original_word_count": len(original_text.split()),
            "translated_word_count": len(translated_text.split()),
            "source_language": result.get('source_language', 'Unknown'),
            "target_language": result.get('target_language', 'Unknown'),
            "has_markdown": bool(result.get('markdown_content'))
        })
        
        return stats 