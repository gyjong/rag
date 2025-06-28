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
        prompt = f"""Please translate the following {source_lang} text to {target_lang}.
Keep the original meaning and tone. If the text contains technical terminology, maintain accuracy.
Only provide the translation without additional explanations.

Text to translate: {sentence}

Translation:"""
        
        try:
            translation = self.llm_manager.generate_response(prompt)
            return translation.strip()
        except Exception as e:
            st.error(f"번역 오류 (문장: {sentence[:50]}...): {str(e)}")
            return f"[번역 오류: {sentence}]"
    
    def translate_document(self, text: str, source_lang: str = "English", 
                          target_lang: str = "Korean", 
                          progress_callback=None) -> Dict[str, Any]:
        """Translate an entire document.
        
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
        
        export_text = f"""# 문서 번역 결과

## 번역 정보
- 원본 언어: {result.get('source_language', 'Unknown')}
- 대상 언어: {result.get('target_language', 'Unknown')}
- 총 문장 수: {result.get('total_sentences', 0)}
- 번역 시간: {result.get('timestamp', 'Unknown')}

## 번역된 문서

{result.get('translated_text', '')}

---

## 문장별 번역 비교

"""
        
        # Add sentence-by-sentence comparison
        sentence_pairs = result.get('sentence_pairs', [])
        for i, pair in enumerate(sentence_pairs, 1):
            if pair.get('skipped', False):
                continue
                
            export_text += f"""### 문장 {i}

**원문:** {pair.get('original', '')}

**번역:** {pair.get('translated', '')}

---

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
        
        sentence_pairs = result.get('sentence_pairs', [])
        
        total_sentences = len(sentence_pairs)
        skipped_sentences = sum(1 for pair in sentence_pairs if pair.get('skipped', False))
        translated_sentences = total_sentences - skipped_sentences
        
        original_text = result.get('original_text', '')
        translated_text = result.get('translated_text', '')
        
        return {
            "total_sentences": total_sentences,
            "translated_sentences": translated_sentences,
            "skipped_sentences": skipped_sentences,
            "original_char_count": len(original_text),
            "translated_char_count": len(translated_text),
            "original_word_count": len(original_text.split()),
            "translated_word_count": len(translated_text.split()),
            "source_language": result.get('source_language', 'Unknown'),
            "target_language": result.get('target_language', 'Unknown')
        } 