"""Translation RAG system for translating documents from English to Korean."""

import re
from typing import List
from ..utils.llm_manager import LLMManager

def split_into_paragraphs(text: str) -> List[str]:
    """Split text into logical paragraphs for translation.
    
    Args:
        text: Input text to split
        
    Returns:
        List of paragraphs
    """
    paragraphs = re.split(r'\n\s*\n', text.strip())
    final_paragraphs = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > 1000:
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

def translate_paragraph(llm_manager: LLMManager, paragraph: str, source_lang: str, target_lang: str, context: str = "") -> str:
    """Translate a paragraph with context awareness.
    
    Args:
        llm_manager: LLM manager instance
        paragraph: Paragraph to translate
        source_lang: Source language
        target_lang: Target language
        context: Additional context for translation
        
    Returns:
        Translated paragraph
    """
    context_prompt = f"\n\nContext: {context}" if context else ""
    lang_map = {"English": "영어", "Korean": "한국어", "Japanese": "일본어", "Chinese": "중국어", "French": "프랑스어", "German": "독일어", "Spanish": "스페인어", "Italian": "이탈리아어", "Portuguese": "포르투갈어", "Dutch": "네덜란드어", "Russian": "러시아어"}
    source_native = lang_map.get(source_lang, source_lang)
    target_native = lang_map.get(target_lang, target_lang)
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
        translation = llm_manager.generate_response(prompt)
        return translation.strip()
    except Exception as e:
        print(f"Error during translation: {e}")
        return f"[번역 오류: {paragraph}]"

def organize_and_format_translation(llm_manager: LLMManager, translated_text: str, original_title: str, target_lang: str) -> str:
    """Organize translated content and format as markdown.
    
    Args:
        llm_manager: LLM manager instance
        translated_text: Translated content
        original_title: Original document title
        target_lang: Target language
        
    Returns:
        Markdown formatted translation
    """
    lang_map = {"English": "영어", "Korean": "한국어", "Japanese": "일본어", "Chinese": "중국어", "French": "프랑스어", "German": "독일어", "Spanish": "스페인어", "Italian": "이탈리아어", "Portuguese": "포르투갈어", "Dutch": "네덜란드어", "Russian": "러시아어"}
    target_native = lang_map.get(target_lang, target_lang)
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
        organized_content = llm_manager.generate_response(prompt)
        return organized_content.strip()
    except Exception:
        lines = translated_text.split('\n')
        markdown_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                markdown_lines.append('')
                continue
            if len(line) < 50 and not line.endswith('.') and not line.endswith(','):
                markdown_lines.append(f"## {line}")
            else:
                markdown_lines.append(line)
        return '\n'.join(markdown_lines)

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences for translation.
    
    Args:
        text: Input text to split
        
    Returns:
        List of sentences
    """
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    final_sentences = []
    for sentence in sentences:
        if '\n' in sentence:
            parts = sentence.split('\n')
            for part in parts:
                if part.strip():
                    final_sentences.append(part.strip())
        else:
            final_sentences.append(sentence)
    return final_sentences

def translate_sentence(llm_manager: LLMManager, sentence: str, source_lang: str, target_lang: str) -> str:
    """Translate a single sentence.
    
    Args:
        llm_manager: LLM manager instance
        sentence: Sentence to translate
        source_lang: Source language
        target_lang: Target language
            
    Returns:
        Translated sentence
    """
    lang_map = {"English": "영어", "Korean": "한국어", "Japanese": "일본어", "Chinese": "중국어", "French": "프랑스어", "German": "독일어", "Spanish": "스페인어", "Italian": "이탈리아어", "Portuguese": "포르투갈어", "Dutch": "네덜란드어", "Russian": "러시아어"}
    source_native = lang_map.get(source_lang, source_lang)
    target_native = lang_map.get(target_lang, target_lang)
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
        translation = llm_manager.generate_response(prompt)
        return translation.strip()
    except Exception as e:
        print(f"Error during translation: {e}")
        return f"[번역 오류: {sentence}]" 