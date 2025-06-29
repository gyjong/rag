"""
Document Discovery RAG System
문서 발견 및 상세 검색을 위한 2단계 RAG 시스템
"""

import streamlit as st
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.components.document_loader import DocumentLoader
from src.utils.llm_manager import LLMManager
from src.utils.vector_store import VectorStoreManager
from src.config import DOCS_FOLDER as DOCS_DIR


class DocumentDiscoveryRAG:
    """문서 발견 및 상세 검색을 위한 2단계 RAG 시스템"""
    
    def __init__(self, llm_manager: LLMManager, embedding_manager, vector_store_manager: VectorStoreManager):
        self.llm_manager = llm_manager
        self.embedding_manager = embedding_manager
        self.vector_store_manager = vector_store_manager
        self.document_loader = DocumentLoader()
        
        # 문서 요약 캐시 파일 경로
        self.cache_dir = Path("vector_stores/document_summaries")
        self.cache_dir.mkdir(exist_ok=True)
        self.summary_cache_file = self.cache_dir / "document_summaries.json"
        
        # 프롬프트 템플릿 정의
        self.summary_prompt = PromptTemplate(
            input_variables=["document_content", "document_name"],
            template="""다음 문서의 내용을 분석하여 요약해주세요:

문서명: {document_name}

문서 내용:
{document_content}

다음 형식으로 요약해주세요:
1. 주요 주제: (문서의 핵심 주제)
2. 키워드: (중요한 키워드들을 쉼표로 구분)
3. 내용 요약: (3-5문장으로 핵심 내용 요약)
4. 문서 유형: (보고서/논문/정책문서/기술문서 등)
5. 대상 독자: (일반인/전문가/정책결정자 등)

요약:"""
        )
        
        self.relevance_prompt = PromptTemplate(
            input_variables=["query", "document_summaries"],
            template="""사용자 질문: {query}

다음은 사용 가능한 문서들의 요약입니다:
{document_summaries}

사용자 질문과 각 문서의 관련성을 0-100점으로 평가하고, 관련성이 높은 순서대로 정렬해주세요.

다음 형식으로 답변해주세요:
[문서명] 점수: XX점, 관련성 설명: (왜 관련이 있는지 간단히 설명)

예시:
[문서1.pdf] 점수: 85점, 관련성 설명: 사용자가 묻는 AI 정책과 직접적으로 관련된 정부 정책 문서
[문서2.pdf] 점수: 70점, 관련성 설명: AI 기술 동향을 다루어 질문과 부분적으로 관련"""
        )
        
        self.detail_qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""다음 문서 내용을 바탕으로 질문에 답변해주세요:

문서 내용:
{context}

질문: {question}

답변 시 다음 사항을 고려해주세요:
1. 문서에서 직접적으로 언급된 내용을 우선적으로 활용
2. 구체적인 수치나 데이터가 있으면 포함
3. 관련된 맥락이나 배경 정보도 함께 제공
4. 문서에 없는 내용은 추측하지 말고 "문서에 명시되지 않음"이라고 명시

답변:"""
        )
    
    def get_available_documents(self) -> List[Dict[str, Any]]:
        """사용 가능한 문서 목록과 기본 정보 반환"""
        documents = []
        pdf_files = list(DOCS_DIR.glob("*.pdf"))
        
        for pdf_file in pdf_files:
            doc_info = self.document_loader.get_document_info(pdf_file.name)
            if doc_info:
                documents.append(doc_info)
        
        return documents
    
    def load_document_summaries(self) -> Dict[str, Dict[str, Any]]:
        """캐시된 문서 요약 로드"""
        if self.summary_cache_file.exists():
            try:
                with open(self.summary_cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_document_summaries(self, summaries: Dict[str, Dict[str, Any]]):
        """문서 요약을 캐시에 저장"""
        try:
            with open(self.summary_cache_file, 'w', encoding='utf-8') as f:
                json.dump(summaries, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"요약 캐시 저장 실패: {str(e)}")
    
    def generate_document_summary(self, filename: str) -> Optional[Dict[str, Any]]:
        """단일 문서 요약 생성"""
        try:
            # 문서 로드
            documents = self.document_loader.load_document(filename)
            if not documents:
                return None
            
            # 문서 내용 합치기 (처음 3페이지 정도만 사용)
            content_parts = []
            for i, doc in enumerate(documents[:3]):  # 처음 3페이지만
                content_parts.append(doc.page_content)
            
            combined_content = "\n\n".join(content_parts)
            
            # 너무 긴 경우 잘라내기 (요약 생성을 위해)
            if len(combined_content) > 4000:
                combined_content = combined_content[:4000] + "..."
            
            # LLM을 사용해 요약 생성
            llm = self.llm_manager.get_llm()
            chain = self.summary_prompt | llm | StrOutputParser()
            
            summary_text = chain.invoke({
                "document_content": combined_content,
                "document_name": filename
            })
            
            # 문서 기본 정보
            doc_info = self.document_loader.get_document_info(filename)
            
            summary_data = {
                "filename": filename,
                "summary": summary_text,
                "generated_at": datetime.now().isoformat(),
                "pages": doc_info.get("pages", 0),
                "size_mb": doc_info.get("size_mb", 0),
                "preview": doc_info.get("preview", ""),
                "title": doc_info.get("title", filename)
            }
            
            return summary_data
            
        except Exception as e:
            st.error(f"문서 요약 생성 실패 ({filename}): {str(e)}")
            return None
    
    def generate_all_summaries(self, progress_callback=None) -> Dict[str, Dict[str, Any]]:
        """모든 문서의 요약 생성"""
        summaries = self.load_document_summaries()
        available_docs = self.get_available_documents()
        
        total_docs = len(available_docs)
        processed = 0
        
        for doc_info in available_docs:
            filename = doc_info["filename"]
            
            # 이미 요약이 존재하는지 확인
            if filename in summaries:
                processed += 1
                if progress_callback:
                    progress_callback(processed, total_docs, f"✅ {filename} (캐시됨)")
                continue
            
            if progress_callback:
                progress_callback(processed, total_docs, f"📝 {filename} 요약 생성 중...")
            
            # 새 요약 생성
            summary = self.generate_document_summary(filename)
            if summary:
                summaries[filename] = summary
                # 진행 중에도 저장 (중단되어도 진행사항 보존)
                self.save_document_summaries(summaries)
            
            processed += 1
            if progress_callback:
                status = "✅ 완료" if summary else "❌ 실패"
                progress_callback(processed, total_docs, f"{status} {filename}")
        
        return summaries
    
    def find_relevant_documents(self, query: str, top_k: int = 5) -> List[Tuple[str, int, str]]:
        """사용자 질문과 관련된 문서들을 찾기"""
        summaries = self.load_document_summaries()
        
        if not summaries:
            return []
        
        # 문서 요약들을 텍스트로 변환
        summary_text_parts = []
        for filename, summary_data in summaries.items():
            summary_text_parts.append(f"[{filename}]\n{summary_data['summary']}\n")
        
        combined_summaries = "\n".join(summary_text_parts)
        
        try:
            # LLM을 사용해 관련성 평가
            llm = self.llm_manager.get_llm()
            chain = self.relevance_prompt | llm | StrOutputParser()
            
            relevance_result = chain.invoke({
                "query": query,
                "document_summaries": combined_summaries
            })
            
            # 결과 파싱
            relevant_docs = self._parse_relevance_result(relevance_result)
            
            # 상위 k개만 반환
            return relevant_docs[:top_k]
            
        except Exception as e:
            st.error(f"관련성 평가 실패: {str(e)}")
            return []
    
    def _parse_relevance_result(self, result_text: str) -> List[Tuple[str, int, str]]:
        """관련성 평가 결과 파싱"""
        relevant_docs = []
        lines = result_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if '[' in line and ']' in line and '점수:' in line:
                try:
                    # [문서명] 추출
                    start_bracket = line.find('[')
                    end_bracket = line.find(']')
                    if start_bracket >= 0 and end_bracket > start_bracket:
                        filename = line[start_bracket+1:end_bracket]
                        
                        # 점수 추출
                        score_start = line.find('점수:') + 3
                        score_end = line.find('점', score_start)
                        if score_start > 2 and score_end > score_start:
                            score_str = line[score_start:score_end].strip()
                            score = int(score_str)
                            
                            # 설명 추출
                            explanation_start = line.find('관련성 설명:')
                            explanation = ""
                            if explanation_start >= 0:
                                explanation = line[explanation_start+7:].strip()
                            
                            relevant_docs.append((filename, score, explanation))
                
                except Exception:
                    continue
        
        # 점수 순으로 정렬
        relevant_docs.sort(key=lambda x: x[1], reverse=True)
        return relevant_docs
    
    def detailed_search(self, filename: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        """선택된 문서에 대한 상세 검색"""
        try:
            # 문서 로드 및 청크 생성
            documents = self.document_loader.process_documents_for_rag([filename])
            
            if not documents:
                return {"error": "문서를 로드할 수 없습니다."}
            
            # 임시 벡터 스토어 생성 (해당 문서만)
            embeddings = self.embedding_manager.get_embeddings()
            
            # 새로운 임시 VectorStoreManager 인스턴스 생성
            from src.utils.vector_store import VectorStoreManager
            temp_collection_name = f"temp_{filename.replace('.', '_').replace(' ', '_')}"
            temp_vector_store_manager = VectorStoreManager(
                embeddings=embeddings,
                vector_store_type=self.vector_store_manager.vector_store_type,
                collection_name=temp_collection_name
            )
            
            # 임시 벡터 스토어 생성
            temp_vector_store = temp_vector_store_manager.create_vector_store(documents)
            
            # 관련 청크 검색
            retriever = temp_vector_store.as_retriever(search_kwargs={"k": top_k})
            relevant_chunks = retriever.get_relevant_documents(query)
            
            # 컨텍스트 구성
            context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
            
            # LLM을 사용해 답변 생성  
            llm = self.llm_manager.get_llm()
            chain = self.detail_qa_prompt | llm | StrOutputParser()
            
            answer = chain.invoke({
                "context": context,
                "question": query
            })
            
            # 임시 벡터 스토어 정리
            try:
                temp_vector_store_manager.cleanup()
            except:
                pass  # 정리 실패는 무시
            
            return {
                "filename": filename,
                "answer": answer,
                "relevant_chunks": [
                    {
                        "content": chunk.page_content,
                        "metadata": chunk.metadata
                    }
                    for chunk in relevant_chunks
                ],
                "total_chunks_found": len(relevant_chunks)
            }
            
        except Exception as e:
            return {"error": f"상세 검색 실패: {str(e)}"}
    
    def get_document_overview(self, filename: str) -> Dict[str, Any]:
        """문서 개요 정보 반환"""
        summaries = self.load_document_summaries()
        
        if filename in summaries:
            return summaries[filename]
        else:
            # 캐시에 없으면 새로 생성
            return self.generate_document_summary(filename) 