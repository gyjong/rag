"""LLM management utilities for the RAG application."""

from typing import Optional, Dict, Any, List, Iterator
import streamlit as st
import requests
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage


class LLMManager:
    """Manages LLM interactions for the RAG systems."""
    
    def __init__(self, model_name: str = "gemma3:12b-it-qat", base_url: str = "http://localhost:11434", temperature: float = 0.1):
        """Initialize the LLM manager.
        
        Args:
            model_name: Name of the Ollama model
            base_url: Base URL for Ollama server
            temperature: LLM temperature (creativity)
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self._llm = None
        
    def check_ollama_connection(self) -> bool:
        """Check if Ollama server is running.
        
        Returns:
            True if server is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def check_model_availability(self) -> bool:
        """Check if the specified model is available.
        
        Returns:
            True if model is available, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                return self.model_name in available_models
            return False
        except Exception:
            return False
    
    def get_llm(self) -> Optional[BaseChatModel]:
        """Get the LLM instance.
        
        Returns:
            ChatLLM instance or None if not available
        """
        if not self.check_ollama_connection():
            st.error(f"Ollama 서버에 연결할 수 없습니다: {self.base_url}")
            return None
            
        if not self.check_model_availability():
            st.error(f"모델 '{self.model_name}'을 찾을 수 없습니다. 먼저 모델을 다운로드해주세요.")
            st.code(f"ollama pull {self.model_name}")
            return None
            
        if self._llm is None:
            try:
                self._llm = ChatOllama(
                    model=self.model_name,
                    base_url=self.base_url,
                    temperature=self.temperature,
                    streaming=True  # Enable streaming by default
                )
                st.success(f"LLM 모델 '{self.model_name}' 로딩 완료")
            except Exception as e:
                st.error(f"LLM 초기화 실패: {str(e)}")
                return None
                
        return self._llm
    
    def generate_response(self, prompt: str, context: str = "", **kwargs) -> str:
        """Generate a response using the LLM.
        
        Args:
            prompt: User prompt/question
            context: Context information for RAG
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        llm = self.get_llm()
        if llm is None:
            return "LLM을 사용할 수 없습니다."
            
        try:
            if context:
                messages = [
                    SystemMessage(content=f"다음 컨텍스트를 바탕으로 질문에 답해주세요:\n{context}"),
                    HumanMessage(content=prompt)
                ]
            else:
                messages = [HumanMessage(content=prompt)]
                
            response = llm.invoke(messages)
            return response.content
            
        except Exception as e:
            st.error(f"응답 생성 실패: {str(e)}")
            return f"오류가 발생했습니다: {str(e)}"
    
    def generate_response_stream(self, prompt: str, context: str = "", **kwargs) -> Iterator[str]:
        """Generate a streaming response using the LLM.
        
        Args:
            prompt: User prompt/question
            context: Context information for RAG
            **kwargs: Additional generation parameters
            
        Yields:
            Response chunks
        """
        llm = self.get_llm()
        if llm is None:
            yield "LLM을 사용할 수 없습니다."
            return
            
        try:
            if context:
                messages = [
                    SystemMessage(content=f"다음 컨텍스트를 바탕으로 질문에 답해주세요:\n{context}"),
                    HumanMessage(content=prompt)
                ]
            else:
                messages = [HumanMessage(content=prompt)]
                
            for chunk in llm.stream(messages):
                if chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            yield f"오류가 발생했습니다: {str(e)}"
    
    def create_rag_chain(self, prompt_template: Optional[str] = None):
        """Create a RAG chain with the LLM.
        
        Args:
            prompt_template: Custom prompt template
            
        Returns:
            RAG chain or None if LLM not available
        """
        llm = self.get_llm()
        if llm is None:
            return None
            
        if prompt_template is None:
            prompt_template = """다음 컨텍스트를 바탕으로 질문에 답해주세요. 
컨텍스트에서 답을 찾을 수 없다면, 모른다고 답해주세요.

컨텍스트:
{context}

질문: {question}

답변:"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "다음 컨텍스트를 바탕으로 질문에 답해주세요. 컨텍스트에서 답을 찾을 수 없다면, 모른다고 답해주세요."),
            ("human", "컨텍스트: {context}\n\n질문: {question}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        return chain
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "connection_status": self.check_ollama_connection(),
            "model_available": self.check_model_availability()
        }
        
        if info["connection_status"]:
            try:
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    info["available_models"] = [model["name"] for model in models]
            except Exception:
                pass
                
        return info
    
    def pull_model(self) -> bool:
        """Pull the model if not available.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                timeout=300  # 5 minutes timeout
            )
            return response.status_code == 200
        except Exception as e:
            st.error(f"모델 다운로드 실패: {str(e)}")
            return False 