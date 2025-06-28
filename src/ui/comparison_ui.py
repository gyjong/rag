"""UI components for RAG system comparison."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any
import time


class ComparisonUI:
    """UI components for comparing RAG systems."""
    
    @staticmethod
    def display_system_comparison(systems_info: List[Dict[str, Any]]):
        """Display comparison of RAG systems."""
        st.subheader("🔍 RAG 시스템 비교")
        
        # Create comparison table
        comparison_data = []
        for system in systems_info:
            comparison_data.append({
                "시스템": system["name"],
                "설명": system["description"],
                "주요 특징": ", ".join(system["features"][:3]),
                "구성 요소 수": len(system["components"])
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Display detailed features
        cols = st.columns(len(systems_info))
        for i, (col, system) in enumerate(zip(cols, systems_info)):
            with col:
                st.write(f"**{system['name']}**")
                st.write("**구성 요소:**")
                for component in system["components"]:
                    st.write(f"• {component}")
                
                if "advantages" in system:
                    st.write("**장점:**")
                    for advantage in system["advantages"]:
                        st.write(f"✓ {advantage}")
    
    @staticmethod
    def display_performance_comparison(results: List[Dict[str, Any]]):
        """Display performance comparison of RAG results."""
        if not results:
            return
            
        st.subheader("⚡ 성능 비교")
        
        # Create performance metrics
        metrics_data = []
        for result in results:
            metadata = result.get("metadata", {})
            metrics_data.append({
                "시스템": result["rag_type"],
                "처리 시간 (초)": round(result["total_time"], 2),
                "검색 문서 수": metadata.get("num_retrieved", metadata.get("total_retrieved", 0)),
                "검색 방법": metadata.get("retrieval_method", "N/A"),
                "생성 방법": metadata.get("generation_method", "N/A")
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Display metrics table
        st.dataframe(df_metrics, use_container_width=True)
        
        # Create performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Processing time chart
            fig_time = px.bar(
                df_metrics, 
                x="시스템", 
                y="처리 시간 (초)",
                title="처리 시간 비교",
                color="시스템"
            )
            fig_time.update_layout(showlegend=False)
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            # Retrieved documents chart
            fig_docs = px.bar(
                df_metrics, 
                x="시스템", 
                y="검색 문서 수",
                title="검색 문서 수 비교",
                color="시스템"
            )
            fig_docs.update_layout(showlegend=False)
            st.plotly_chart(fig_docs, use_container_width=True)
    
    @staticmethod
    def display_answer_comparison(results: List[Dict[str, Any]]):
        """Display side-by-side answer comparison."""
        if not results:
            return
            
        st.subheader("📝 답변 비교")
        
        # Create tabs for each system
        if len(results) <= 3:
            tabs = st.tabs([result["rag_type"] for result in results])
            
            for tab, result in zip(tabs, results):
                with tab:
                    st.write("**질문:**")
                    st.write(result["question"])
                    
                    st.write("**답변:**")
                    st.write(result["answer"])
                    
                    st.write("**메타데이터:**")
                    metadata = result.get("metadata", {})
                    for key, value in metadata.items():
                        if isinstance(value, (int, float, str)):
                            st.write(f"• {key}: {value}")
                    
                    if result.get("retrieved_docs"):
                        with st.expander(f"검색된 문서 ({len(result['retrieved_docs'])}개)"):
                            for i, doc in enumerate(result["retrieved_docs"][:3]):
                                st.write(f"**문서 {i+1}:**")
                                st.write(f"출처: {doc.metadata.get('source', 'Unknown')}")
                                st.write(f"내용: {doc.page_content[:150]}...")
                                st.divider()
        else:
            # For more than 3 systems, use selectbox
            selected_system = st.selectbox(
                "비교할 시스템 선택:",
                [result["rag_type"] for result in results]
            )
            
            selected_result = next(r for r in results if r["rag_type"] == selected_system)
            
            st.write("**질문:**")
            st.write(selected_result["question"])
            
            st.write("**답변:**")
            st.write(selected_result["answer"])
            
            st.write("**메타데이터:**")
            metadata = selected_result.get("metadata", {})
            for key, value in metadata.items():
                if isinstance(value, (int, float, str)):
                    st.write(f"• {key}: {value}")
    
    @staticmethod
    def create_processing_flow_diagram(rag_type: str):
        """Create a processing flow diagram for the RAG system."""
        st.subheader(f"🔄 {rag_type} 처리 흐름")
        
        if rag_type == "Naive RAG":
            flow_steps = [
                "사용자 질문",
                "벡터 유사도 검색",
                "문서 검색",
                "LLM 답변 생성",
                "최종 답변"
            ]
        elif rag_type == "Advanced RAG":
            flow_steps = [
                "사용자 질문",
                "쿼리 전처리",
                "벡터 유사도 검색",
                "문서 재순위화",
                "컨텍스트 압축",
                "LLM 답변 생성",
                "최종 답변"
            ]
        elif rag_type == "Modular RAG":
            flow_steps = [
                "사용자 질문",
                "쿼리 분류 & 확장",
                "라우팅 결정",
                "다중 검색 전략",
                "문서 필터링 & 다양성",
                "답변 생성",
                "신뢰도 평가",
                "반복 제어",
                "최종 답변"
            ]
        else:
            flow_steps = ["사용자 질문", "처리", "최종 답변"]
        
        # Create a simple flow diagram using columns
        cols = st.columns(len(flow_steps))
        for i, (col, step) in enumerate(zip(cols, flow_steps)):
            with col:
                st.write(f"**{i+1}.**")
                st.write(step)
                if i < len(flow_steps) - 1:
                    st.write("↓")
    
    @staticmethod
    def display_detailed_metrics(result: Dict[str, Any]):
        """Display detailed metrics for a single result."""
        st.subheader(f"📊 {result['rag_type']} 상세 메트릭")
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("처리 시간", f"{result['total_time']:.2f}초")
        
        with col2:
            retrieved_count = len(result.get('retrieved_docs', []))
            st.metric("검색 문서", f"{retrieved_count}개")
        
        metadata = result.get("metadata", {})
        
        with col3:
            if "confidence" in metadata or "final_confidence" in metadata:
                confidence = metadata.get("final_confidence", metadata.get("confidence", 0))
                st.metric("신뢰도", f"{confidence:.2f}")
        
        with col4:
            if "iterations" in metadata:
                st.metric("반복 횟수", f"{metadata['iterations']}회")
            elif "compression_ratio" in metadata:
                st.metric("압축률", f"{metadata['compression_ratio']:.1%}")
        
        # Additional metrics
        if metadata:
            st.write("**추가 정보:**")
            for key, value in metadata.items():
                if key not in ["confidence", "final_confidence", "iterations", "compression_ratio"]:
                    if isinstance(value, list):
                        st.write(f"• {key}: {', '.join(map(str, value))}")
                    else:
                        st.write(f"• {key}: {value}")
    
    @staticmethod
    def create_summary_report(results: List[Dict[str, Any]]):
        """Create a summary report of all results."""
        if not results:
            return
            
        st.subheader("📋 실험 요약 보고서")
        
        # Summary statistics
        total_systems = len(results)
        avg_time = sum(r["total_time"] for r in results) / total_systems
        total_docs = sum(len(r.get("retrieved_docs", [])) for r in results)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("테스트 시스템", f"{total_systems}개")
        with col2:
            st.metric("평균 처리 시간", f"{avg_time:.2f}초")
        with col3:
            st.metric("총 검색 문서", f"{total_docs}개")
        
        # Best performing system
        fastest_system = min(results, key=lambda x: x["total_time"])
        st.write(f"**⚡ 가장 빠른 시스템:** {fastest_system['rag_type']} ({fastest_system['total_time']:.2f}초)")
        
        most_docs_system = max(results, key=lambda x: len(x.get("retrieved_docs", [])))
        st.write(f"**📚 가장 많은 문서 검색:** {most_docs_system['rag_type']} ({len(most_docs_system.get('retrieved_docs', []))}개)")
        
        # Recommendations
        st.write("**💡 권장사항:**")
        if fastest_system["rag_type"] == "Naive RAG":
            st.write("• 빠른 응답이 필요한 경우 Naive RAG 사용")
        if any("Advanced" in r["rag_type"] for r in results):
            st.write("• 높은 정확도가 필요한 경우 Advanced RAG 사용")
        if any("Modular" in r["rag_type"] for r in results):
            st.write("• 복잡한 질문이나 반복적 개선이 필요한 경우 Modular RAG 사용") 