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
        
        # Enhanced comparison with more detailed information
        st.write("### 📊 시스템 특성 비교")
        
        for system in systems_info:
            with st.expander(f"🔧 {system['name']} - {system['description']}", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🛠️ 핵심 구성 요소:**")
                    for component in system["components"]:
                        st.write(f"• {component}")
                    
                    if "retrieval_methods" in system:
                        st.write("**🔍 검색 방법:**")
                        for method, desc in system["retrieval_methods"].items():
                            if method == "semantic":
                                st.write(f"• 🧠 의미적 검색: {desc}")
                            elif method == "keyword":
                                st.write(f"• 🔤 키워드 검색: {desc}")
                            elif method == "hybrid":
                                st.write(f"• 🔀 하이브리드: {desc}")
                
                with col2:
                    st.write("**✨ 주요 특징:**")
                    for feature in system["features"]:
                        if "BM25" in feature or "키워드" in feature:
                            st.write(f"🔤 {feature}")
                        elif "하이브리드" in feature or "재순위" in feature:
                            st.write(f"🔀 {feature}")
                        elif "분류" in feature or "질문" in feature:
                            st.write(f"🎯 {feature}")
                        elif "반복" in feature or "개선" in feature:
                            st.write(f"🔄 {feature}")
                        elif "압축" in feature or "스트리밍" in feature:
                            st.write(f"⚡ {feature}")
                        else:
                            st.write(f"• {feature}")
                    
                    if "advantages" in system:
                        st.write("**🏆 주요 장점:**")
                        for advantage in system["advantages"]:
                            st.write(f"✅ {advantage}")
        
        # Create comparison table
        st.write("### 📋 요약 비교표")
        comparison_data = []
        for system in systems_info:
            retrieval_info = ""
            if "retrieval_methods" in system:
                methods = []
                if "semantic" in system["retrieval_methods"]:
                    methods.append("벡터")
                if "keyword" in system["retrieval_methods"]:
                    methods.append("BM25")
                if "hybrid" in system["retrieval_methods"]:
                    methods.append("하이브리드")
                retrieval_info = " + ".join(methods)
            
            comparison_data.append({
                "시스템": system["name"],
                "검색 방식": retrieval_info or "벡터",
                "구성 요소": len(system["components"]),
                "특화 기능": len(system["features"]),
                "복잡도": "높음" if len(system["components"]) > 4 else ("중간" if len(system["components"]) > 2 else "낮음")
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
    
    @staticmethod
    def display_performance_comparison(results: List[Dict[str, Any]]):
        """Display enhanced performance comparison of RAG results."""
        if not results:
            return
            
        st.subheader("⚡ 성능 비교")
        
        # Create enhanced performance metrics
        metrics_data = []
        for result in results:
            metadata = result.get("metadata", {})
            
            # Handle different metadata keys for retrieved document count
            retrieved_count = (
                metadata.get("final_retrieved") or  # Advanced RAG
                metadata.get("num_retrieved") or    # Naive RAG
                metadata.get("total_retrieved") or  # Modular RAG
                len(result.get("retrieved_docs", []))  # Fallback to counting docs
            )
            
            # Determine search methods used
            search_methods = []
            if metadata.get("retrieval_method") == "hybrid":
                search_methods.append("벡터+재순위")
            elif "semantic" in str(metadata.get("retrieval_method", "")):
                search_methods.append("벡터")
            
            if "keyword_scores" in metadata or "bm25" in str(metadata).lower():
                search_methods.append("BM25")
                
            search_method_str = " + ".join(search_methods) if search_methods else metadata.get("retrieval_method", "벡터")
            
            # Advanced metrics
            query_type = metadata.get("query_type", "일반")
            confidence = metadata.get("final_confidence", metadata.get("confidence", 0))
            iterations = metadata.get("iterations", 1)
            compression_ratio = metadata.get("compression_ratio", 0)
            
            metrics_data.append({
                "시스템": result["rag_type"],
                "처리 시간 (초)": round(result["total_time"], 2),
                "검색 문서 수": retrieved_count,
                "검색 방법": search_method_str,
                "질문 유형": query_type,
                "신뢰도": round(confidence, 2) if confidence > 0 else "N/A",
                "반복 횟수": iterations,
                "압축률": f"{compression_ratio:.1%}" if compression_ratio > 0 else "N/A"
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Display enhanced metrics table
        st.dataframe(df_metrics, use_container_width=True)
        
        # Create enhanced performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Processing time chart with color coding
            fig_time = px.bar(
                df_metrics, 
                x="시스템", 
                y="처리 시간 (초)",
                title="⏱️ 처리 시간 비교",
                color="검색 방법",
                text="처리 시간 (초)"
            )
            fig_time.update_traces(texttemplate='%{text}s', textposition='outside')
            fig_time.update_layout(showlegend=True)
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            # Retrieved documents chart with search method info
            fig_docs = px.bar(
                df_metrics, 
                x="시스템", 
                y="검색 문서 수",
                title="📚 검색 문서 수 비교",
                color="검색 방법",
                text="검색 문서 수"
            )
            fig_docs.update_traces(texttemplate='%{text}', textposition='outside')
            fig_docs.update_layout(showlegend=True)
            st.plotly_chart(fig_docs, use_container_width=True)
        
        # Additional performance insights
        st.write("### 🎯 성능 인사이트")
        
        # Performance analysis
        insights = []
        
        # Find best performing systems
        fastest_system = min(results, key=lambda x: x["total_time"])
        most_docs_system = max(results, key=lambda x: len(x.get("retrieved_docs", [])))
        
        insights.append(f"⚡ **최고 속도**: {fastest_system['rag_type']} ({fastest_system['total_time']:.2f}초)")
        insights.append(f"📚 **최다 검색**: {most_docs_system['rag_type']} ({len(most_docs_system.get('retrieved_docs', []))}개 문서)")
        
        # Check for advanced features
        for result in results:
            metadata = result.get("metadata", {})
            if metadata.get("final_confidence", 0) > 0.8:
                insights.append(f"🎯 **고신뢰도**: {result['rag_type']} (신뢰도 {metadata.get('final_confidence', 0):.2f})")
            if metadata.get("iterations", 1) > 1:
                insights.append(f"🔄 **반복 개선**: {result['rag_type']} ({metadata.get('iterations')}회 반복)")
            if metadata.get("compression_ratio", 0) > 0:
                insights.append(f"📦 **컨텍스트 압축**: {result['rag_type']} ({metadata.get('compression_ratio'):.1%} 압축)")
        
        for insight in insights:
            st.write(insight)
    
    @staticmethod
    def display_answer_comparison(results: List[Dict[str, Any]]):
        """Display enhanced side-by-side answer comparison."""
        if not results:
            return
            
        st.subheader("📝 답변 비교")
        
        # Create tabs for each system
        if len(results) <= 3:
            tabs = st.tabs([f"{result['rag_type']}" for result in results])
            
            for tab, result in zip(tabs, results):
                with tab:
                    metadata = result.get("metadata", {})
                    
                    # Show enhanced header with key metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("처리 시간", f"{result['total_time']:.2f}초")
                    with col2:
                        if metadata.get("final_confidence", 0) > 0:
                            st.metric("신뢰도", f"{metadata.get('final_confidence', 0):.2f}")
                        elif metadata.get("confidence", 0) > 0:
                            st.metric("신뢰도", f"{metadata.get('confidence', 0):.2f}")
                    with col3:
                        doc_count = (
                            metadata.get("final_retrieved") or 
                            metadata.get("total_retrieved") or 
                            len(result.get("retrieved_docs", []))
                        )
                        st.metric("검색 문서", f"{doc_count}개")
                    
                    st.write("**질문:**")
                    st.write(result["question"])
                    
                    # Show query analysis for Modular RAG
                    if "query_type" in metadata:
                        st.info(f"🎯 질문 유형: {metadata['query_type']}")
                    if "processing_path" in metadata:
                        st.info(f"🛤️ 처리 경로: {metadata['processing_path']}")
                    if "expansion_terms" in metadata and metadata["expansion_terms"]:
                        st.info(f"🔍 확장 용어: {', '.join(metadata['expansion_terms'])}")
                    
                    st.write("**답변:**")
                    st.write(result["answer"])
                    
                    # Enhanced metadata display
                    with st.expander("📊 상세 메타데이터"):
                        for key, value in metadata.items():
                            if key == "retrieval_method":
                                st.write(f"🔍 검색 방법: {value}")
                            elif key == "query_type":
                                st.write(f"🎯 질문 유형: {value}")
                            elif key == "iterations":
                                st.write(f"🔄 반복 횟수: {value}")
                            elif key == "compression_ratio":
                                st.write(f"📦 압축률: {value:.1%}")
                            elif key == "rerank_scores" and isinstance(value, list):
                                st.write(f"📈 재순위 점수: {[f'{s:.3f}' for s in value[:3]]}")
                            elif key == "keyword_scores" and isinstance(value, list):
                                st.write(f"🔤 키워드 점수: {[f'{s:.3f}' for s in value[:3]]}")
                            elif isinstance(value, (int, float, str)) and len(str(value)) < 100:
                                st.write(f"• {key}: {value}")
                    
                    if result.get("retrieved_docs"):
                        with st.expander(f"📚 검색된 문서 ({len(result['retrieved_docs'])}개)"):
                            for i, doc in enumerate(result["retrieved_docs"][:5]):  # Show top 5
                                st.write(f"**문서 {i+1}:**")
                                st.write(f"📂 출처: {doc.metadata.get('source', 'Unknown')}")
                                if "score" in doc.metadata:
                                    st.write(f"📊 점수: {doc.metadata['score']:.3f}")
                                st.write(f"📄 내용: {doc.page_content[:200]}...")
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
        """Create an enhanced processing flow diagram for the RAG system."""
        st.subheader(f"🔄 {rag_type} 처리 흐름")
        
        if rag_type == "Naive RAG":
            flow_steps = [
                "📝 사용자 질문",
                "🧠 벡터 유사도 검색",
                "📚 문서 검색",
                "🤖 LLM 답변 생성",
                "✅ 최종 답변"
            ]
            description = "단순하고 빠른 검색 → 생성 파이프라인"
            
        elif rag_type == "Advanced RAG":
            flow_steps = [
                "📝 사용자 질문",
                "🔧 쿼리 전처리 & 최적화",
                "🧠 벡터 유사도 검색",
                "🔀 하이브리드 재순위화<br/>(TF-IDF + 벡터)",
                "📦 컨텍스트 압축",
                "🤖 실시간 스트리밍 생성",
                "✅ 최종 답변"
            ]
            description = "Enhanced Search: 하이브리드 검색 + 압축 + 스트리밍"
            
        elif rag_type == "Modular RAG":
            flow_steps = [
                "📝 사용자 질문",
                "🎯 쿼리 분류 & 확장<br/>(7가지 유형)",
                "🛤️ 라우팅 결정<br/>(처리 경로 선택)",
                "🔍 하이브리드 검색<br/>(벡터 + BM25)",
                "🔧 필터링 & 다양성",
                "🤖 유형별 맞춤 생성",
                "📊 신뢰도 평가",
                "🔄 반복 제어<br/>(신뢰도 < 0.7시 재시도)",
                "✅ 최종 답변"
            ]
            description = "모듈형 아키텍처: 질문 유형별 최적화 + 반복적 개선"
            
        else:
            flow_steps = ["📝 사용자 질문", "🔧 처리", "✅ 최종 답변"]
            description = "기본 처리 흐름"
        
        st.info(f"**특징**: {description}")
        
        # Create an enhanced flow diagram
        num_cols = min(len(flow_steps), 4)  # Max 4 columns per row
        rows = [flow_steps[i:i+num_cols] for i in range(0, len(flow_steps), num_cols)]
        
        for row_idx, row in enumerate(rows):
            cols = st.columns(len(row))
            for i, (col, step) in enumerate(zip(cols, row)):
                with col:
                    step_num = row_idx * num_cols + i + 1
                    
                    # Use different colors for different types of steps
                    if "질문" in step:
                        st.markdown(f"""
                        <div style="background-color: #e3f2fd; padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 10px;">
                        <strong>{step_num}.</strong><br/>{step}
                        </div>
                        """, unsafe_allow_html=True)
                    elif "검색" in step or "BM25" in step:
                        st.markdown(f"""
                        <div style="background-color: #f3e5f5; padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 10px;">
                        <strong>{step_num}.</strong><br/>{step}
                        </div>
                        """, unsafe_allow_html=True)
                    elif "생성" in step or "답변" in step:
                        st.markdown(f"""
                        <div style="background-color: #e8f5e8; padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 10px;">
                        <strong>{step_num}.</strong><br/>{step}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: #fff3e0; padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 10px;">
                        <strong>{step_num}.</strong><br/>{step}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Add arrow except for last step
                    if step_num < len(flow_steps):
                        if i == len(row) - 1 and row_idx < len(rows) - 1:
                            # End of row, point down
                            st.markdown("<div style='text-align: center; font-size: 20px;'>⬇️</div>", unsafe_allow_html=True)
                        elif i < len(row) - 1:
                            # Within row, point right
                            st.markdown("<div style='text-align: center; font-size: 20px;'>➡️</div>", unsafe_allow_html=True)
    
    @staticmethod
    def display_detailed_metrics(result: Dict[str, Any]):
        """Display enhanced detailed metrics for a single result."""
        st.subheader(f"📊 {result['rag_type']} 상세 메트릭")
        
        metadata = result.get("metadata", {})
        
        # Enhanced basic metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("⏱️ 처리 시간", f"{result['total_time']:.2f}초")
        
        with col2:
            retrieved_count = (
                metadata.get("final_retrieved") or  
                metadata.get("num_retrieved") or    
                metadata.get("total_retrieved") or  
                len(result.get("retrieved_docs", []))
            )
            st.metric("📚 검색 문서", f"{retrieved_count}개")
        
        with col3:
            if "confidence" in metadata or "final_confidence" in metadata:
                confidence = metadata.get("final_confidence", metadata.get("confidence", 0))
                st.metric("🎯 신뢰도", f"{confidence:.2f}")
        
        with col4:
            if "iterations" in metadata:
                st.metric("🔄 반복 횟수", f"{metadata['iterations']}회")
            elif "compression_ratio" in metadata:
                st.metric("📦 압축률", f"{metadata['compression_ratio']:.1%}")
        
        with col5:
            if "query_type" in metadata:
                st.metric("🎯 질문 유형", metadata["query_type"])
        
        # Advanced metrics section
        if metadata:
            st.write("### 🔍 고급 메트릭")
            
            # Create metrics categories
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔧 처리 정보:**")
                if "retrieval_method" in metadata:
                    st.write(f"• 검색 방법: {metadata['retrieval_method']}")
                if "processing_path" in metadata:
                    st.write(f"• 처리 경로: {metadata['processing_path']}")
                if "expansion_terms" in metadata and metadata["expansion_terms"]:
                    st.write(f"• 확장 용어: {', '.join(metadata['expansion_terms'][:5])}")
                if "generation_method" in metadata:
                    st.write(f"• 생성 방법: {metadata['generation_method']}")
            
            with col2:
                st.write("**📊 성능 지표:**")
                if "rerank_scores" in metadata and isinstance(metadata["rerank_scores"], list):
                    avg_rerank = sum(metadata["rerank_scores"]) / len(metadata["rerank_scores"])
                    st.write(f"• 평균 재순위 점수: {avg_rerank:.3f}")
                if "keyword_scores" in metadata and isinstance(metadata["keyword_scores"], list):
                    avg_keyword = sum(metadata["keyword_scores"]) / len(metadata["keyword_scores"])
                    st.write(f"• 평균 키워드 점수: {avg_keyword:.3f}")
                if "context_length" in metadata:
                    st.write(f"• 컨텍스트 길이: {metadata['context_length']} 토큰")
        
        # Visualization of search results if available
        if "rerank_scores" in metadata or "keyword_scores" in metadata:
            st.write("### 📈 검색 점수 분포")
            
            score_data = []
            if "rerank_scores" in metadata:
                for i, score in enumerate(metadata["rerank_scores"][:10]):
                    score_data.append({"문서": f"Doc {i+1}", "재순위 점수": score, "유형": "재순위"})
            
            if "keyword_scores" in metadata:
                for i, score in enumerate(metadata["keyword_scores"][:10]):
                    if i < len(score_data):
                        score_data[i]["키워드 점수"] = score
                    else:
                        score_data.append({"문서": f"Doc {i+1}", "키워드 점수": score, "유형": "키워드"})
            
            if score_data:
                df_scores = pd.DataFrame(score_data)
                
                if "재순위 점수" in df_scores.columns and "키워드 점수" in df_scores.columns:
                    # Dual score visualization
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='재순위 점수',
                        x=df_scores['문서'],
                        y=df_scores['재순위 점수'],
                        yaxis='y'
                    ))
                    fig.add_trace(go.Bar(
                        name='키워드 점수',
                        x=df_scores['문서'],
                        y=df_scores['키워드 점수'],
                        yaxis='y2'
                    ))
                    
                    fig.update_layout(
                        xaxis=dict(title='문서'),
                        yaxis=dict(title='재순위 점수', side='left'),
                        yaxis2=dict(title='키워드 점수', side='right', overlaying='y'),
                        title='문서별 검색 점수 비교'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Single score type
                    score_col = "재순위 점수" if "재순위 점수" in df_scores.columns else "키워드 점수"
                    fig = px.bar(df_scores, x="문서", y=score_col, title=f"문서별 {score_col}")
                    st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_summary_report(results: List[Dict[str, Any]]):
        """Create an enhanced summary report of all results."""
        if not results:
            return
            
        st.subheader("📋 실험 요약 보고서")
        
        # Enhanced summary statistics
        total_systems = len(results)
        avg_time = sum(r["total_time"] for r in results) / total_systems
        total_docs = sum(len(r.get("retrieved_docs", [])) for r in results)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🧪 테스트 시스템", f"{total_systems}개")
        with col2:
            st.metric("⏱️ 평균 처리 시간", f"{avg_time:.2f}초")
        with col3:
            st.metric("📚 총 검색 문서", f"{total_docs}개")
        with col4:
            avg_docs = total_docs / total_systems if total_systems > 0 else 0
            st.metric("📄 평균 문서수", f"{avg_docs:.1f}개")
        
        # Enhanced performance analysis
        st.write("### 🏆 성능 분석")
        
        fastest_system = min(results, key=lambda x: x["total_time"])
        most_docs_system = max(results, key=lambda x: len(x.get("retrieved_docs", [])))
        
        # Advanced analysis
        analysis_results = []
        
        for result in results:
            metadata = result.get("metadata", {})
            analysis = {
                "시스템": result["rag_type"],
                "속도": "🚀 빠름" if result["total_time"] < avg_time else "🐢 느림",
                "검색량": "📚 많음" if len(result.get("retrieved_docs", [])) > avg_docs else "📄 적음",
                "특수기능": []
            }
            
            # Check for special features
            if metadata.get("final_confidence", 0) > 0.7:
                analysis["특수기능"].append("🎯 고신뢰도")
            if metadata.get("iterations", 1) > 1:
                analysis["특수기능"].append("🔄 반복개선")
            if metadata.get("compression_ratio", 0) > 0:
                analysis["특수기능"].append("📦 압축")
            if "bm25" in str(metadata).lower() or "keyword" in str(metadata).lower():
                analysis["특수기능"].append("🔤 키워드검색")
            if metadata.get("query_type") != "general":
                analysis["특수기능"].append("🎯 질문분류")
            
            analysis_results.append(analysis)
        
        # Display analysis
        for analysis in analysis_results:
            feature_text = " ".join(analysis["특수기능"]) if analysis["특수기능"] else "기본"
            st.write(f"**{analysis['시스템']}**: {analysis['속도']} | {analysis['검색량']} | {feature_text}")
        
        # System recommendations
        st.write("### 💡 시스템 선택 가이드")
        
        recommendations = []
        
        if fastest_system["rag_type"] == "Naive RAG":
            recommendations.append("⚡ **빠른 응답이 최우선**인 경우 → Naive RAG")
        
        # Check for Advanced RAG features
        for result in results:
            if "Advanced" in result["rag_type"]:
                metadata = result.get("metadata", {})
                if metadata.get("compression_ratio", 0) > 0:
                    recommendations.append("📦 **정확성과 효율성의 균형**이 필요한 경우 → Advanced RAG")
                break
        
        # Check for Modular RAG features
        for result in results:
            if "Modular" in result["rag_type"]:
                metadata = result.get("metadata", {})
                if metadata.get("iterations", 1) > 1:
                    recommendations.append("🔄 **복잡한 질문과 높은 정확성**이 필요한 경우 → Modular RAG")
                if metadata.get("query_type") != "general":
                    recommendations.append("🎯 **다양한 질문 유형**을 다루는 경우 → Modular RAG")
                break
        
        # Check for hybrid search
        hybrid_systems = [r for r in results if "hybrid" in str(r.get("metadata", {})).lower()]
        if hybrid_systems:
            recommendations.append("🔍 **의미적 + 키워드 검색**이 모두 필요한 경우 → 하이브리드 시스템")
        
        for rec in recommendations:
            st.write(rec)
        
        # Performance summary table
        st.write("### 📊 종합 성능표")
        summary_data = []
        for result in results:
            metadata = result.get("metadata", {})
            summary_data.append({
                "시스템": result["rag_type"],
                "처리시간": f"{result['total_time']:.2f}s",
                "검색문서": len(result.get("retrieved_docs", [])),
                "신뢰도": f"{metadata.get('final_confidence', metadata.get('confidence', 0)):.2f}",
                "특징": ", ".join([
                    "하이브리드" if "hybrid" in str(metadata).lower() else "",
                    "반복" if metadata.get("iterations", 1) > 1 else "",
                    "압축" if metadata.get("compression_ratio", 0) > 0 else "",
                    "분류" if metadata.get("query_type", "general") != "general" else ""
                ]).strip(", ") or "기본"
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True) 