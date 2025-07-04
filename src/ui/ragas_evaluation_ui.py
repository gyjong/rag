"""RAGAS Evaluation UI module for the RAG application."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datasets import Dataset

from ..config import *
from ..config.settings import (
    RAGAS_EVALUATION_METRICS,
    RAGAS_SAMPLE_SIZE,
    RAGAS_TIMEOUT,
    RAGAS_BATCH_SIZE,
    RAGAS_AVAILABLE_MODELS,
    RAGAS_RESULTS_DIR
)
from ..graphs.naive_rag_graph import create_naive_rag_graph
from ..graphs.advanced_rag_graph import create_advanced_rag_graph
from ..graphs.modular_rag_graph import create_modular_rag_graph
from ..rag_systems.modular_rag import BM25
from ..utils.vector_store import VectorStoreManager
from ..utils.llm_manager import LLMManager
from src.config import langfuse_handler

# 모델 이름 매핑 (설정의 소문자 -> UI 표시용 대문자)
MODEL_NAME_MAPPING = {
    "naive": "Naive RAG",
    "advanced": "Advanced RAG", 
    "modular": "Modular RAG"
}

# 역방향 매핑 (UI -> 설정)
REVERSE_MODEL_MAPPING = {v: k for k, v in MODEL_NAME_MAPPING.items()}

class RagasEvaluationUI:
    """UI for RAG model evaluation using RAGAS."""

    def __init__(self):
        if "evaluation_results" not in st.session_state:
            st.session_state.evaluation_results = None
        if "evaluation_running" not in st.session_state:
            st.session_state.evaluation_running = False

    def display(self):
        """Display the RAGAS evaluation page."""
        st.title("📊 RAGAS 결과 평가")
        
        st.info(
            "이 페이지에서는 Naive, Advanced, Modular RAG 시스템의 성능을 RAGAS를 사용하여 평가합니다.\n\n"
            "**평가 데이터셋 형식:**\n"
            "- `question`: 모델에 질의할 질문 (필수)\n"
            "- `ground_truth`: 이상적인 정답 (필수)\n\n"
            "평가 데이터셋을 아래 표에 직접 입력하거나 CSV/JSON 파일로 업로드할 수 있습니다."
        )

        self._display_dataset_input()
        
        if "eval_dataset" in st.session_state and st.session_state.eval_dataset:
            is_ready = self._check_vector_store()
            if is_ready:
                self._manage_bm25_for_modular_rag()
                self._display_model_selection_and_run()
        
        self._display_results()

    def _get_llm_manager(self):
        """Initializes and returns the LLMManager based on session state."""
        return LLMManager(
            model_name=st.session_state.get("selected_llm_model", "llama3.2:latest"),
            base_url=OLLAMA_BASE_URL,
            temperature=st.session_state.get("llm_temperature", 0.1)
        )

    def _check_vector_store(self):
        """Checks if the vector store is ready for evaluation."""
        if "vector_store_manager" not in st.session_state or not st.session_state.vector_store_manager.get_vector_store():
            st.warning("RAGAS 평가를 실행하려면 먼저 '문서 로딩' 탭에서 문서를 로드하고 '벡터 스토어' 탭에서 벡터 스토어를 생성해야 합니다.")
            return False
        st.success("✅ 벡터 스토어 준비 완료. 평가를 진행할 수 있습니다.")
        return True

    def _manage_bm25_for_modular_rag(self):
        """UI for managing the BM25 index required for Modular RAG."""
        with st.container(border=True):
            st.subheader("🛠️ Modular RAG 사전 설정: BM25 인덱스")
            bm25_index = st.session_state.get("bm25_index")
            bm25_docs_count = len(st.session_state.get("bm25_documents", []))

            if bm25_index:
                st.success(f"✅ BM25 인덱스가 준비되었습니다. ({bm25_docs_count}개 문서 인덱싱됨)")
                if st.button("🔄 BM25 인덱스 재생성", key="ragas_regenerate_bm25"):
                    st.session_state.pop("bm25_index", None)
                    st.session_state.pop("bm25_documents", None)
                    st.rerun()
            else:
                st.warning("Modular RAG를 평가하려면 BM25 키워드 검색을 위한 인덱스 생성이 필요합니다.")
                if st.button("🚀 BM25 인덱스 생성", key="create_bm25_index_ragas"):
                    self._create_bm25_index()

    def _create_bm25_index(self):
        """Creates and stores the BM25 index in the session state."""
        try:
            with st.spinner("벡터 스토어에서 문서를 로드하여 BM25 인덱스를 생성합니다..."):
                vector_store_manager = st.session_state.get("vector_store_manager")
                if not vector_store_manager:
                    st.error("벡터 스토어 매니저를 찾을 수 없습니다.")
                    return

                vector_store = vector_store_manager.get_vector_store()
                stats = vector_store_manager.get_collection_stats()
                total_docs = stats.get("document_count", 1000)

                if total_docs == 0:
                    st.error("인덱싱할 문서가 벡터 스토어에 없습니다.")
                    return

                docs = vector_store.similarity_search("", k=total_docs)
                
                if not docs:
                    st.error("벡터 스토어에서 문서를 가져오지 못했습니다.")
                    return
                
                corpus = [doc.page_content for doc in docs]
                st.session_state.bm25_index = BM25(corpus)
                st.session_state.bm25_documents = docs
            st.rerun()
        except Exception as e:
            st.error(f"BM25 인덱스 생성 실패: {e}")
            st.session_state.pop("bm25_index", None)
            st.session_state.pop("bm25_documents", None)

    def _display_dataset_input(self):
        """Handles the input for the evaluation dataset."""
        st.subheader("1. 평가 데이터셋 준비")

        # Initialize with sample data if not present
        if "eval_df" not in st.session_state:
            sample_data = {
                "question": ["RAG의 한계는 무엇인가요?"],
                "ground_truth": ["RAG는 검색된 문서의 품질에 따라 답변의 정확성이 크게 좌우되며, 관련 없는 정보가 검색되면 환각(Hallucination) 현상이 발생할 수 있습니다. 또한, 실시간 정보나 최신 데이터를 반영하는 데 한계가 있을 수 있습니다."]
            }
            # 설정된 샘플 크기만큼 데이터를 생성 (기본값 사용)
            st.session_state.eval_df = pd.DataFrame(sample_data)

        # Display editable dataframe
        st.session_state.eval_df = st.data_editor(
            st.session_state.eval_df,
            num_rows="dynamic",
            use_container_width=True
        )

        # File uploader
        uploaded_file = st.file_uploader("CSV 또는 JSON 파일 업로드", type=["csv", "json"])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_json(uploaded_file)
                
                # Validate columns
                if "question" not in df.columns or "ground_truth" not in df.columns:
                    st.error("오류: 업로드된 파일에 'question'과 'ground_truth' 컬럼이 모두 포함되어야 합니다.")
                else:
                    st.session_state.eval_df = df
                    st.success(f"✅ '{uploaded_file.name}' 파일이 성공적으로 로드되었습니다.")
            except Exception as e:
                st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

        if not st.session_state.eval_df.empty:
            # Convert to Hugging Face Dataset
            dataset = Dataset.from_pandas(st.session_state.eval_df)
            st.session_state.eval_dataset = dataset
            st.write(f"**데이터셋 정보:** {len(dataset)}개 질문")

    def _display_model_selection_and_run(self):
        """Allows model selection and initiates the evaluation."""
        st.subheader("2. 평가 대상 RAG 모델 선택")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # 설정에서 가져온 모델들을 UI용 이름으로 변환
            available_model_names = [MODEL_NAME_MAPPING[model] for model in RAGAS_AVAILABLE_MODELS]
            models_to_evaluate = st.multiselect(
                "평가할 모델을 선택하세요.",
                options=available_model_names,
                default=available_model_names
            )
        
        with col2:
            st.write(" ") # For alignment
            if st.button("🚀 평가 시작", type="primary", disabled=not models_to_evaluate or st.session_state.evaluation_running):
                self._run_evaluation(models_to_evaluate)

    def _run_evaluation(self, models_to_evaluate):
        """Executes the evaluation process."""
        st.session_state.evaluation_running = True
        st.session_state.evaluation_results = {}

        vector_store_manager = st.session_state.get("vector_store_manager")
        if not vector_store_manager:
            st.error("오류: 벡터 스토어가 초기화되지 않았습니다. '문서 로딩' 탭에서 문서를 먼저 처리해주세요.")
            st.session_state.evaluation_running = False
            return
        
        llm_manager = self._get_llm_manager()

        # RAGAS can be slow, so let's provide a progress bar for the models
        progress_bar = st.progress(0)
        progress_text = st.empty()

        for i, model_name in enumerate(models_to_evaluate):
            progress_text.text(f"'{model_name}' 평가 진행 중... ({i+1}/{len(models_to_evaluate)})")
            
            with st.spinner(f"⏳ **{model_name}** 평가 진행 중... (질문들을 처리하고 있습니다. 다소 시간이 걸릴 수 있습니다.)"):
                try:
                    # Graph setup
                    graph = None
                    if model_name == "Naive RAG":
                        graph = create_naive_rag_graph(llm_manager, vector_store_manager)
                    
                    elif model_name == "Advanced RAG":
                        graph = create_advanced_rag_graph(llm_manager, vector_store_manager)

                    elif model_name == "Modular RAG":
                        if "bm25_index" not in st.session_state:
                            st.warning(f"BM25 인덱스가 없어 '{model_name}' 평가를 건너뜁니다. '사전 설정'에서 인덱스를 생성해주세요.")
                            continue
                        bm25_index = st.session_state.bm25_index
                        bm25_documents = st.session_state.bm25_documents
                        graph = create_modular_rag_graph(llm_manager, vector_store_manager, bm25_index, bm25_documents)

                    if not graph:
                        st.error(f"'{model_name}'에 대한 그래프를 생성할 수 없습니다.")
                        continue
                        
                    # Run inference and collect results
                    results_list = []
                    for item in st.session_state.eval_dataset:
                        inputs = {"query": item["question"]}
                        # Add specific inputs for each graph type
                        if model_name == "Naive RAG":
                            inputs["k"] = 5
                        elif model_name == "Advanced RAG":
                            inputs.update({"k": 10, "rerank_top_k": 5})
                        elif model_name == "Modular RAG":
                            inputs["max_iterations"] = 2
                        
                        response = graph.invoke(inputs, config={"callbacks": [langfuse_handler]})
                        
                        # Extract answer and context based on graph's output keys
                        answer = response.get("answer", "")
                        if model_name == "Naive RAG":
                            contexts = response.get("documents", [])
                        elif model_name == "Advanced RAG":
                            contexts = response.get("reranked_docs", [])
                        elif model_name == "Modular RAG":
                            contexts = response.get("all_retrieved_docs", [])
                        else:
                            contexts = []
                            
                        results_list.append({
                            "question": item["question"],
                            "answer": answer,
                            "contexts": [doc.page_content for doc in contexts],
                            "ground_truth": item["ground_truth"],
                        })

                    result_dataset = Dataset.from_list(results_list)
                    
                    # Perform RAGAS evaluation
                    from ragas import evaluate
                    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
                    
                    # 설정에서 지정된 메트릭들을 동적으로 가져오기
                    metrics = []
                    metric_map = {
                        "faithfulness": faithfulness,
                        "answer_relevancy": answer_relevancy,
                        "context_recall": context_recall,
                        "context_precision": context_precision
                    }
                    
                    for metric_name in RAGAS_EVALUATION_METRICS:
                        if metric_name in metric_map:
                            metrics.append(metric_map[metric_name])
                    
                    score = evaluate(
                        result_dataset,
                        metrics=metrics,
                    )
                    
                    st.session_state.evaluation_results[model_name] = score.to_pandas()
                    st.success(f"✅ **{model_name}** 평가 완료!")
                
                except Exception as e:
                    st.error(f"**{model_name}** 평가 중 오류 발생: {e}")
            
            progress_bar.progress((i + 1) / len(models_to_evaluate))

        progress_text.text("모든 평가 완료!")
        st.session_state.evaluation_running = False
        
        # 평가 결과 저장 (선택사항)
        if st.session_state.evaluation_results:
            self._save_evaluation_results()

    def _save_evaluation_results(self):
        """평가 결과를 설정된 경로에 저장합니다."""
        try:
            import json
            import os
            from datetime import datetime
            
            # 결과 디렉토리 생성
            RAGAS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            
            # 결과 데이터 준비
            results_data = {
                "timestamp": datetime.now().isoformat(),
                "evaluation_metrics": RAGAS_EVALUATION_METRICS,
                "results": {}
            }
            
            for model_name, result_df in st.session_state.evaluation_results.items():
                results_data["results"][model_name] = {
                    "detailed_scores": result_df.to_dict(),
                    "average_scores": result_df.mean(numeric_only=True).to_dict()
                }
            
            # JSON 파일로 저장
            with open(RAGAS_RESULTS_DIR / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            st.warning(f"평가 결과 저장 중 오류 발생: {e}")

    def _display_results(self):
        """Displays the evaluation results with advanced Plotly visualizations."""
        if st.session_state.evaluation_results:
            st.subheader("3. 평가 결과")
            
            # 개별 모델 결과 표시
            for model_name, result_df in st.session_state.evaluation_results.items():
                with st.expander(f"#### {model_name} 상세 결과", expanded=False):
                    st.dataframe(result_df, use_container_width=True)
                    
                    # Display average scores
                    avg_scores = result_df.mean(numeric_only=True)
                    st.write("##### 📈 평균 점수")
                    st.dataframe(avg_scores.to_frame("평균 점수").T, use_container_width=True)
            
            # 종합 비교 시각화
            if len(st.session_state.evaluation_results) > 1:
                self._create_comprehensive_visualizations()
            else:
                # 단일 모델 결과 시각화
                self._create_single_model_visualization()

    def _create_comprehensive_visualizations(self):
        """다중 모델 비교를 위한 종합 시각화를 생성합니다."""
        st.subheader("📊 모델별 종합 비교")
        
        # 데이터 준비
        summary_data = self._prepare_summary_data()
        summary_df = pd.DataFrame(summary_data)
        
        # 시각화 옵션 선택
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            show_radar = st.checkbox("🎯 레이더 차트", value=True)
            show_bar = st.checkbox("📊 바 차트", value=True)
        
        with viz_col2:
            show_heatmap = st.checkbox("🔥 히트맵", value=True)
            show_line = st.checkbox("📈 라인 차트", value=False)
        
        # 요약 테이블
        st.write("##### 📋 종합 점수 테이블")
        st.dataframe(summary_df.set_index("Model"), use_container_width=True)
        
        # 시각화 생성
        if show_radar:
            self._create_radar_chart(summary_df)
        
        if show_bar:
            self._create_bar_chart(summary_df)
        
        if show_heatmap:
            self._create_heatmap(summary_df)
        
        if show_line:
            self._create_line_chart(summary_df)
        
        # 성능 인사이트
        self._display_performance_insights(summary_df)

    def _create_single_model_visualization(self):
        """단일 모델 결과에 대한 시각화를 생성합니다."""
        model_name, result_df = list(st.session_state.evaluation_results.items())[0]
        
        st.subheader(f"📊 {model_name} 상세 분석")
        
        # 메트릭별 분포 차트
        metrics = [col for col in result_df.columns if col in ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']]
        
        if metrics:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=metrics,
                specs=[[{"type": "histogram"}, {"type": "histogram"}],
                       [{"type": "histogram"}, {"type": "histogram"}]]
            )
            
            positions = [(1,1), (1,2), (2,1), (2,2)]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            for i, metric in enumerate(metrics[:4]):
                row, col = positions[i]
                fig.add_trace(
                    go.Histogram(
                        x=result_df[metric],
                        name=metric,
                        marker_color=colors[i],
                        opacity=0.7,
                        nbinsx=10
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title=f"{model_name} 메트릭별 점수 분포",
                height=600,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

    def _prepare_summary_data(self):
        """요약 데이터를 준비합니다."""
        summary_data = {
            "Model": [],
            "Faithfulness": [],
            "Answer Relevancy": [],
            "Context Recall": [],
            "Context Precision": [],
            "Overall Score": []
        }
        
        for model_name, result_df in st.session_state.evaluation_results.items():
            avg_scores = result_df.mean(numeric_only=True)
            summary_data["Model"].append(model_name)
            summary_data["Faithfulness"].append(avg_scores.get("faithfulness", 0))
            summary_data["Answer Relevancy"].append(avg_scores.get("answer_relevancy", 0))
            summary_data["Context Recall"].append(avg_scores.get("context_recall", 0))
            summary_data["Context Precision"].append(avg_scores.get("context_precision", 0))
            
            # 전체 점수 계산 (평균)
            overall_score = avg_scores.mean()
            summary_data["Overall Score"].append(overall_score)
        
        return summary_data

    def _create_radar_chart(self, summary_df):
        """레이더 차트를 생성합니다."""
        st.write("##### 🎯 레이더 차트 - 종합 성능 비교")
        
        fig = go.Figure()
        
        metrics = ["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision"]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        for i, model in enumerate(summary_df["Model"]):
            values = [summary_df.loc[i, metric] for metric in metrics]
            values.append(values[0])  # 차트를 닫기 위해 첫 번째 값 추가
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model,
                line_color=colors[i % len(colors)],
                fillcolor=colors[i % len(colors)],
                opacity=0.3
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="RAG 모델 성능 레이더 차트",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def _create_bar_chart(self, summary_df):
        """인터랙티브 바 차트를 생성합니다."""
        st.write("##### 📊 바 차트 - 메트릭별 성능 비교")
        
        # 메트릭 선택
        metrics = ["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision", "Overall Score"]
        selected_metrics = st.multiselect(
            "표시할 메트릭을 선택하세요:",
            metrics,
            default=metrics[:4],
            key="bar_chart_metrics"
        )
        
        if selected_metrics:
            # 데이터 재구성
            melted_df = summary_df.melt(
                id_vars=["Model"],
                value_vars=selected_metrics,
                var_name="Metric",
                value_name="Score"
            )
            
            fig = px.bar(
                melted_df,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                title="메트릭별 성능 비교",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig.update_layout(
                height=500,
                xaxis_title="RAG 모델",
                yaxis_title="점수",
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)

    def _create_heatmap(self, summary_df):
        """히트맵을 생성합니다."""
        st.write("##### 🔥 히트맵 - 모델 × 메트릭 성능 매트릭스")
        
        metrics = ["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision"]
        heatmap_data = summary_df[["Model"] + metrics].set_index("Model")
        
        fig = px.imshow(
            heatmap_data.T,
            labels=dict(x="RAG 모델", y="평가 메트릭", color="점수"),
            x=heatmap_data.index,
            y=heatmap_data.columns,
            color_continuous_scale="RdYlGn",
            aspect="auto",
            title="모델별 메트릭 성능 히트맵"
        )
        
        # 텍스트 주석 추가
        for i, metric in enumerate(heatmap_data.columns):
            for j, model in enumerate(heatmap_data.index):
                fig.add_annotation(
                    x=j, y=i,
                    text=f"{heatmap_data.loc[model, metric]:.3f}",
                    showarrow=False,
                    font=dict(color="black" if heatmap_data.loc[model, metric] > 0.5 else "white")
                )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    def _create_line_chart(self, summary_df):
        """라인 차트를 생성합니다."""
        st.write("##### 📈 라인 차트 - 메트릭 트렌드")
        
        metrics = ["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision"]
        
        fig = go.Figure()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, model in enumerate(summary_df["Model"]):
            values = [summary_df.loc[i, metric] for metric in metrics]
            
            fig.add_trace(go.Scatter(
                x=metrics,
                y=values,
                mode='lines+markers',
                name=model,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="메트릭별 성능 트렌드",
            xaxis_title="평가 메트릭",
            yaxis_title="점수",
            height=400,
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def _display_performance_insights(self, summary_df):
        """성능 인사이트를 표시합니다."""
        st.write("##### 🎯 성능 인사이트")
        
        # 최고 성능 모델 찾기
        best_overall = summary_df.loc[summary_df["Overall Score"].idxmax(), "Model"]
        best_faithfulness = summary_df.loc[summary_df["Faithfulness"].idxmax(), "Model"]
        best_relevancy = summary_df.loc[summary_df["Answer Relevancy"].idxmax(), "Model"]
        best_recall = summary_df.loc[summary_df["Context Recall"].idxmax(), "Model"]
        best_precision = summary_df.loc[summary_df["Context Precision"].idxmax(), "Model"]
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.info(f"🏆 **종합 최고 성능**: {best_overall}")
            st.success(f"🎯 **최고 충실성**: {best_faithfulness}")
            st.success(f"🎯 **최고 관련성**: {best_relevancy}")
        
        with insights_col2:
            st.success(f"🎯 **최고 재현율**: {best_recall}")
            st.success(f"🎯 **최고 정밀도**: {best_precision}")
        
        # 성능 격차 분석
        st.write("##### 📊 성능 격차 분석")
        
        performance_gap = {}
        for metric in ["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision"]:
            max_score = summary_df[metric].max()
            min_score = summary_df[metric].min()
            gap = max_score - min_score
            performance_gap[metric] = gap
        
        gap_df = pd.DataFrame(list(performance_gap.items()), columns=["메트릭", "성능 격차"])
        gap_df = gap_df.sort_values("성능 격차", ascending=False)
        
        fig = px.bar(
            gap_df,
            x="메트릭",
            y="성능 격차",
            title="메트릭별 모델 간 성능 격차",
            color="성능 격차",
            color_continuous_scale="Reds"
        )
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # 권장사항
        largest_gap_metric = gap_df.iloc[0]["메트릭"]
        st.warning(f"💡 **개선 권장사항**: {largest_gap_metric}에서 가장 큰 성능 격차가 발견되었습니다. 이 영역에 집중하여 모델을 개선하는 것을 권장합니다.") 