"""Font utilities for applying custom fonts globally."""

import streamlit as st
import base64
from pathlib import Path
from ..config import FONT_PATH


def apply_custom_css():
    """Apply Paperlogy font globally to all Streamlit elements."""
    try:
        if FONT_PATH.exists():
            # Paperlogy.ttf 파일을 base64로 인코딩
            with open(FONT_PATH, "rb") as f:
                font_data = f.read()
            font_base64 = base64.b64encode(font_data).decode()

            # CSS로 폰트 등록 및 적용 (모든 요소에 강제 적용)
            st.markdown(
                f"""
                <style>
                @font-face {{
                    font-family: 'Paperlogy';
                    src: url(data:font/ttf;base64,{font_base64}) format('truetype');
                    font-weight: normal;
                    font-style: normal;
                }}
                
                /* 전역 폰트 적용 - 최우선 적용 */
                * {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                
                html, body, div, span, p, h1, h2, h3, h4, h5, h6 {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                
                /* Streamlit 특정 요소들 */
                .stApp, .stApp *, [class*="st"], [class*="css"] {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                
                /* 모든 텍스트 요소 */
                .element-container, .element-container * {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                
                /* 제목 및 헤더 스타일 */
                .main-title {{
                    font-family: 'Paperlogy', sans-serif !important;
                    font-size: 3rem;
                    color: #1f77b4;
                    text-align: center;
                    margin-bottom: 2rem;
                    font-weight: bold;
                }}
                
                .subtitle {{
                    font-family: 'Paperlogy', sans-serif !important;
                    font-size: 1.2rem;
                    color: #666;
                    text-align: center;
                    margin-bottom: 3rem;
                }}
                
                /* 사이드바 스타일 */
                .sidebar .sidebar-content {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                
                /* 버튼 스타일 */
                .stButton > button {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                
                /* 입력 필드 스타일 */
                .stTextInput > div > div > input,
                .stTextArea > div > div > textarea,
                .stSelectbox > div > div > select {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                
                /* 메트릭 스타일 */
                .metric-container {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                
                /* 탭 스타일 */
                .stTabs [data-baseweb="tab-list"] {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                
                /* 마크다운 스타일 */
                .stMarkdown, .stMarkdown * {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                
                /* 추가 Streamlit 요소들 */
                .stSelectbox label, .stTextInput label, .stTextArea label,
                .stNumberInput label, .stSlider label, .stCheckbox label,
                .stRadio label, .stMultiselect label {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                
                /* 데이터프레임 및 테이블 */
                .stDataFrame, .stDataFrame *, .stTable, .stTable * {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                
                /* 경고 및 정보 메시지 */
                .stAlert, .stAlert *, .stSuccess, .stSuccess *,
                .stInfo, .stInfo *, .stWarning, .stWarning *,
                .stError, .stError * {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                
                /* 프로그레스 바 */
                .stProgress, .stProgress * {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                
                /* 스피너 */
                .stSpinner, .stSpinner * {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                
                /* 확장 가능한 섹션 */
                .streamlit-expanderHeader, .streamlit-expanderContent,
                .streamlit-expanderHeader *, .streamlit-expanderContent * {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
            return True
        else:
            st.warning(f"폰트 파일을 찾을 수 없습니다: {FONT_PATH}")
            return False
            
    except Exception as e:
        st.error(f"폰트 로딩 실패: {str(e)}")
        return False


def load_custom_font():
    """Legacy function for backward compatibility."""
    return apply_custom_css() 