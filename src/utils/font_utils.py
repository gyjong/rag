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
                /* Google Material Icons 폰트 로드 */
                @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
                @import url('https://fonts.googleapis.com/icon?family=Material+Icons+Outlined');
                @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
                
                @font-face {{
                    font-family: 'Paperlogy';
                    src: url(data:font/ttf;base64,{font_base64}) format('truetype');
                    font-weight: normal;
                    font-style: normal;
                }}
                
                /* 전역 폰트 적용 - 아이콘 폰트 제외 */
                html, body, div, span, p, h1, h2, h3, h4, h5, h6 {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                
                /* Streamlit 특정 요소들 - 아이콘 요소 제외 */
                .stApp {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                
                /* 아이콘이 아닌 텍스트 요소들만 적용 */
                [class*="st"]:not([class*="icon"]):not([class*="Icon"]) {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                
                /* Material Icons 및 Symbols 폰트 유지 - 매우 강력한 선택자 */
                .material-icons, .material-icons-outlined, .material-icons-round,
                .material-icons-sharp, .material-icons-two-tone, .material-symbols-outlined,
                [data-testid*="collapsedControl"], [data-testid*="expandedControl"],
                [data-testid="collapsedControl"] *, [data-testid="expandedControl"] *,
                button[kind="icon"], button[kind="icon"] *,
                [class*="chevron"], [class*="arrow"], [class*="expand"],
                [aria-label*="expand"], [aria-label*="collapse"],
                [title*="expand"], [title*="collapse"],
                /* Streamlit 사이드바 토글 관련 모든 가능한 선택자 */
                [class*="sidebar"] button, [class*="sidebar"] button *,
                [class*="Sidebar"] button, [class*="Sidebar"] button *,
                button[aria-label*="sidebar"], button[aria-label*="Sidebar"],
                button[title*="sidebar"], button[title*="Sidebar"],
                .st-emotion-cache-* button[kind="icon"],
                .st-emotion-cache-* button[kind="icon"] *,
                /* 토글 관련 요소들 */
                [class*="toggle"], [class*="Toggle"],
                [class*="collapse"], [class*="Collapse"],
                [data-baseweb="button"][kind="icon"],
                [data-baseweb="button"][kind="icon"] * {{
                    font-family: 'Material Icons', 'Material Icons Outlined', 'Material Symbols Outlined', monospace !important;
                    font-feature-settings: 'liga' 1;
                    font-display: block;
                    text-rendering: optimizeLegibility;
                    -webkit-font-feature-settings: 'liga';
                    -webkit-font-smoothing: antialiased;
                }}
                
                /* 텍스트 요소 - 아이콘 제외 */
                .element-container:not([class*="icon"]):not([class*="Icon"]) {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                
                .element-container *:not([class*="icon"]):not([class*="Icon"]):not([class*="chevron"]):not([class*="arrow"]) {{
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
                
                /* 사이드바 스타일 - 아이콘 버튼 제외 */
                .sidebar .sidebar-content {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                
                /* 사이드바 토글 버튼은 아이콘 폰트 유지 - 강력한 선택자 */
                .sidebar button[kind="icon"], 
                .sidebar button[kind="icon"] *,
                [data-testid="collapsedControl"],
                [data-testid="expandedControl"],
                [data-testid="collapsedControl"] *,
                [data-testid="expandedControl"] *,
                /* 모든 사이드바 관련 버튼 */
                section[data-testid="stSidebar"] button,
                section[data-testid="stSidebar"] button *,
                .css-* button[kind="icon"],
                .css-* button[kind="icon"] *,
                div[class*="sidebar"] button,
                div[class*="sidebar"] button * {{
                    font-family: 'Material Icons', 'Material Symbols Outlined' !important;
                    font-feature-settings: 'liga' 1 !important;
                    -webkit-font-feature-settings: 'liga' !important;
                    font-display: block !important;
                }}
                
                /* 버튼 스타일 - 아이콘 버튼 제외 */
                .stButton > button:not([kind="icon"]) {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                
                /* 모든 아이콘 버튼은 Material Icons 폰트 유지 */
                button[kind="icon"], button[kind="icon"] *,
                button[data-testid*="icon"], button[data-testid*="icon"] *,
                .stButton button[kind="icon"], .stButton button[kind="icon"] *,
                [role="button"][kind="icon"], [role="button"][kind="icon"] *,
                button:has([class*="icon"]), button:has([class*="icon"]) *,
                /* CSS 클래스로 생성된 모든 아이콘 요소 */
                .st-emotion-cache-* [kind="icon"],
                .st-emotion-cache-* [kind="icon"] *,
                [class*="css-"] button[kind="icon"],
                [class*="css-"] button[kind="icon"] * {{
                    font-family: 'Material Icons', 'Material Symbols Outlined' !important;
                    font-feature-settings: 'liga' 1 !important;
                    -webkit-font-feature-settings: 'liga' !important;
                    font-display: block !important;
                    text-rendering: optimizeLegibility !important;
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
                
                /* 최종 강력한 아이콘 폰트 보장 - 모든 선택자를 덮어씀 */
                * {{
                    font-family: 'Paperlogy', sans-serif !important;
                }}
                
                /* 하지만 아이콘 관련 요소는 반드시 Material Icons 사용 */
                *[class*="icon" i] *, *[kind="icon"] *, *[data-testid*="Control"] *,
                button[kind="icon"], button[kind="icon"] *,
                [data-testid="collapsedControl"], [data-testid="expandedControl"],
                [data-testid="collapsedControl"] *, [data-testid="expandedControl"] *,
                section[data-testid="stSidebar"] button *,
                section[data-testid="stSidebar"] button {{
                    font-family: 'Material Icons', 'Material Symbols Outlined', sans-serif !important;
                    font-feature-settings: 'liga' 1 !important;
                    -webkit-font-feature-settings: 'liga' !important;
                    font-display: block !important;
                    text-rendering: optimizeLegibility !important;
                }}
                
                /* 최종 보장: keyboard_double_arrow가 포함된 텍스트는 Material Icons로 */
                *:contains("keyboard_double_arrow"),
                *[innerHTML*="keyboard_double_arrow"],
                *[textContent*="keyboard_double_arrow"] {{
                    font-family: 'Material Icons', 'Material Symbols Outlined' !important;
                    font-feature-settings: 'liga' 1 !important;
                    -webkit-font-feature-settings: 'liga' !important;
                }}
                </style>
                
                <script>
                // JavaScript로 사이드바 토글 버튼 아이콘 문제 직접 해결
                function fixSidebarIcons() {{
                    // 모든 텍스트 노드에서 keyboard_double_arrow를 찾아서 아이콘으로 변경
                    const walker = document.createTreeWalker(
                        document.body,
                        NodeFilter.SHOW_TEXT,
                        null,
                        false
                    );
                    
                    const textNodesToReplace = [];
                    let node;
                    
                    while (node = walker.nextNode()) {{
                        if (node.textContent.includes('keyboard_double_arrow_left')) {{
                            textNodesToReplace.push({{node: node, text: '◀', original: 'keyboard_double_arrow_left'}});
                        }} else if (node.textContent.includes('keyboard_double_arrow_right')) {{
                            textNodesToReplace.push({{node: node, text: '▶', original: 'keyboard_double_arrow_right'}});
                        }}
                    }}
                    
                    // 텍스트 노드 교체
                    textNodesToReplace.forEach(item => {{
                        const newText = item.node.textContent.replace(item.original, item.text);
                        item.node.textContent = newText;
                        
                        // 부모 요소에 Material Icons 폰트 적용
                        let parent = item.node.parentElement;
                        if (parent) {{
                            parent.style.fontFamily = 'Material Icons, Material Symbols Outlined, sans-serif';
                            parent.style.fontFeatureSettings = '"liga" 1';
                        }}
                    }});
                }}
                
                // 페이지 로드 시 실행
                document.addEventListener('DOMContentLoaded', fixSidebarIcons);
                
                // Streamlit 렌더링 후에도 실행 (지연 실행)
                setTimeout(fixSidebarIcons, 100);
                setTimeout(fixSidebarIcons, 500);
                setTimeout(fixSidebarIcons, 1000);
                
                // MutationObserver로 동적 변경 감지
                const observer = new MutationObserver(function(mutations) {{
                    mutations.forEach(function(mutation) {{
                        if (mutation.type === 'childList') {{
                            mutation.addedNodes.forEach(function(node) {{
                                if (node.nodeType === Node.TEXT_NODE) {{
                                    if (node.textContent.includes('keyboard_double_arrow')) {{
                                        fixSidebarIcons();
                                    }}
                                }} else if (node.nodeType === Node.ELEMENT_NODE) {{
                                    if (node.textContent.includes('keyboard_double_arrow')) {{
                                        fixSidebarIcons();
                                    }}
                                }}
                            }});
                        }}
                    }});
                }});
                
                observer.observe(document.body, {{
                    childList: true,
                    subtree: true
                }});
                </script>
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