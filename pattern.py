import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ---------- 패턴 정의 ----------
predefined_patterns = {
    "Glider": np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ]),
    "Blinker": np.array([
        [1, 1, 1]
    ]),
    "Block": np.array([
        [1, 1],
        [1, 1]
    ]),
    "Toad": np.array([
        [0, 1, 1, 1],
        [1, 1, 1, 0]
    ])
}

# ---------- 1D 오토마타 ----------
def get_rule_binary(rule_number):
    return np.array([int(bit) for bit in np.binary_repr(rule_number, width=8)], dtype=np.uint8)

def create_rule_map(rule_binary):
    patterns = [(1,1,1),(1,1,0),(1,0,1),(1,0,0),(0,1,1),(0,1,0),(0,0,1),(0,0,0)]
    return {p: rule_binary[i] for i, p in enumerate(patterns)}

def run_1d_automaton(rule_number, width, generations, start_mode="center"):
    rule_bin = get_rule_binary(rule_number)
    rule_map = create_rule_map(rule_bin)
    grid = np.zeros((generations, width), dtype=np.uint8)

    if start_mode == "center":
        grid[0, width // 2] = 1
    elif start_mode == "random":
        np.random.seed(42)
        grid[0] = np.random.choice([0, 1], size=width)

    for g in range(1, generations):
        for i in range(width):
            left = grid[g-1, (i-1)%width]
            center = grid[g-1, i]
            right = grid[g-1, (i+1)%width]
            grid[g, i] = rule_map[(left, center, right)]
    return grid

def generate_1d_animation(grid):
    generations, width = grid.shape
    fig = go.Figure(
        frames=[
            go.Frame(data=[go.Heatmap(z=grid[:g+1], colorscale="gray", showscale=False)], name=str(g))
            for g in range(1, generations)
        ]
    )
    fig.add_trace(go.Heatmap(z=grid[:1], colorscale="gray", showscale=False))
    fig.update_layout(
        title="1차원 셀룰러 오토마타 애니메이션",
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "▶️ 재생", "method": "animate", "args": [None, {"frame": {"duration": 100, "redraw": True}}]},
                {"label": "⏸️ 정지", "method": "animate", "args": [[None], {"mode": "immediate"}]}
            ]
        }],
        height=600,
        xaxis_title="셀 위치",
        yaxis_title="세대",
    )
    return fig

# ---------- 2D 생명 게임 ----------
def run_game_of_life(rows, cols, generations, seed=42):
    np.random.seed(seed)
    grid = np.random.choice([0, 1], size=(rows, cols))
    history = [grid.copy()]

    for _ in range(generations):
        new_grid = grid.copy()
        for i in range(rows):
            for j in range(cols):
                neighbors = np.sum(grid[max(i-1,0):i+2, max(j-1,0):j+2]) - grid[i, j]
                if grid[i, j] == 1 and (neighbors < 2 or neighbors > 3):
                    new_grid[i, j] = 0
                elif grid[i, j] == 0 and neighbors == 3:
                    new_grid[i, j] = 1
        grid = new_grid
        history.append(grid.copy())
    return history

def generate_2d_animation(history):
    frames = [
        go.Frame(data=[go.Heatmap(z=frame, colorscale='Greys', showscale=False)], name=str(i))
        for i, frame in enumerate(history)
    ]
    fig = go.Figure(
        data=[go.Heatmap(z=history[0], colorscale='Greys', showscale=False)],
        frames=frames
    )
    fig.update_layout(
        title="셀룰러 오토마타 시뮬레이션",
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "▶️ 재생", "method": "animate", "args": [None, {"frame": {"duration": 300, "redraw": True}}]},
                {"label": "⏸️ 정지", "method": "animate", "args": [[None], {"mode": "immediate"}]}
            ]
        }],
        height=600
    )
    return fig

# ---------- 사용자 지정 생명 게임 ----------
def simulate_custom_life(initial_grid, generations):
    history = [initial_grid.copy()]
    grid = initial_grid.copy()
    for _ in range(generations):
        new_grid = grid.copy()
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                neighbors = np.sum(grid[max(i-1,0):i+2, max(j-1,0):j+2]) - grid[i,j]
                if grid[i,j] == 1 and (neighbors < 2 or neighbors > 3):
                    new_grid[i,j] = 0
                elif grid[i,j] == 0 and neighbors == 3:
                    new_grid[i,j] = 1
        grid = new_grid
        history.append(grid.copy())
    return history

# ---------- Streamlit 앱 ----------
st.set_page_config(layout="wide")
st.title("🧬 셀룰러 오토마타 실험실")

tabs = st.tabs([
    "📏 1차원 오토마타",
    "🧱 2차원 오토마타",
    "🌱 생명 게임",
    "📊 Rule 비교",
    "📖 이론 요약",
    "🧪 패턴 실험실"
])

# 각 탭 구현
with tabs[0]:
    st.subheader("📏 1차원 셀룰러 오토마타")
    rule_number = st.slider("Rule 번호", 0, 255, 30)
    width = st.slider("셀 수", 20, 201, 101)
    generations = st.slider("세대 수", 10, 100, 40)
    start_mode = st.radio("초기 상태", ["center", "random"])
    grid = run_1d_automaton(rule_number, width, generations, start_mode)
    fig = generate_1d_animation(grid)
    st.plotly_chart(fig, use_container_width=True, key="1d_automaton")

with tabs[1]:
    st.subheader("🧱 2차원 오토마타 (랜덤 초기 상태)")
    rows = st.slider("행 수", 10, 100, 30)
    cols = st.slider("열 수", 10, 100, 30)
    generations = st.slider("세대 수", 5, 50, 20)
    history = run_game_of_life(rows, cols, generations)
    fig = generate_2d_animation(history)
    st.plotly_chart(fig, use_container_width=True, key="2d_automaton")

with tabs[2]:
    st.subheader("🌱 생명 게임 (Game of Life)")
    rows = st.slider("행 수", 10, 100, 30, key="life_rows")
    cols = st.slider("열 수", 10, 100, 30, key="life_cols")
    generations = st.slider("세대 수", 5, 50, 20, key="life_gen")
    history = run_game_of_life(rows, cols, generations)
    fig = generate_2d_animation(history)
    st.plotly_chart(fig, use_container_width=True, key="game_of_life")

with tabs[3]:
    st.subheader("📊 Rule 비교")
    selected_rules = st.multiselect("비교할 Rule 선택", [30, 60, 90, 110, 184], default=[30, 110])
    width = st.slider("비교용 셀 수", 30, 201, 101, key="cmp_width")
    generations = st.slider("비교용 세대 수", 10, 100, 40, key="cmp_gen")
    for rule in selected_rules:
        grid = run_1d_automaton(rule, width, generations)
        fig = px.imshow(grid, color_continuous_scale="gray", title=f"Rule {rule}")
        st.plotly_chart(fig, use_container_width=True, key=f"rule_comparison_{rule}")

with tabs[4]:
    st.subheader("📖 이론 요약")
    st.markdown("""
### Conway의 생명 게임 규칙:
- 이웃이 2~3명: 생존
- 정확히 3명: 새로운 생명 탄생
- 1명 이하 or 4명 이상: 죽음

### 오토마타 분류:
- Class 1: 모두 멈춤
- Class 2: 반복 무늬
- Class 3: 무작위 혼돈
- Class 4: 복잡성 (계산 가능성 포함)
""")

with tabs[5]:
    st.subheader("🧪 패턴 실험실")
    pattern_choice = st.selectbox("초기 패턴", list(predefined_patterns.keys()))
    rows = st.slider("격자 행 수", 10, 100, 30, key="custom_row")
    cols = st.slider("격자 열 수", 10, 100, 30, key="custom_col")
    generations = st.slider("세대 수", 5, 100, 30, key="custom_gen")

    pattern = predefined_patterns[pattern_choice]
    grid = np.zeros((rows, cols), dtype=np.uint8)
    start_x = rows // 2 - pattern.shape[0] // 2
    start_y = cols // 2 - pattern.shape[1] // 2
    grid[start_x:start_x + pattern.shape[0], start_y:start_y + pattern.shape[1]] = pattern

    history = simulate_custom_life(grid, generations)
    fig = generate_2d_animation(history)
    st.plotly_chart(fig, use_container_width=True, key="pattern_lab")

