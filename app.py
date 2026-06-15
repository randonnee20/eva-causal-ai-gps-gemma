"""
Causal AI Pipeline - LBS App Privacy Risk Analysis
위치정보서비스 앱 인지된 위험 인과 추론 파이프라인
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx
import warnings
import os
import streamlit.components.v1 as components
import explain

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Causal AI Pipeline - LBS Risk",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Minimal CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1100px; }
    h1 { font-size: 1.4rem; font-weight: 600; margin-bottom: 0.2rem; }
    h2 { font-size: 1.1rem; font-weight: 600; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; margin-top: 1.5rem; }
    h3 { font-size: 0.95rem; font-weight: 600; margin-top: 1rem; }
    .status-ok  { color: #2d6a4f; font-weight: 600; }
    .status-warn { color: #b5461a; font-weight: 600; }
    .info-box {
        background: #f5f5f5;
        border-left: 3px solid #999;
        padding: 0.6rem 0.9rem;
        font-size: 0.85rem;
        margin: 0.5rem 0;
        border-radius: 0 3px 3px 0;
    }
    .metric-card {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 0.7rem 1rem;
        text-align: center;
    }
    .metric-label { font-size: 0.75rem; color: #666; margin-bottom: 2px; }
    .metric-value { font-size: 1.3rem; font-weight: 700; color: #1a1a2e; }
    div[data-testid="stSidebar"] { background: #f8f8f8; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Korean font
# ---------------------------------------------------------------------------
@st.cache_resource
def setup_korean_font():
    font_candidates = ["NanumGothic", "Malgun Gothic", "AppleGothic", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in font_candidates:
        if font in available:
            matplotlib.rcParams["font.family"] = font
            matplotlib.rcParams["axes.unicode_minus"] = False
            return font
    matplotlib.rcParams["axes.unicode_minus"] = False
    return "default"

setup_korean_font()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "lbs_survey_332.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="cp949")
    df.columns = df.columns.str.strip()
    rename_map = {
        df.columns[0]: "gender",
        df.columns[1]: "age",
        df.columns[2]: "job",
        df.columns[3]: "it_field",
        df.columns[4]: "smartphone_years",
        df.columns[5]: "lbs_used_3mo",
        df.columns[6]: "lbs_app_count",
        df.columns[7]: "lbs_app_type",
        df.columns[8]: "lbs_motive",
    }
    df = df.rename(columns=rename_map)
    numeric_cols = [c for c in df.columns if c not in rename_map.values()]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Uavg", "IFavg", "SIavg", "PRavg", "SQavg", "USEavg", "ICUavg"])
    return df

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Causal AI Pipeline")
    st.markdown("LBS 앱 위험 인과 추론")
    st.divider()
    step = st.radio(
        "Step",
        options=[
            "Step 0  |  Data Overview",
            "Step 1  |  Variable Definition",
            "Step 2  |  DAG Construction",
            "Step 3  |  Identification (DoWhy)",
            "Step 4  |  Estimation (EconML)",
            "Step 5  |  Refutation",
            "Step 6  |  Counterfactual",
            "Step 7  |  H1-H5 인과 분석",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.session_state["ai_explain_on"] = st.toggle("AI 해석 자동 표시", value=True,
        help="각 Step 결과 아래에 비전공자용 해석을 자동으로 보여줍니다.")
    _ai = explain.llm.usage_info()
    if explain.llm.detect_mode() == "gemini":
        st.caption(f"AI 엔진: {_ai['mode']}  ·  오늘 {_ai['count']}/{_ai['limit']}회")
    else:
        st.caption(f"AI 엔진: {_ai['mode']}")
    st.divider()
    st.markdown("<small>Data: LBS Survey n=332 (2013)</small>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
try:
    df = load_data()
except Exception as e:
    st.error(f"Data load failed: {e}")
    st.stop()


# ===========================================================================
# STEP 0 - Data Overview
# ===========================================================================
if step.startswith("Step 0"):
    st.markdown("## Step 0 - Data Overview  /  데이터 개요")
    st.markdown(
        '<div class="info-box">원본 CSV(332행, 53컬럼) 변수 구조와 기본 통계를 확인한다.<br>'
        'Review variable structure and descriptive stats from raw CSV (332 rows, 53 cols).</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="metric-card"><div class="metric-label">Respondents / 응답자</div><div class="metric-value">332</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="metric-card"><div class="metric-label">Variables / 변수</div><div class="metric-value">53</div></div>', unsafe_allow_html=True)
    c3.markdown('<div class="metric-card"><div class="metric-label">Scale / 척도</div><div class="metric-value">7-pt Likert</div></div>', unsafe_allow_html=True)

    st.markdown("### Variable Map  /  변수 구성")
    var_info = pd.DataFrame({
        "Group": ["U", "SQ", "IF", "SI", "PR", "USE", "ICU"],
        "Label (KR)": ["유용성", "시스템 품질", "인터페이스 품질", "사회적 영향", "인지된 위험", "사용의도", "지속사용의도"],
        "Label (EN)": ["Usefulness", "System Quality", "Interface Quality", "Social Influence", "Perceived Risk", "Use Intention", "Continuance Intention"],
        "Items": [5, 5, 5, 4, 6, 5, 5],
        "Original Role": ["Independent", "Independent", "Independent", "Independent", "Moderator", "Dependent", "Dependent"],
        "Causal Role (Step 7)": ["Treatment T (H1)", "Treatment T (H2)", "Treatment T (H3)", "Treatment T (H4)", "Confounder W", "Outcome Y1", "Outcome Y2"],
        "Causal Role (Step 1-6)": ["Confounder W", "Confounder W", "Confounder W", "Confounder W", "Treatment T (선택)", "Outcome Y1", "Outcome Y2"],
        "Mean": [round(df[c].mean(), 2) for c in ["Uavg", "SQavg", "IFavg", "SIavg", "PRavg", "USEavg", "ICUavg"]],
    })
    st.dataframe(var_info, use_container_width=True, hide_index=True)

    st.markdown("### Descriptive Statistics  /  기술통계")
    avg_cols = ["Uavg", "SQavg", "IFavg", "SIavg", "PRavg", "USEavg", "ICUavg"]
    st.dataframe(df[avg_cols].describe().round(2), use_container_width=True)

    st.markdown("### Binary Treatment Variables  /  이진 처치 변수")
    col1, col2 = st.columns(2)
    with col1:
        cnt = df["PR_3.5PR"].value_counts().sort_index()
        st.markdown("**PR_3.5PR**  (척도 중간점 Scale Midpoint 3.5 기준 이진화, 불균형 63:269)")
        fig_b1, ax_b1 = plt.subplots(figsize=(3, 2.5))
        ax_b1.bar(["0 (낮음)", "1 (높음)"], [cnt.get(0, 0), cnt.get(1, 0)], color="#4c78a8")
        for i, v in enumerate([cnt.get(0, 0), cnt.get(1, 0)]):
            ax_b1.text(i, v + 3, str(v), ha="center", fontsize=10)
        ax_b1.set_ylabel("count")
        ax_b1.set_ylim(0, max(cnt.get(0,0), cnt.get(1,0)) * 1.15)
        plt.tight_layout()
        st.pyplot(fig_b1)
        plt.close(fig_b1)
    with col2:
        cnt2 = df["PR_2.6PR"].value_counts().sort_index()
        st.markdown("**PR_2.6PR**  (중앙값 Median 2.6 기준 이진화, 균형)")
        fig_b2, ax_b2 = plt.subplots(figsize=(3, 2.5))
        ax_b2.bar(["0 (낮음)", "1 (높음)"], [cnt2.get(0, 0), cnt2.get(1, 0)], color="#4c78a8")
        for i, v in enumerate([cnt2.get(0, 0), cnt2.get(1, 0)]):
            ax_b2.text(i, v + 3, str(v), ha="center", fontsize=10)
        ax_b2.set_ylabel("count")
        ax_b2.set_ylim(0, max(cnt2.get(0,0), cnt2.get(1,0)) * 1.15)
        plt.tight_layout()
        st.pyplot(fig_b2)
        plt.close(fig_b2)

    st.markdown("### Correlation Matrix  /  상관행렬")
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = df[avg_cols].corr()
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(avg_cols)))
    ax.set_yticks(range(len(avg_cols)))
    ax.set_xticklabels(avg_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(avg_cols, fontsize=9)
    for i in range(len(avg_cols)):
        for j in range(len(avg_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlation Matrix", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- AI 해석 ---
    _means = "; ".join(f"{explain.vlabel(c)} {df[c].mean():.2f}점" for c in avg_cols)
    _corr_pr_use = df["PRavg"].corr(df["USEavg"])
    _corr_u_use = df["Uavg"].corr(df["USEavg"])
    explain.show(
        "step0_overview",
        f"위치정보(LBS) 앱 이용자 332명에게 7점 척도로 물은 설문이다. 항목별 평균은 다음과 같다: {_means}. "
        f"상관관계를 보면, '인지된 위험'과 '사용의도'는 {_corr_pr_use:+.2f}(위험을 크게 느낄수록 쓰려는 마음이 {'줄어드는' if _corr_pr_use<0 else '느는'} 쪽), "
        f"'유용성'과 '사용의도'는 {_corr_u_use:+.2f}로 비교적 강하게 같이 움직인다. "
        f"이 데이터가 전반적으로 어떤 모습인지, 특히 '인지된 위험'과 '유용성'이 '사용의도'와 어떻게 함께 움직이는지 2~3문장으로 설명하라. "
        f"단 이것은 아직 '상관'일 뿐 '인과'는 아님을 한 번 짚어라.",
    )


# ===========================================================================
# STEP 1 - Variable Definition
# ===========================================================================
elif step.startswith("Step 1"):
    st.markdown("## Step 1 - Variable Definition  /  변수 정의")
    st.markdown(
        '<div class="info-box">'
        'T(처치), Y(결과), W(혼란변수)를 선택한다. T는 복수 선택 가능 — Step 4에서 모든 T에 대해 ATE를 비교 추정한다.<br>'
        'Steps 2·3·5·6은 첫 번째 선택 T 기준으로 동작한다.  Step 7은 H1-H5 고정 전체 분석.'
        '</div>',
        unsafe_allow_html=True,
    )

    # 변수 설명 테이블 (간결)
    var_ref = pd.DataFrame({
        "변수": ["Uavg","SQavg","IFavg","SIavg","PRavg","USEavg","ICUavg"],
        "설명 (KR)": ["유용성","시스템품질","인터페이스품질","사회적영향","인지된위험","사용의도","지속사용의도"],
        "유형": ["연속","연속","연속","연속","연속/이진","연속","연속"],
        "기본 역할": ["T (H1)","T (H2)","T (H3)","T (H4)","W 교란변수","T (H5) / Y","Y"],
    })
    st.dataframe(var_ref, use_container_width=True, hide_index=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Treatment T  /  처치 변수  (복수 선택 가능)")
        T_DEFS = {
            "Uavg":     ("유용성",          "H1", "연속형"),
            "SQavg":    ("시스템품질",       "H2", "연속형"),
            "IFavg":    ("인터페이스품질",   "H3", "연속형"),
            "SIavg":    ("사회적영향",       "H4", "연속형"),
            "USEavg":   ("사용의도",         "H5", "연속형"),
            "PR_2.6PR": ("인지된위험 이진",  "균형 156:176 권장", "이진형"),
            "PR_3.5PR": ("인지된위험 이진",  "불균형 62:270",     "이진형"),
            "PRavg":    ("인지된위험",       "연속형",            "연속형"),
        }
        # W에서 선택될 변수 미리 파악 (T와 겹치면 제외용)
        # 기본 선택: H1-H4 (Uavg, SQavg, IFavg, SIavg)
        selected_t_list = []
        for t_key, (kr, hyp, ttype) in T_DEFS.items():
            default_on = t_key in ["Uavg", "SQavg", "IFavg", "SIavg"]
            label_str = f"{t_key}  —  {kr}  ({hyp}, {ttype})"
            if st.checkbox(label_str, value=default_on, key=f"t_{t_key}"):
                selected_t_list.append(t_key)

        if not selected_t_list:
            st.warning("T를 최소 1개 선택하세요.")
            selected_t_list = ["Uavg"]

        # 이진형 여부 (복수 선택 시 첫 번째 기준)
        t_primary = selected_t_list[0]
        t_is_binary = t_primary in ["PR_2.6PR", "PR_3.5PR"]

        # session_state 저장
        st.session_state["T"] = t_primary          # 단일 T (Steps 2/3/5/6용)
        st.session_state["T_list"] = selected_t_list  # 복수 T (Step 4용)
        st.session_state["T_is_multi"] = len(selected_t_list) > 1

        if len(selected_t_list) > 1:
            st.markdown(
                f'<div class="info-box" style="font-size:0.82rem;border-left-color:#5578B8">'
                f'<b>{len(selected_t_list)}개 T 선택됨:</b> {", ".join(selected_t_list)}<br>'
                f'Step 4에서 전체 ATE 비교 실행. Steps 2·3·5·6은 <b>{t_primary}</b> 기준 동작.'
                f'</div>', unsafe_allow_html=True,
            )

        st.markdown("### Outcome Y  /  결과 변수")
        y_structure = st.radio(
            "Y structure",
            options=[
                "단일  |  T → 사용의도(USEavg)",
                "단일  |  T → 지속사용의도(ICUavg)",
                "매개  |  T → USEavg → ICUavg",
            ],
            index=0,
        )
        if "USEavg" in y_structure and "매개" not in y_structure:
            st.session_state["Y"] = "USEavg"
            st.session_state["Y2"] = None
            st.session_state["y_mode"] = "single"
        elif "ICUavg" in y_structure and "매개" not in y_structure:
            st.session_state["Y"] = "ICUavg"
            st.session_state["Y2"] = None
            st.session_state["y_mode"] = "single"
        else:
            st.session_state["Y"] = "USEavg"
            st.session_state["Y2"] = "ICUavg"
            st.session_state["y_mode"] = "mediation"

    with col2:
        st.markdown("### Confounders W  /  혼란변수  (T 선택 시 자동 제외)")
        W_ALL = {
            "Uavg":  "Uavg  — 유용성",
            "SQavg": "SQavg — 시스템품질",
            "IFavg": "IFavg — 인터페이스품질",
            "SIavg": "SIavg — 사회적영향",
            "PRavg": "PRavg — 인지된위험 (교란변수)",
        }
        selected_w = []
        for col_key, label in W_ALL.items():
            if col_key in selected_t_list:
                st.checkbox(f"{label}  [T 선택됨 — 제외]", value=False, disabled=True, key=f"w_{col_key}")
            else:
                # PRavg: T에 PR계열이 없으면 기본 체크
                pr_as_t = any(k in selected_t_list for k in ["PR_2.6PR","PR_3.5PR","PRavg"])
                default_w = True if col_key != "PRavg" or not pr_as_t else False
                if st.checkbox(f"{label}", value=default_w, key=f"w_{col_key}"):
                    selected_w.append(col_key)
        st.session_state["W"] = selected_w

        st.markdown("### W→T 관계  /  DAG Direction")
        wt_mode = st.radio(
            "W-T",
            options=[
                "W → T  (교란변수 구조, H1-H5 권장)",
                "W, T 독립  (병렬 선행변수)",
            ],
            index=0,
        )
        st.session_state["wt_mode"] = wt_mode
        if "W → T" in wt_mode:
            st.markdown(
                '<div class="info-box" style="font-size:0.8rem;border-left-color:#2d6a4f">'
                '<b>W → T 구조:</b> PRavg가 W에 포함된 경우 PRavg → T, PRavg → Y 화살표 자동 생성.'
                '</div>', unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="info-box" style="font-size:0.8rem">'
                '<b>W, T 독립:</b> W와 T가 병렬 선행변수. T=PR 계열 선택 시 사용.'
                '</div>', unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("### 설정 확인  /  Confirmed Setup")
    y_mode = st.session_state.get("y_mode", "single")
    y_display = st.session_state.get("Y", "—")
    if y_mode == "mediation":
        y_display = "USEavg → ICUavg"
    t_display = ", ".join(selected_t_list) if selected_t_list else "—"
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-label">T (처치)</div><div class="metric-value" style="font-size:0.78rem">{t_display}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-label">T 수</div><div class="metric-value">{len(selected_t_list)}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-label">Y (결과)</div><div class="metric-value" style="font-size:0.9rem">{y_display}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-label">W (교란변수)</div><div class="metric-value" style="font-size:0.75rem">{", ".join(selected_w)}</div></div>', unsafe_allow_html=True)

    # --- AI 해석 (선택에 따라 실시간) ---
    _t_kr = explain.vlabels(selected_t_list)
    if y_mode == "mediation":
        _y_kr = "사용의도 → 지속사용의도(매개)"
    else:
        _y_kr = explain.vlabel(st.session_state.get("Y", "USEavg"))
    _w_kr = explain.vlabels(selected_w)
    explain.show(
        f"step1_{t_display}_{y_display}_{','.join(selected_w)}_{y_mode}",
        f"사용자가 인과분석 설정을 다음과 같이 골랐다. "
        f"원인(처치)으로 알아볼 것 = [{_t_kr}]. "
        f"측정할 목표(결과) = {_y_kr}. "
        f"배경 요인으로 통제할 교란변수 = [{_w_kr}]. "
        f"이 설정이 '무엇이 사용의도에 영향을 주는지'를 어떻게 알아보려는 것인지 2~3문장으로 설명하라. "
        f"원인 후보가 여러 개면 '여러 원인을 동시에 비교하는 설정'이라고 알려주고, "
        f"교란변수는 '결과에 영향을 줄 수 있는 배경 요인이라 미리 통제하는 것'이라고 풀어라.",
    )


# ===========================================================================
# STEP 2 - DAG Construction
# ===========================================================================
elif step.startswith("Step 2"):
    st.markdown("## Step 2 - DAG Construction  /  인과 구조 그래프")
    st.markdown(
        '<div class="info-box">'
        '인과 구조 DAG를 구성한다. Step 1 설정이 반영된다.<br>'
        '화살표 추가/제거로 수정 가능. 이 DAG가 DoWhy 식별의 입력이 된다.'
        '</div>',
        unsafe_allow_html=True,
    )

    T = st.session_state.get("T", "PR_2.6PR")
    T_list = st.session_state.get("T_list", [T])
    Y = st.session_state.get("Y", "USEavg")
    Y2 = st.session_state.get("Y2", None)
    W = st.session_state.get("W", ["Uavg", "SQavg", "IFavg", "SIavg"])
    wt_mode = st.session_state.get("wt_mode", "W, T 독립  (원래 연구 모형 - 권장)")
    if len(T_list) > 1:
        st.info(f"T {len(T_list)}개 선택됨 ({', '.join(T_list)}) — DAG는 기본 T={T} 기준으로 표시됩니다.")
    y_mode = st.session_state.get("y_mode", "single")

    wt_label = "W → T 포함" if "W → T" in wt_mode else "W, T 독립"
    y_label = "T → USEavg → ICUavg (매개)" if y_mode == "mediation" else f"T → {Y}"
    st.markdown(
        f'<div class="info-box">'
        f'현재 설정: <b>W-T 관계 = {wt_label}</b>  /  <b>Y 구조 = {y_label}</b><br>'
        f'Step 1에서 변경 가능. Reset 버튼으로 현재 설정 기준 기본 DAG 재생성.'
        f'</div>',
        unsafe_allow_html=True,
    )

    def build_default_edges(W, T_list, wt_mode, y_mode):
        edges = []
        for t in T_list:
            for w in W:
                if (w, "USEavg") not in edges:
                    edges.append((w, "USEavg"))
                if "W → T" in wt_mode:
                    edges.append((w, t))
            edges.append((t, "USEavg"))
        edges.append(("USEavg", "ICUavg"))
        return list(dict.fromkeys(edges))

    default_edges = build_default_edges(W, T_list, wt_mode, y_mode)

    col_r, _ = st.columns([1, 4])
    with col_r:
        if st.button("Reset to Default  /  기본값으로"):
            st.session_state["dag_edges"] = default_edges.copy()
            st.rerun()

    if "dag_edges" not in st.session_state:
        st.session_state["dag_edges"] = default_edges.copy()

    edges = st.session_state["dag_edges"]

    st.markdown("#### Current Edges  /  현재 엣지 목록")
    if edges:
        edge_cols = st.columns(4)
        for i, (src, tgt) in enumerate(edges):
            with edge_cols[i % 4]:
                if st.button(f"Remove: {src} -> {tgt}", key=f"rm_{i}"):
                    st.session_state["dag_edges"].pop(i)
                    st.rerun()

    st.markdown("#### Add Edge  /  엣지 추가")
    all_nodes = sorted(set([n for e in edges for n in e]) | set(W) | set(T_list) | {"USEavg", "ICUavg"})
    ac1, ac2, ac3 = st.columns([2, 2, 1])
    with ac1:
        new_src = st.selectbox("From", all_nodes, key="new_src")
    with ac2:
        new_tgt = st.selectbox("To", all_nodes, key="new_tgt")
    with ac3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Add  /  추가"):
            if new_src != new_tgt and (new_src, new_tgt) not in st.session_state["dag_edges"]:
                st.session_state["dag_edges"].append((new_src, new_tgt))
                st.rerun()

    # Draw DAG — pyvis (interactive, draggable)
    st.markdown("#### DAG Visualization  (노드를 드래그해서 위치 조정 가능)")
    try:
        from pyvis.network import Network as PVNet
        import tempfile, os as _os

        pv = PVNet(height="440px", width="100%", directed=True, bgcolor="#f9f9f9", font_color="#222")
        pv.set_options('''
{
  "physics": {
    "enabled": true,
    "barnesHut": {"gravitationalConstant": -4000, "centralGravity": 0.3, "springLength": 160, "damping": 0.18},
    "stabilization": {"iterations": 120, "updateInterval": 25}
  },
  "interaction": {"dragNodes": true, "zoomView": true},
  "edges": {"arrows": {"to": {"enabled": true, "scaleFactor": 0.7}}, "color": "#555", "width": 2, "smooth": {"type": "dynamic"}},
  "nodes": {"borderWidth": 1, "shadow": true}
}
''')

        y_nodes = {Y, "ICUavg", "USEavg"} if y_mode == "mediation" else {Y}
        dag_nodes = set(n for e in edges for n in e) | set(T_list) | set(W) | y_nodes

        for node in dag_nodes:
            if node in T_list:
                color, font_c, title = "#1a1a2e", "white", f"T (처치변수)"
            elif node in y_nodes:
                color, font_c, title = "#2d6a4f", "white", f"Y (결과변수)"
            elif node in W:
                color, font_c, title = "#6b6b8a", "white", f"W (교란변수)"
            else:
                color, font_c, title = "#aaaaaa", "#222", node
            pv.add_node(node, label=node, color={"background": color, "border": "#fff"},
                        font={"color": font_c, "size": 12, "bold": True},
                        size=30, title=title)

        for src, tgt in edges:
            pv.add_edge(src, tgt)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as tf:
            pv.save_graph(tf.name)
            tf_path = tf.name
        with open(tf_path, "r", encoding="utf-8") as f_html:
            html_dag = f_html.read()
        _os.unlink(tf_path)

        # legend 아래 표시
        t_labels = ", ".join(T_list)
        st.markdown(
            f'<div class="info-box">' +
            f'<b>T (처치)</b>: {t_labels} &nbsp;|&nbsp; ' +
            f'<b>Y (결과)</b>: {", ".join(y_nodes)} &nbsp;|&nbsp; ' +
            f'<b>W (교란)</b>: {", ".join(W) if W else "없음"}' +
            '</div>', unsafe_allow_html=True
        )
        components.html(html_dag, height=460, scrolling=False)

    except ImportError:
        st.error("pyvis 미설치. 터미널에서 실행: pip install pyvis")
    except Exception as _pve:
        st.warning(f"pyvis 렌더링 실패: {_pve}  — matplotlib fallback")
        G = nx.DiGraph()
        G.add_edges_from(edges)
        fig, ax = plt.subplots(figsize=(9, 5))
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=2.5)
        node_colors = []
        for n in G.nodes():
            if n in T_list: node_colors.append("#1a1a2e")
            elif n in {Y, "ICUavg", "USEavg"}: node_colors.append("#2d6a4f")
            elif n in W: node_colors.append("#6b6b8a")
            else: node_colors.append("#aaaaaa")
        nx.draw_networkx(G, pos=pos, ax=ax, node_color=node_colors, node_size=1600,
            font_color="white", font_size=8, font_weight="bold",
            arrows=True, arrowsize=20, edge_color="#444", width=1.5)
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Build GML for DoWhy
    gml_lines = ["digraph {"]
    for src, tgt in edges:
        gml_lines.append(f'  "{src}" -> "{tgt}";')
    gml_lines.append("}")
    dag_str = "\n".join(gml_lines)
    st.session_state["dag_str"] = dag_str

    with st.expander("DAG (GML string for DoWhy)"):
        st.code(dag_str, language="dot")

    # --- AI 해석 ---
    _edge_desc = ", ".join([f"{explain.vlabel(s)}→{explain.vlabel(t)}" for s, t in edges]) if edges else "없음"
    explain.show(
        f"step2_{_edge_desc}",
        f"이 위치정보(LBS) 앱 설문 분석의 인과 구조 그래프(DAG)를 그렸다. 설정한 화살표(원인→결과) 목록은 다음과 같다: {_edge_desc}. "
        f"이 화살표들이 '무엇이 무엇의 원인이라고 가정했는지'를 일반인에게 2~3문장으로 설명하라. "
        f"반드시 위 목록에 있는 변수만 언급하고, 목록에 없는 소재는 절대 만들지 마라. "
        f"이 가정 그림이 앞으로 효과를 계산하는 출발점이 된다는 점을 알려라.",
    )


# ===========================================================================
# STEP 3 - Identification (DoWhy)
# ===========================================================================
elif step.startswith("Step 3"):
    st.markdown("## Step 3 - Identification  /  식별 (DoWhy)")
    st.markdown(
        '<div class="info-box">'
        'DoWhy로 backdoor 기준을 적용해 인과효과 E[Y|do(T)]를 식별한다.<br>'
        'DoWhy identifies the causal estimand using backdoor criterion.'
        '</div>',
        unsafe_allow_html=True,
    )

    T = st.session_state.get("T", "PR_2.6PR")
    T_list = st.session_state.get("T_list", [T])
    Y = st.session_state.get("Y", "USEavg")
    W = st.session_state.get("W", [])
    dag_str = st.session_state.get("dag_str", "")
    if len(T_list) > 1:
        st.info(f"T {len(T_list)}개 선택됨 — 식별은 기본 T={T} 기준. 전체 ATE 비교는 Step 4에서 확인.")

    if not dag_str:
        st.warning("Step 2에서 DAG를 먼저 구성해주세요.")
        st.stop()

    if len(T_list) > 1:
        st.markdown(
            '<div class="info-box">' +
            f'<b>다중 T 식별 모드 ({len(T_list)}개):</b> {", ".join(T_list)}<br>' +
            '같은 W 세트로 각 T에 대해 backdoor criterion을 순차 검증한다.' +
            '</div>', unsafe_allow_html=True
        )

    if st.button("Run Identification  /  식별 실행"):
        try:
            from dowhy import CausalModel

            id_results = []
            prog_id = st.progress(0, text="식별 진행 중...")
            with st.spinner("DoWhy identification running..."):
                for idx_t, t_i in enumerate(T_list):
                    try:
                        model_i = CausalModel(data=df, treatment=t_i, outcome=Y, graph=dag_str)
                        est_i = model_i.identify_effect(proceed_when_unidentifiable=True)
                        # 첫 번째 T를 primary로 저장
                        if t_i == T:
                            st.session_state["dowhy_model"] = model_i
                            st.session_state["identified_estimand"] = est_i
                        try:
                            bd_i = est_i.get_backdoor_variables()
                            bd_str = str(bd_i) if bd_i else "없음"
                        except Exception:
                            bd_str = "확인 불가"
                        id_results.append({
                            "T (처치)": t_i,
                            "Y (결과)": Y,
                            "Backdoor set (W)": bd_str,
                            "backdoor 기준": "충족" if bd_i else "미충족",
                        })
                    except Exception as e_i:
                        id_results.append({"T (처치)": t_i, "Y (결과)": Y,
                            "Backdoor set (W)": f"ERROR: {str(e_i)[:40]}", "backdoor 기준": "오류"})
                    prog_id.progress((idx_t+1)/len(T_list), text=f"{t_i} 식별 완료 ({idx_t+1}/{len(T_list)})")
            prog_id.empty()

            st.success("식별 완료 / Identification complete")
            st.markdown("### 식별 결과 요약  /  Identification Summary")
            st.dataframe(pd.DataFrame(id_results), use_container_width=True, hide_index=True)

            st.markdown(
                '<div class="info-box">' +
                '백도어 기준 충족: W를 conditioning하면 T←W→Y 교란 경로가 차단된다.<br>' +
                'Backdoor criterion: conditioning on W blocks the T←W→Y confounding path.<br>' +
                '<b>※ W 세트가 동일하므로 모든 T에 동일하게 적용된다.</b>' +
                '</div>', unsafe_allow_html=True
            )

            # 첫 번째 T estimand 상세 표시
            if "identified_estimand" in st.session_state:
                with st.expander(f"Estimand 상세 (T={T})"):
                    st.code(str(st.session_state["identified_estimand"]))

            # --- AI 해석 ---
            _ok = all(r.get("backdoor 기준") == "충족" for r in id_results)
            _t_kr = explain.vlabels([r["T (처치)"] for r in id_results])
            _y_kr = explain.vlabel(Y)
            explain.show(
                f"step3_{[r['T (처치)'] for r in id_results]}_{Y}",
                f"원인 후보 [{_t_kr}]가 결과 '{_y_kr}'에 주는 순수한 효과를 계산할 수 있는지 점검하는 '식별' 단계를 수행했고, "
                f"결과는 '{'모두 통과' if _ok else '일부 미통과'}'다. "
                f"'식별을 통과했다'는 것이 '배경 요인(교란변수)을 통제하면 다른 영향에 휘둘리지 않고 원인의 순수한 효과만 가려낼 수 있다는 뜻'임을, "
                f"위 변수 이름만 써서 일반인에게 2~3문장으로 설명하라. 가격·할인·매출 같은 없는 소재는 절대 만들지 마라.",
            )

        except ImportError:
            st.error("DoWhy not installed. Run: pip install dowhy")
        except Exception as e:
            st.error(f"Identification error: {e}")
    else:
        if "identified_estimand" in st.session_state:
            st.markdown('<span class="status-ok">식별 결과 있음 / Already identified</span>', unsafe_allow_html=True)
            st.code(str(st.session_state["identified_estimand"]))
        else:
            st.info("버튼을 눌러 식별을 실행하세요.")


# ===========================================================================
# STEP 4 - Estimation (EconML)
# ===========================================================================
elif step.startswith("Step 4"):
    st.markdown("## Step 4 - Estimation  /  효과 추정 (EconML)")

    T = st.session_state.get("T", "PR_2.6PR")
    T_list = st.session_state.get("T_list", [T])
    Y = st.session_state.get("Y", "USEavg")
    Y2 = st.session_state.get("Y2", None)
    y_mode = st.session_state.get("y_mode", "single")
    W = st.session_state.get("W", ["Uavg", "SQavg", "IFavg", "SIavg"])
    is_multi_t = len(T_list) > 1

    if is_multi_t:
        st.markdown(
            f'<div class="info-box">'
            f'<b>다중 T 비교 모드 ({len(T_list)}개):</b> {", ".join(T_list)}<br>'
            f'각 T에 대해 CausalForestDML을 실행하고 ATE를 비교한다. W={W}, Y={Y}'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif y_mode == "mediation":
        st.markdown(
            f'<div class="info-box">'
            f'<b>매개 분석 모드 (Mediation Analysis)</b><br>'
            f'Path 1: T({T}) → Y1(사용의도)  ATE1 추정<br>'
            f'Path 2: Y1(사용의도) → Y2(지속사용의도)  ATE2 추정<br>'
            f'총 간접 효과 = ATE1 x ATE2'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="info-box">'
            f'EconML CausalForestDML로 T={T} → Y={Y} ATE/CATE를 추정한다.'
            f'</div>',
            unsafe_allow_html=True,
        )

    col_est, col_param = st.columns([3, 1])
    with col_param:
        st.markdown("### Parameters")
        n_estimators = st.slider("n_estimators", 50, 500, 200, 50)
        min_samples = st.slider("min_samples_leaf", 5, 50, 10, 5)
        cv_folds = st.slider("cv (cross-fit)", 2, 5, 3)

    with col_est:
        if st.button("Run Estimation  /  추정 실행"):
            with st.spinner("EconML estimation running... (약 30-60초 소요)"):
                try:
                    from econml.dml import CausalForestDML
                    from sklearn.ensemble import GradientBoostingRegressor
                    from sklearn.linear_model import LogisticRegression

                    # ── 다중 T 모드 ──────────────────────────────────────
                    if is_multi_t:
                        X_multi = df[W].values if W else np.ones((len(df), 1))
                        multi_results = []
                        multi_cate = {}
                        estimator_map = {}
                        prog_m = st.progress(0, text="다중 T 추정 중...")
                        for mi, t_i in enumerate(T_list):
                            try:
                                tv = df[t_i].values.astype(float)
                                yv = df[Y].values.astype(float)
                                bflag = set(np.unique(tv)).issubset({0.0, 1.0})
                                m_t = LogisticRegression(max_iter=500) if bflag else GradientBoostingRegressor(n_estimators=100)
                                est_m = CausalForestDML(
                                    model_y=GradientBoostingRegressor(n_estimators=100),
                                    model_t=m_t,
                                    n_estimators=n_estimators,
                                    min_samples_leaf=min_samples,
                                    cv=cv_folds,
                                    random_state=42,
                                    discrete_treatment=bflag,
                                )
                                est_m.fit(yv, tv, X=X_multi)
                                ate_m = float(est_m.ate(X_multi))
                                lb_m, ub_m = est_m.ate_interval(X_multi, alpha=0.05)
                                cate_m = est_m.effect(X_multi)
                                multi_cate[t_i] = (cate_m, ate_m, float(lb_m), float(ub_m))
                                estimator_map[t_i] = est_m
                                sig_m = "유의" if lb_m * ub_m > 0 else "비유의"
                                multi_results.append({
                                    "T (처치)": t_i,
                                    "Y (결과)": Y,
                                    "ATE": round(ate_m, 4),
                                    "CI 하한": round(float(lb_m), 4),
                                    "CI 상한": round(float(ub_m), 4),
                                    "방향": "+" if ate_m >= 0 else "-",
                                    "유의성 (95%)": sig_m,
                                    "CATE 평균": round(float(cate_m.mean()), 4),
                                    "CATE SD": round(float(cate_m.std()), 4),
                                })
                                # 첫 번째 T는 단일 분석용으로도 저장
                                if mi == 0:
                                    st.session_state["estimator"] = est_m
                                    st.session_state["cate"] = cate_m
                                    st.session_state["ate"] = ate_m
                                    st.session_state["ate_ci"] = (float(lb_m), float(ub_m))
                            except Exception as e_m:
                                multi_results.append({
                                    "T (처치)": t_i, "Y (결과)": Y,
                                    "ATE": "ERROR", "CI 하한": str(e_m)[:30],
                                    "CI 상한": "", "방향": "", "유의성 (95%)": "",
                                    "CATE 평균": "", "CATE SD": "",
                                })
                            prog_m.progress((mi+1)/len(T_list), text=f"{t_i} 완료 ({mi+1}/{len(T_list)})")
                        prog_m.empty()
                        st.session_state["multi_results"] = multi_results
                        st.session_state["multi_cate"] = multi_cate
                        st.session_state["estimator_map"] = estimator_map
                        st.success("다중 T 추정 완료")

                        # 비교 테이블
                        st.markdown("### ATE 비교 테이블  /  Multi-T Comparison")
                        mr_df = pd.DataFrame(multi_results)
                        def hl_sig(row):
                            c = "background-color: #d1efe0" if row.get("유의성 (95%)") == "유의" else "background-color: #fae3d0"
                            return [c]*len(row)
                        try:
                            st.dataframe(mr_df.style.apply(hl_sig, axis=1), use_container_width=True, hide_index=True)
                        except Exception:
                            st.dataframe(mr_df, use_container_width=True, hide_index=True)

                        # ATE 비교 바 차트
                        valid_m = [r for r in multi_results if isinstance(r["ATE"], float)]
                        if valid_m:
                            fig_mb, ax_mb = plt.subplots(figsize=(max(6, len(valid_m)*1.5), 4))
                            xlabels = [r["T (처치)"] for r in valid_m]
                            ates_m  = [r["ATE"] for r in valid_m]
                            lbs_m   = [r["CI 하한"] for r in valid_m]
                            ubs_m   = [r["CI 상한"] for r in valid_m]
                            bc_m = ["#2d6a4f" if l*u > 0 else "#b07040" for l,u in zip(lbs_m,ubs_m)]
                            bars_m = ax_mb.bar(xlabels, ates_m, color=bc_m, edgecolor="white", width=0.55)
                            for br, lb, ub in zip(bars_m, lbs_m, ubs_m):
                                x = br.get_x() + br.get_width()/2
                                ax_mb.plot([x,x],[lb,ub], color="#444", lw=2)
                            ax_mb.axhline(0, color="#888", lw=1, ls="--")
                            ax_mb.set_ylabel(f"ATE (T 1단위 → {Y} 변화량)")
                            ax_mb.set_title(f"다중 T ATE 비교  (W={W}, Y={Y})", fontsize=10)
                            gp = plt.Rectangle((0,0),1,1,color="#2d6a4f",label="95% CI 유의")
                            ap = plt.Rectangle((0,0),1,1,color="#b07040",label="95% CI 비유의")
                            ax_mb.legend(handles=[gp,ap],fontsize=8)
                            plt.tight_layout()
                            st.pyplot(fig_mb)
                            plt.close()

                        # CATE 분포 비교
                        if multi_cate:
                            n_mc = len(multi_cate)
                            fig_mc, axes_mc = plt.subplots(1, n_mc, figsize=(3.5*n_mc, 3.5))
                            if n_mc == 1: axes_mc = [axes_mc]
                            pal_m = ["#1a1a2e","#2d6a4f","#5855A0","#B07040","#3A6B9A","#6b6b8a","#c0392b","#2980b9"]
                            for ci, (t_k, (cate_k, ate_k, lb_k, ub_k)) in enumerate(multi_cate.items()):
                                axes_mc[ci].hist(cate_k, bins=25, color=pal_m[ci%len(pal_m)], edgecolor="white", alpha=0.85)
                                axes_mc[ci].axvline(ate_k, color="#c0392b", lw=2, ls="--", label=f"ATE={ate_k:.3f}")
                                axes_mc[ci].axvline(0, color="#aaa", lw=1, ls=":")
                                axes_mc[ci].set_title(f"T={t_k}", fontsize=9)
                                axes_mc[ci].set_xlabel("CATE", fontsize=8)
                                axes_mc[ci].legend(fontsize=7.5)
                            plt.suptitle(f"CATE 분포 비교  (Y={Y})", fontsize=10, y=1.02)
                            plt.tight_layout()
                            st.pyplot(fig_mc)
                            plt.close()

                        # --- AI 해석 (다중 T) ---
                        _y_kr = explain.vlabel(Y)
                        _facts = []
                        for r in multi_results:
                            if not isinstance(r["ATE"], float):
                                continue
                            _tk = explain.vlabel(r["T (처치)"])
                            if r["유의성 (95%)"] == "유의":
                                _dir = "높이면 늘어나는" if r["ATE"] >= 0 else "높이면 줄어드는"
                                _facts.append(f"'{_tk}'은(는) '{_y_kr}'을(를) {_dir} 확실한 효과가 있다(효과 {r['ATE']:+.2f})")
                            else:
                                _facts.append(f"'{_tk}'은(는) '{_y_kr}'에 뚜렷한 효과가 없다(추정치 {r['ATE']:+.2f}, 불확실)")
                        _facts_str = "; ".join(_facts)
                        explain.show(
                            f"step4_multi_{_facts_str}",
                            f"여러 원인을 동시에 비교해, 각각이 결과 '{_y_kr}'에 주는 순수 효과를 계산했다. 결과는 다음과 같다: {_facts_str}. "
                            f"이 사실을 그대로(원인↔결과 바꾸지 말고, 방향·확실성 그대로) 일반인에게 2~3문장으로 설명하라. "
                            f"어떤 요인이 확실하게 영향을 주고 어떤 것은 효과가 불확실한지 알려주고, 효과가 생긴 이유는 지어내지 마라.",
                        )
                        st.stop()

                    # ── 단일 T 모드 (기존 로직) ───────────────────────────
                    # W가 비어있으면 상수 열(1) 사용 — CausalForestDML은 X 최소 1열 필요
                    X = df[W].values if W else np.ones((len(df), 1))
                    t_vals = df[T].values.astype(float)
                    y_vals = df[Y].values.astype(float)

                    # T 타입 감지: 이진이면 LogisticRegression, 연속형이면 GBR
                    t_is_binary = set(np.unique(t_vals)).issubset({0.0, 1.0})
                    model_t = (
                        LogisticRegression(max_iter=500)
                        if t_is_binary
                        else GradientBoostingRegressor(n_estimators=100)
                    )

                    # Path 1: T -> Y1 (사용의도)
                    est1 = CausalForestDML(
                        model_y=GradientBoostingRegressor(n_estimators=100),
                        model_t=model_t,
                        n_estimators=n_estimators,
                        min_samples_leaf=min_samples,
                        cv=cv_folds,
                        random_state=42,
                        discrete_treatment=t_is_binary,
                    )
                    est1.fit(y_vals, t_vals, X=X)
                    ate1 = float(est1.ate(X))
                    ate1_lb, ate1_ub = est1.ate_interval(X, alpha=0.05)
                    cate1 = est1.effect(X)

                    st.session_state["estimator"] = est1
                    st.session_state["estimator_map"] = {T: est1}
                    st.session_state["cate"] = cate1
                    st.session_state["ate"] = ate1
                    st.session_state["ate_ci"] = (float(ate1_lb), float(ate1_ub))

                    # Path 2: Y1 -> Y2 (지속사용의도) - 매개 분석 모드만
                    ate2 = ate2_lb = ate2_ub = cate2 = None
                    if y_mode == "mediation" and Y2:
                        y2_vals = df[Y2].values.astype(float)
                        use_vals = df["USEavg"].values.astype(float)
                        est2 = CausalForestDML(
                            model_y=GradientBoostingRegressor(n_estimators=100),
                            model_t=GradientBoostingRegressor(n_estimators=100),
                            n_estimators=n_estimators,
                            min_samples_leaf=min_samples,
                            cv=cv_folds,
                            random_state=42,
                        )
                        est2.fit(y2_vals, use_vals, X=X)
                        ate2 = float(est2.ate(X))
                        ate2_lb, ate2_ub = est2.ate_interval(X, alpha=0.05)
                        cate2 = est2.effect(X)
                        st.session_state["estimator2"] = est2
                        st.session_state["cate2"] = cate2
                        st.session_state["ate2"] = ate2

                        st.success("추정 완료 / Estimation complete")

                    # --- 결과 출력 ---
                    if y_mode == "mediation" and ate2 is not None:
                        indirect = ate1 * ate2
                        st.markdown("### Path Analysis  /  경로 분석 결과")
                        pa1, pa2, pa3 = st.columns(3)
                        pa1.markdown(
                            f'<div class="metric-card">'
                            f'<div class="metric-label">Path 1: T({T}) → Y1(사용의도)</div>'
                            f'<div class="metric-value">{ate1:.4f}</div>'
                            f'<div class="metric-label">95% CI [{ate1_lb:.3f}, {ate1_ub:.3f}]</div>'
                            f'</div>', unsafe_allow_html=True)
                        pa2.markdown(
                            f'<div class="metric-card">'
                            f'<div class="metric-label">Path 2: Y1(사용의도) → Y2(지속사용의도)</div>'
                            f'<div class="metric-value">{ate2:.4f}</div>'
                            f'<div class="metric-label">95% CI [{ate2_lb:.3f}, {ate2_ub:.3f}]</div>'
                            f'</div>', unsafe_allow_html=True)
                        pa3.markdown(
                            f'<div class="metric-card">'
                            f'<div class="metric-label">총 간접 효과 / Indirect Effect</div>'
                            f'<div class="metric-value">{indirect:.4f}</div>'
                            f'<div class="metric-label">Path1 x Path2</div>'
                            f'</div>', unsafe_allow_html=True)
                        _t_kr_med = explain.vlabel(T)
                        st.markdown(
                            f'<div class="info-box">'
                            f'{_t_kr_med}(T)이 1단위 높아지면 사용의도(Y1)는 <b>{ate1:+.4f}</b> 변화 (Path 1).<br>'
                            f'사용의도(Y1)가 1단위 높아지면 지속사용의도(Y2)는 <b>{ate2:+.4f}</b> 변화 (Path 2).<br>'
                            f'T가 Y2에 미치는 총 간접 효과: <b>{indirect:+.4f}</b> (Path1 x Path2)<br>'
                            f'즉, {_t_kr_med}이 높아지면 사용의도를 거쳐 지속사용의도까지 연쇄적으로 영향을 준다.'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        m1, m2, m3 = st.columns(3)
                        m1.markdown(f'<div class="metric-card"><div class="metric-label">ATE  평균 처치 효과</div><div class="metric-value">{ate1:.4f}</div></div>', unsafe_allow_html=True)
                        m2.markdown(f'<div class="metric-card"><div class="metric-label">95% CI Lower</div><div class="metric-value">{ate1_lb:.4f}</div></div>', unsafe_allow_html=True)
                        m3.markdown(f'<div class="metric-card"><div class="metric-label">95% CI Upper</div><div class="metric-value">{ate1_ub:.4f}</div></div>', unsafe_allow_html=True)
                        direction_kr = "낮춘다" if ate1 < 0 else "높인다"
                        sig_kr = "0 미포함 - 통계적으로 유의" if ate1_lb * ate1_ub > 0 else "0 포함 - 5% 수준에서 유의하지 않음"
                        st.markdown(
                            f'<div class="info-box">'
                            f'T={T}가 {Y}를 평균 <b>{abs(ate1):.4f}</b>만큼 {direction_kr}.<br>'
                            f'95% CI: [{ate1_lb:.4f}, {ate1_ub:.4f}]  ({sig_kr})'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    # CATE 분포
                    st.markdown("### CATE Distribution  /  개별 처치 효과 분포  (Path 1: T → 사용의도)")
                    fig, ax = plt.subplots(figsize=(7, 3.5))
                    ax.hist(cate1, bins=30, color="#1a1a2e", edgecolor="white", alpha=0.85)
                    ax.axvline(ate1, color="#c0392b", lw=2, ls="--", label=f"ATE = {ate1:.3f}")
                    ax.axvline(0, color="#888", lw=1, ls=":")
                    ax.set_xlabel("CATE (개별 처치 효과)")
                    ax.set_ylabel("응답자 수")
                    ax.set_title(f"CATE: T={T} → {Y}", fontsize=10)
                    ax.legend(fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    if y_mode == "mediation" and cate2 is not None:
                        fig3, ax3 = plt.subplots(figsize=(7, 3.5))
                        ax3.hist(cate2, bins=30, color="#2d6a4f", edgecolor="white", alpha=0.85)
                        ax3.axvline(ate2, color="#c0392b", lw=2, ls="--", label=f"ATE = {ate2:.3f}")
                        ax3.axvline(0, color="#888", lw=1, ls=":")
                        ax3.set_xlabel("CATE (개별 처치 효과)")
                        ax3.set_ylabel("응답자 수")
                        ax3.set_title("CATE: 사용의도(Y1) → 지속사용의도(Y2)", fontsize=10)
                        ax3.legend(fontsize=9)
                        plt.tight_layout()
                        st.pyplot(fig3)
                        plt.close()

                    # CATE by W — W 없으면 T 값으로 대체 표시
                    st.markdown("### CATE by Confounder  /  혼란변수별 처치 효과")
                    plot_cols = W if W else [T]
                    plot_label = "CATE vs Confounders (Path 1)" if W else f"CATE vs T={T} (W 없음 — T 자체와의 관계)"
                    fig2, axes = plt.subplots(1, len(plot_cols), figsize=(3.5 * len(plot_cols), 3.5))
                    if len(plot_cols) == 1:
                        axes = [axes]
                    for ax2, w_col in zip(axes, plot_cols):
                        ax2.scatter(df[w_col].values, cate1, alpha=0.4, s=15, color="#1a1a2e")
                        ax2.axhline(0, color="#888", lw=1, ls=":")
                        ax2.set_xlabel(w_col, fontsize=9)
                        ax2.set_ylabel("CATE", fontsize=9)
                        ax2.set_title(w_col + (" [T]" if not W else " [W]"), fontsize=9)
                    plt.suptitle(plot_label, fontsize=10, y=1.01)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close()

                    # --- AI 해석 (단일 / 매개) ---
                    _t_kr = explain.vlabel(T)
                    _y_kr = explain.vlabel(Y)
                    _conf = "확실하다" if ate1_lb * ate1_ub > 0 else "아직 확실하지 않다"
                    _dir = "올라가는" if ate1 >= 0 else "내려가는"
                    if y_mode == "mediation" and ate2 is not None:
                        _d2 = "올라가는" if ate2 >= 0 else "내려가는"
                        _p = (f"다음 사실을 방향·숫자 그대로 일반인에게 2~3문장으로 풀어 설명하라. "
                              f"1단계: '{_t_kr}'이(가) 높아지면 '사용의도'가 {_dir} 방향으로 평균 {abs(ate1):.2f}만큼 움직인다. "
                              f"2단계: '사용의도'가 높아지면 '지속사용의도'가 {_d2} 방향으로 평균 {abs(ate2):.2f}만큼 움직인다. "
                              f"즉 '{_t_kr}'이(가) '사용의도'를 거쳐 '지속사용의도'까지 연쇄적으로 영향을 준다는 점을 설명하라.")
                    else:
                        _sig = ate1_lb * ate1_ub > 0
                        if _sig:
                            _move = "늘어난다" if ate1 >= 0 else "줄어든다"
                            _core = (f"원인은 '{_t_kr}'이고 결과는 '{_y_kr}'이다. "
                                     f"'{_t_kr}'이(가) 높아지면 '{_y_kr}'이(가) {_move}. "
                                     f"평균적으로 '{_t_kr}'이 1점 오를 때 '{_y_kr}'은 약 {abs(ate1):.2f}점 {_move}며, 이 효과는 통계적으로 확실하다.")
                        else:
                            _core = (f"원인은 '{_t_kr}'이고 결과는 '{_y_kr}'이다. "
                                     f"이 데이터에서는 '{_t_kr}'이(가) '{_y_kr}'에 주는 뚜렷한 인과효과가 확인되지 않았다. "
                                     f"(추정치는 약 {ate1:+.2f}점이지만 통계적으로 불확실해 방향을 단정하기 어렵다.)")
                        _p = (f"다음 분석 결론을 일반인에게 2~3문장으로 쉽게 풀어 써라. "
                              f"원인과 결과를 절대 서로 바꾸지 말고, 방향도 그대로 두며, 효과가 생긴 '이유'를 지어내지 마라. "
                              f"사람마다 효과가 다를 수 있다는 점을 한 번 덧붙여라. 결론: {_core}")
                    explain.show(f"step4_single_{T}_{Y}_{y_mode}_{round(ate1,4)}", _p)

                except ImportError:
                    st.error("EconML not installed. Run: pip install econml")
                except Exception as e:
                    st.error(f"Estimation error: {e}")
        else:
            if "ate" in st.session_state:
                ate = st.session_state["ate"]
                ate_lb, ate_ub = st.session_state["ate_ci"]
                st.markdown('<span class="status-ok">추정 결과 있음 / Estimation loaded</span>', unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.markdown(f'<div class="metric-card"><div class="metric-label">ATE</div><div class="metric-value">{ate:.4f}</div></div>', unsafe_allow_html=True)
                m2.markdown(f'<div class="metric-card"><div class="metric-label">95% CI Lower</div><div class="metric-value">{ate_lb:.4f}</div></div>', unsafe_allow_html=True)
                m3.markdown(f'<div class="metric-card"><div class="metric-label">95% CI Upper</div><div class="metric-value">{ate_ub:.4f}</div></div>', unsafe_allow_html=True)
            else:
                st.info("버튼을 눌러 추정을 실행하세요.")


# ===========================================================================
# STEP 5 - Refutation
# ===========================================================================
elif step.startswith("Step 5"):
    st.markdown("## Step 5 - Refutation  /  반증 검증")
    st.markdown(
        '<div class="info-box">'
        '추정 결과의 견고성을 3가지 방법으로 검증한다. 모두 통과해야 인과 주장의 신뢰도가 높아진다.<br>'
        'Test robustness of the causal estimate. All passing increases confidence.'
        '</div>',
        unsafe_allow_html=True,
    )

    T = st.session_state.get("T", "PR_2.6PR")
    T_list = st.session_state.get("T_list", [T])
    Y = st.session_state.get("Y", "USEavg")
    W = st.session_state.get("W", [])
    dag_str = st.session_state.get("dag_str", "")
    is_multi_t_r = len(T_list) > 1
    if is_multi_t_r:
        st.markdown(
            '<div class="info-box">' +
            f'<b>다중 T Refutation ({len(T_list)}개):</b> {", ".join(T_list)}<br>' +
            '각 T에 대해 선택된 검증 방법을 순차 실행한다. 소요 시간이 길 수 있다.' +
            '</div>', unsafe_allow_html=True
        )

    if not dag_str:
        st.warning("Step 2 DAG가 필요합니다.")
        st.stop()

    methods = {
        "Placebo Treatment  /  가짜 처치 주입": "placebo_treatment_refuter",
        "Random Common Cause  /  가짜 교란변수 추가": "random_common_cause",
        "Data Subset  /  데이터 서브셋 검증": "data_subset_refuter",
    }

    # 판정 기준 문서화
    st.markdown("### 판정 기준  /  PASS/FAIL Criteria")
    criteria_df = pd.DataFrame({
        "검증 방법": ["Placebo Treatment", "Random Common Cause", "Data Subset"],
        "PASS 조건": [
            "p_value < 0.05  (원래 효과가 랜덤처치보다 유의하게 큼)",
            "p_value > 0.05  (무작위 교란 추가 후 추정치 변화 유의하지 않음)",
            "p_value > 0.05  (서브셋 추정치 변화 유의하지 않음)",
        ],
        "FAIL 의미": [
            "처치 효과가 인과적이지 않을 수 있음 (허위 상관 의심)",
            "추정치가 교란변수에 민감 — 미관측 교란 위험",
            "소표본 불안정성 존재",
        ],
        "p_value 미지원 시 대체 기준": [
            "|new_effect| < 5% × |orig| + 0.02",
            "|변화율| < 10%",
            "|변화율| < 10%",
        ],
    })
    st.dataframe(criteria_df, use_container_width=True, hide_index=True)

    selected_methods = []
    st.markdown("### 검증 방법 선택  /  Select Methods")
    for label in methods:
        if st.checkbox(label, value=True, key=f"ref_{label}"):
            selected_methods.append(label)

    n_sims = st.slider(
        "Simulations per method  /  시뮬레이션 횟수",
        min_value=50, max_value=1000, value=200, step=50,
        help="퍼뮤테이션 OLS 방식 — sim 200회도 0.1초 이내"
    )

    # 소요 시간 예상치 표시 (퍼뮤테이션 OLS: sim 1회 ≈ 0.3ms)
    if selected_methods:
        est_ms = len(T_list) * len(selected_methods) * n_sims * 0.3
        time_str = f"{est_ms/1000:.1f}초" if est_ms >= 1000 else f"{int(est_ms)}ms"
        st.markdown(
            f'<div class="info-box">' +
            f'예상 소요 시간: T {len(T_list)}개 × 방법 {len(selected_methods)}개 × sim {n_sims}회' +
            f' = 약 <b>{time_str}</b>.' +
            '</div>', unsafe_allow_html=True
        )

    if st.button("Run Refutation  /  반증 검증 실행"):
        if not selected_methods:
            st.warning("검증 방법을 1개 이상 선택해주세요.")
            st.stop()

        prog_r = st.progress(0, text="반증 검증 시작...")
        with st.spinner("Refutation running... (수초 내 완료)"):
            try:
                from sklearn.linear_model import LinearRegression

                # ── 빠른 퍼뮤테이션 기반 반증 (DoWhy overhead 제거) ──────────
                def _ols_coef(t_arr, y_arr, w_arr):
                    """W 통제 후 T의 OLS 계수 반환"""
                    if w_arr.shape[1] > 0:
                        Xf = np.column_stack([t_arr, w_arr])
                    else:
                        Xf = t_arr.reshape(-1, 1)
                    return LinearRegression().fit(Xf, y_arr).coef_[0]

                def _refute_fast(t_arr, y_arr, w_arr, method, n_sims, orig_coef):
                    rng = np.random.default_rng(42)
                    sim_coefs = []
                    for _ in range(n_sims):
                        if method == "placebo_treatment_refuter":
                            t_sim = rng.permutation(t_arr)
                        elif method == "random_common_cause":
                            noise = rng.normal(size=len(t_arr))
                            if w_arr.shape[1] > 0:
                                w_sim = np.column_stack([w_arr, noise])
                            else:
                                w_sim = noise.reshape(-1, 1)
                            sim_coefs.append(_ols_coef(t_arr, y_arr, w_sim))
                            continue
                        else:  # data_subset_refuter
                            idx = rng.choice(len(t_arr), size=int(len(t_arr)*0.8), replace=False)
                            t_sim = t_arr[idx]
                            y_sim = y_arr[idx]
                            w_sim = w_arr[idx] if w_arr.shape[1] > 0 else w_arr[idx]
                            sim_coefs.append(_ols_coef(t_sim, y_sim, w_sim))
                            continue
                        sim_coefs.append(_ols_coef(t_sim, y_arr, w_arr))
                    sim_coefs = np.array(sim_coefs)
                    new_eff = float(sim_coefs.mean())
                    # p-value: placebo는 시뮬 분포에서 orig만큼 극단적일 확률
                    if method == "placebo_treatment_refuter":
                        pval = float(np.mean(np.abs(sim_coefs) >= abs(orig_coef)))
                    else:
                        pval = float(np.mean(np.abs(sim_coefs - orig_coef) >= abs(new_eff - orig_coef)))
                    return new_eff, pval

                all_t_results = {}
                total_steps = len(T_list) * len(selected_methods)
                step_n = 0

                for t_i in T_list:
                    t_arr = df[t_i].values.astype(float)
                    y_arr = df[Y].values.astype(float)
                    w_arr = df[W].values if W else np.empty((len(df), 0))
                    orig_coef = _ols_coef(t_arr, y_arr, w_arr)
                    results_t = []
                    for label in selected_methods:
                        method_name = methods[label]
                        step_n += 1
                        prog_r.progress(step_n / total_steps,
                                        text=f"T={t_i} / {label.split('  /  ')[1]} ({step_n}/{total_steps})")
                        try:
                            new_eff, pval = _refute_fast(t_arr, y_arr, w_arr, method_name, n_sims, orig_coef)
                            change_pct = abs(new_eff - orig_coef) / (abs(orig_coef) + 1e-9) * 100
                            if method_name == "placebo_treatment_refuter":
                                # placebo: p = P(랜덤처치 효과 >= 원래효과)
                                # p < 0.05 → 원래 효과가 유의하게 큼 → 인과 효과 실재 → PASS
                                passed = pval < 0.05
                                criteria_used = f"p={pval:.4f} < 0.05 (원래 효과가 랜덤보다 유의하게 큼)"
                            else:
                                # random_common_cause / data_subset:
                                # p > 0.05 → 변화가 우연 수준 → 추정치 안정 → PASS
                                passed = pval > 0.05
                                criteria_used = f"p={pval:.4f} > 0.05 (추정치 변화가 유의하지 않음)"
                            results_t.append({
                                "T": t_i,
                                "방법": label.split("  /  ")[1],
                                "원래 효과": round(orig_coef, 4),
                                "새 효과": round(new_eff, 4),
                                "변화율": f"{change_pct:.1f}%",
                                "p_value": f"{pval:.4f}",
                                "판정 기준": criteria_used,
                                "결과": "PASS" if passed else "FAIL",
                            })
                        except Exception as e_ref:
                            results_t.append({
                                "T": t_i, "방법": label.split("  /  ")[1],
                                "원래 효과": "—", "새 효과": "—", "변화율": "—",
                                "p_value": "—", "판정 기준": "—",
                                "결과": f"ERROR: {str(e_ref)[:40]}",
                            })
                    all_t_results[t_i] = results_t
                prog_r.empty()

                st.success("반증 검증 완료 / Refutation complete")

                # 결과 표시: T별 탭
                if len(T_list) > 1:
                    tabs_r = st.tabs([f"T={t}" for t in T_list])
                    for ti_idx, t_i in enumerate(T_list):
                        with tabs_r[ti_idx]:
                            st.markdown(f"### T={t_i} 검증 요약")
                            df_rt = pd.DataFrame(all_t_results[t_i])
                            def _hl(row):
                                c = "background-color: #d1efe0" if row.get("결과") == "PASS" else "background-color: #fae3d0"
                                return [c]*len(row)
                            try:
                                st.dataframe(df_rt.style.apply(_hl, axis=1), use_container_width=True, hide_index=True)
                            except Exception:
                                st.dataframe(df_rt, use_container_width=True, hide_index=True)
                    # 전체 통합 요약
                    all_rows = [r for rows in all_t_results.values() for r in rows]
                    st.markdown("### 전체 통합 요약  /  All-T Refutation Summary")
                    df_all_r = pd.DataFrame(all_rows)
                    try:
                        st.dataframe(df_all_r.style.apply(_hl, axis=1), use_container_width=True, hide_index=True)
                    except Exception:
                        st.dataframe(df_all_r, use_container_width=True, hide_index=True)
                else:
                    rows_single = all_t_results[T_list[0]]
                    st.markdown("### 검증 요약  /  Refutation Summary")
                    df_rs = pd.DataFrame(rows_single)
                    def _hl_s(row):
                        c = "background-color: #d1efe0" if row.get("결과") == "PASS" else "background-color: #fae3d0"
                        return [c]*len(row)
                    try:
                        st.dataframe(df_rs.style.apply(_hl_s, axis=1), use_container_width=True, hide_index=True)
                    except Exception:
                        st.dataframe(df_rs, use_container_width=True, hide_index=True)

                st.session_state["refutation_done"] = True

                # --- AI 해석 ---
                _all = [r for rows in all_t_results.values() for r in rows]
                _npass = sum(1 for r in _all if r["결과"] == "PASS")
                _allpass = (_npass == len(_all))
                _t_kr = explain.vlabels(T_list)
                _y_kr = explain.vlabel(Y)
                explain.show(
                    f"step5_{_npass}_{len(_all)}_{T_list}_{Y}",
                    f"앞에서 계산한 '[{_t_kr}]가 {_y_kr}에 주는 효과'가 우연이나 착시가 아닌지 검사하는 '반증' 테스트를 "
                    f"총 {len(_all)}번 해서 {_npass}번 통과(PASS)했다. ({'전부 통과' if _allpass else '일부 미통과'}) "
                    f"핵심만 2~3문장으로: 통과는 '결과를 믿어도 된다', 미통과는 '주의가 필요하다'는 뜻이며, "
                    f"{'대부분 통과했으니 앞의 분석 결과를 신뢰할 수 있다' if _allpass else '미통과가 있어 해석에 주의가 필요하다'}고 알려라. "
                    f"여기서는 '결과가 견고한지'만 본다 — 효과의 구체적 방향이나 크기는 언급하지 말고, 위에 없는 새 변수나 소재를 절대 지어내지 마라. 짧게 써라.",
                )

            except Exception as e:
                st.error(f"Refutation error: {e}")
            finally:
                try:
                    prog_r.empty()
                except Exception:
                    pass
    else:
        if st.session_state.get("refutation_done"):
            st.markdown('<span class="status-ok">검증 완료 / Refutation done</span>', unsafe_allow_html=True)
        else:
            st.info("버튼을 눌러 검증을 실행하세요.")


# ===========================================================================
# STEP 6 - Counterfactual
# ===========================================================================
elif step.startswith("Step 6"):
    st.markdown("## Step 6 - Counterfactual  /  반사실 추론")
    T_step6 = st.session_state.get("T", "T")
    T_list_6 = st.session_state.get("T_list", [T_step6])
    is_multi_t_6 = len(T_list_6) > 1
    if is_multi_t_6:
        st.markdown(
            '<div class="info-box">' +
            f'<b>다중 T 반사실 모드 ({len(T_list_6)}개):</b> {", ".join(T_list_6)}<br>' +
            '각 T별 탭에서 시나리오를 설정하고 실행한다. estimator는 Step 4에서 저장된 T[0] 기준.' +
            '</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="info-box">' +
            f'"만약 T={T_step6}이 달랐다면?" 시나리오를 개인 수준에서 시뮬레이션한다.<br>' +
            f'Simulate "What if T={T_step6} had been different?" at the individual level.' +
            f'</div>',
            unsafe_allow_html=True,
        )

    T = st.session_state.get("T", "PR_2.6PR")
    Y = st.session_state.get("Y", "USEavg")
    Y2 = st.session_state.get("Y2", None)
    y_mode = st.session_state.get("y_mode", "single")
    W = st.session_state.get("W", ["Uavg", "SQavg", "IFavg", "SIavg"])

    estimator = st.session_state.get("estimator", None)
    estimator_map = st.session_state.get("estimator_map", {})
    multi_cate_map = st.session_state.get("multi_cate", {})  # {t: (cate, ate, lb, ub)}

    if estimator is None:
        st.warning("Step 4 추정을 먼저 실행해주세요.")
        st.stop()

    # 다중 T: selectbox로 분석할 T 선택
    if is_multi_t_6:
        cf_T = st.selectbox(
            "분석할 T 선택  /  Select T for counterfactual",
            T_list_6, key="cf_T_select"
        )
    else:
        cf_T = T

    # cf_T에 맞는 estimator 선택 (estimator_map 우선, 없으면 T[0] estimator 사용)
    active_estimator = estimator_map.get(cf_T, estimator)

    st.markdown(f"### 시나리오 설정  /  Counterfactual Scenario  (T={cf_T})")
    t_vals_all = df[cf_T].values.astype(float)
    t_is_binary = set(np.unique(t_vals_all)).issubset({0.0, 1.0})

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if t_is_binary:
            t0_val = st.selectbox(
                f"Factual T  /  실제 {cf_T} 값",
                [0, 1],
                format_func=lambda x: f"{x}  ({'1 (높음)' if x==1 else '0 (낮음)'})",
                index=1, key=f"cf_t0_{cf_T}"
            )
        else:
            t0_val = st.number_input(
                f"Factual T  /  실제 {cf_T} 값",
                value=float(t_vals_all.mean()), step=0.1,
                key=f"cf_t0_{cf_T}"
            )
    with col_s2:
        if t_is_binary:
            t1_val = st.selectbox(
                f"Counterfactual T  /  가정 {cf_T} 값",
                [0, 1],
                format_func=lambda x: f"{x}  ({'1 (높음)' if x==1 else '0 (낮음)'})",
                index=0, key=f"cf_t1_{cf_T}"
            )
        else:
            t1_val = st.number_input(
                f"Counterfactual T  /  가정 {cf_T} 값",
                value=float(t_vals_all.mean()) - 1.0, step=0.1,
                key=f"cf_t1_{cf_T}"
            )

    st.markdown("### 서브그룹 필터  /  Subgroup Filter (선택)")
    filter_col = st.selectbox("Filter by", ["None"] + W)
    df_filtered = df.copy()
    if filter_col != "None":
        min_v, max_v = float(df[filter_col].min()), float(df[filter_col].max())
        r = st.slider(f"{filter_col} range", min_v, max_v, (min_v, max_v), 0.1)
        df_filtered = df_filtered[(df_filtered[filter_col] >= r[0]) & (df_filtered[filter_col] <= r[1])]
        st.markdown(f"Filtered: {len(df_filtered)} / {len(df)} 명")

    if st.button("Run Counterfactual  /  반사실 추론 실행"):
        with st.spinner("Computing counterfactuals..."):
            try:
                X_sub = df_filtered[W].values if W else np.ones((len(df_filtered), 1))
                y_obs = df_filtered[Y].values.astype(float)
                effect = active_estimator.effect(X_sub, T0=t0_val, T1=t1_val)
                y_cf = y_obs + effect
                delta = y_cf.mean() - y_obs.mean()
                sign = "+" if delta >= 0 else ""

                # 매개 분석 Y2 반사실
                y2_obs = y2_cf = delta2 = sign2 = None
                if y_mode == "mediation" and Y2 and "estimator2" in st.session_state:
                    est2 = st.session_state["estimator2"]
                    y2_obs = df_filtered[Y2].values.astype(float)
                    use_obs = df_filtered["USEavg"].values.astype(float)
                    use_cf = use_obs + effect
                    effect2 = est2.effect(X_sub, T0=use_obs, T1=use_cf)
                    y2_cf = y2_obs + effect2
                    delta2 = y2_cf.mean() - y2_obs.mean()
                    sign2 = "+" if delta2 >= 0 else ""

                # 결과 지표
                st.markdown("### 결과  /  Results")
                if y_mode == "mediation" and y2_obs is not None:
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.markdown(f'<div class="metric-card"><div class="metric-label">사용의도 실제</div><div class="metric-value">{y_obs.mean():.3f}</div></div>', unsafe_allow_html=True)
                    mc2.markdown(f'<div class="metric-card"><div class="metric-label">사용의도 반사실</div><div class="metric-value">{y_cf.mean():.3f}</div></div>', unsafe_allow_html=True)
                    mc3.markdown(f'<div class="metric-card"><div class="metric-label">지속사용의도 실제</div><div class="metric-value">{y2_obs.mean():.3f}</div></div>', unsafe_allow_html=True)
                    mc4.markdown(f'<div class="metric-card"><div class="metric-label">지속사용의도 반사실</div><div class="metric-value">{y2_cf.mean():.3f}</div></div>', unsafe_allow_html=True)
                    if t_is_binary:
                        label_t0 = f"{cf_T}={t0_val} ({'높음' if t0_val == 1 else '낮음'})"
                        label_t1 = f"{cf_T}={t1_val} ({'높음' if t1_val == 1 else '낮음'})"
                    else:
                        label_t0 = f"{cf_T}={t0_val:.3f}"
                        label_t1 = f"{cf_T}={t1_val:.3f}"
                    st.markdown(
                        f'<div class="info-box">'
                        f'{cf_T}를 {label_t0}에서 {label_t1}으로 바꿨을 때:<br>'
                        f'사용의도(Y1): <b>{y_obs.mean():.3f} → {y_cf.mean():.3f}</b>  ({sign}{delta:.3f})<br>'
                        f'지속사용의도(Y2, 매개): <b>{y2_obs.mean():.3f} → {y2_cf.mean():.3f}</b>  ({sign2}{delta2:.3f})'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    m1, m2, m3, m4 = st.columns(4)
                    if t_is_binary:
                        label_t0 = f"T={t0_val} ({'높음' if t0_val == 1 else '낮음'})"
                        label_t1 = f"T={t1_val} ({'높음' if t1_val == 1 else '낮음'})"
                    else:
                        label_t0 = f"{cf_T}={t0_val:.3f}"
                        label_t1 = f"{cf_T}={t1_val:.3f}"
                    m1.markdown(f'<div class="metric-card"><div class="metric-label">실제 {Y} 평균</div><div class="metric-value">{y_obs.mean():.3f}</div></div>', unsafe_allow_html=True)
                    m2.markdown(f'<div class="metric-card"><div class="metric-label">반사실 {Y} 평균</div><div class="metric-value">{y_cf.mean():.3f}</div></div>', unsafe_allow_html=True)
                    m3.markdown(f'<div class="metric-card"><div class="metric-label">변화량 (Delta)</div><div class="metric-value">{sign}{delta:.3f}</div></div>', unsafe_allow_html=True)
                    m4.markdown(f'<div class="metric-card"><div class="metric-label">대상 인원</div><div class="metric-value">{len(df_filtered)}</div></div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="info-box">'
                        f'{cf_T}를 {label_t0}에서 {label_t1}으로 바꿨을 때, '
                        f'{Y} 평균: <b>{y_obs.mean():.3f} → {y_cf.mean():.3f}</b>  ({sign}{delta:.3f})'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # 분포 비교
                st.markdown("### 분포 비교  /  Distribution Comparison")
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                axes[0].hist(y_obs, bins=25, alpha=0.7, color="#6b6b8a", edgecolor="white", label=f"실제 (T={t0_val})")
                axes[0].hist(y_cf, bins=25, alpha=0.7, color="#2d6a4f", edgecolor="white", label=f"반사실 (T={t1_val})")
                axes[0].axvline(y_obs.mean(), color="#6b6b8a", ls="--", lw=1.5)
                axes[0].axvline(y_cf.mean(), color="#2d6a4f", ls="--", lw=1.5)
                axes[0].set_xlabel(Y)
                axes[0].set_ylabel("응답자 수")
                axes[0].set_title(f"실제 vs 반사실  ({Y})")
                axes[0].legend(fontsize=8)

                axes[1].scatter(y_obs, y_cf, alpha=0.4, s=18, color="#1a1a2e")
                lim = [min(y_obs.min(), y_cf.min()) - 0.2, max(y_obs.max(), y_cf.max()) + 0.2]
                axes[1].plot(lim, lim, "r--", lw=1, label="변화 없음")
                axes[1].set_xlabel(f"실제 {Y}")
                axes[1].set_ylabel(f"반사실 {Y}")
                axes[1].set_title("개별 반사실 산점도")
                axes[1].legend(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # 샘플 테이블
                st.markdown("### 샘플 결과  /  Sample Table (상위 20명)")
                out_df = df_filtered[W + [T, Y]].copy().head(20).reset_index(drop=True)
                out_df["실제_Y"] = y_obs[:20].round(3)
                out_df["반사실_Y"] = y_cf[:20].round(3)
                out_df["효과(Effect)"] = effect[:20].round(3)
                st.dataframe(out_df, use_container_width=True, hide_index=True)
                st.success("반사실 추론 완료 / Counterfactual complete")

                # --- AI 해석 (방향·대상집단을 Python에서 확정해 전달) ---
                _t_kr = explain.vlabel(cf_T)
                _y_kr = explain.vlabel(Y)
                if filter_col != "None":
                    _med = float(df[filter_col].median())
                    if r[1] <= _med + 0.01:
                        _lvl = "낮은 편"
                    elif r[0] >= _med - 0.01:
                        _lvl = "높은 편"
                    else:
                        _lvl = "중간 범위"
                    _sub = f"'{explain.vlabel(filter_col)}'이(가) {_lvl}({r[0]:.1f}~{r[1]:.1f}점)인 응답자 {len(df_filtered)}명"
                else:
                    _sub = f"전체 응답자 {len(df_filtered)}명"
                if t_is_binary:
                    _from = "높음" if float(t0_val) == 1 else "낮음"
                    _to = "높음" if float(t1_val) == 1 else "낮음"
                    _change = f"'{_t_kr}'을(를) {_from}에서 {_to}(으)로 바꿨다면"
                else:
                    _verb = "높였다면" if float(t1_val) > float(t0_val) else "낮췄다면"
                    _change = f"'{_t_kr}'을(를) 약 {abs(float(t1_val)-float(t0_val)):.1f}점 {_verb}"
                _ychg = "늘어났을" if float(delta) >= 0 else "줄어들었을"
                _gap = abs(float(delta))
                if _gap < 0.15:
                    _fact = (f"{_sub}을(를) 대상으로 {_change}, 이들의 '{_y_kr}'에는 사실상 거의 변화가 없을 것으로 추정됐다 "
                             f"(평균 {y_obs.mean():.2f}점 → {y_cf.mean():.2f}점, 약 {_gap:.2f}점 차이로 무시할 만한 수준)")
                    _extra = "효과가 거의 없으니 '거의 차이가 없다'는 점을 분명히 하고, 억지로 의미를 부여하지 마라."
                else:
                    _fact = (f"{_sub}을(를) 대상으로, {_change} 이들의 '{_y_kr}' 평균이 "
                             f"{y_obs.mean():.2f}점에서 {y_cf.mean():.2f}점으로 약 {_gap:.2f}점 {_ychg} 것으로 추정됐다")
                    _extra = ""
                explain.show(
                    f"step6_{cf_T}_{t0_val}_{t1_val}_{filter_col}_{round(float(delta),4)}_{len(df_filtered)}",
                    f"다음은 '만약 그때 다르게 했다면?'을 추정한 반사실 시뮬레이션 결과다. 이 사실을 방향·숫자 그대로(절대 반대로 바꾸지 말 것) "
                    f"일반인에게 2~3문장으로 풀어 설명하라. 어떤 집단을 대상으로 한 결과인지 반드시 밝히고, 변화가 일어난 '이유'는 추측하거나 지어내지 마라. {_extra} 결과: {_fact}. "
                    f"마지막에 이런 가정 시뮬레이션이 실제 의사결정에 어떻게 도움이 되는지 한 문장 덧붙여라.",
                )

            except Exception as e:
                st.error(f"Counterfactual error: {e}")
    else:
        st.info("시나리오 설정 후 버튼을 누르세요.")


# ===========================================================================
# STEP 7 - H1-H5 인과 분석
# Paper-Aligned Analysis: T=W vars, W=PR, Y=USE/ICU
# ===========================================================================
elif step.startswith("Step 7"):
    st.markdown("## Step 7 - H1-H5 인과 분석  /  Causal Analysis (H1-H5)")
    st.markdown(
        '<div class="info-box">'
        '<b>H1~H5 인과 구조를 CausalForestDML로 추정한다.</b><br>'
        'H1~H4: 유용성/품질/사회적영향 → 사용의도 (PRavg를 교란변수로 통제)<br>'
        'H5: 사용의도 → 지속사용의도 (PRavg 교란변수 통제)<br>'
        '<br>H1-H4: Independent vars → Use Intention (PRavg controlled as confounder)'
        '<br>H5: Use Intention → Continuance Intention (PRavg controlled)'
        '</div>',
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # 이론적 구조 테이블
    # ------------------------------------------------------------------
    st.markdown("### 분석 설계  /  Analysis Design")
    design_df = pd.DataFrame({
        "가설": ["H1", "H2", "H3", "H4", "H5"],
        "T (처치/원인)": ["Uavg (유용성)", "SQavg (시스템품질)", "IFavg (인터페이스품질)", "SIavg (사회적영향)", "USEavg (사용의도)"],
        "W (교란변수)": ["PRavg + SQavg,IFavg,SIavg", "PRavg + Uavg,IFavg,SIavg", "PRavg + Uavg,SQavg,SIavg", "PRavg + Uavg,SQavg,IFavg", "PRavg + Uavg,SQavg,IFavg,SIavg"],
        "Y (결과)": ["USEavg", "USEavg", "USEavg", "USEavg", "ICUavg"],
        "T 유형": ["연속형", "연속형", "연속형", "연속형", "연속형"],
    })
    st.dataframe(design_df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # 통합 DAG (전체 인과 구조)
    # ------------------------------------------------------------------
    st.markdown("### 전체 인과 구조 DAG  /  Unified Causal Structure")
    st.markdown(
        '<div class="info-box" style="font-size:0.82rem">'
        '전체 변수와 인과 경로를 단일 그래프로 표시.<br>'
        '왼쪽: 독립변수(T 후보) — 중앙: PRavg(교란변수, 빨강) — 오른쪽: 결과변수(Y)'
        '</div>', unsafe_allow_html=True,
    )

    def _draw_unified_dag(ax):
        G_u = nx.DiGraph()
        indep_vars = ["Uavg", "SQavg", "IFavg", "SIavg"]
        for v in indep_vars:
            G_u.add_edge(v, "USEavg")
            G_u.add_edge("PRavg", v)
        G_u.add_edge("PRavg", "USEavg")
        G_u.add_edge("PRavg", "ICUavg")
        G_u.add_edge("USEavg", "ICUavg")

        pos_u = {
            "Uavg":   (-2.2,  1.8),
            "SQavg":  (-2.2,  0.6),
            "IFavg":  (-2.2, -0.6),
            "SIavg":  (-2.2, -1.8),
            "PRavg":  ( 0.0,  3.0),
            "USEavg": ( 0.0,  0.0),
            "ICUavg": ( 2.5,  0.0),
        }
        node_labels_u = {
            "Uavg":   "Uavg\n[Usefulness]",
            "SQavg":  "SQavg\n[Sys.Quality]",
            "IFavg":  "IFavg\n[Interface]",
            "SIavg":  "SIavg\n[Soc.Influence]",
            "PRavg":  "PRavg\n[Confounder]",
            "USEavg": "USEavg\n[Use Intention]",
            "ICUavg": "ICUavg\n[Continuance]",
        }
        node_colors_u = []
        for n in G_u.nodes():
            if n in indep_vars:      node_colors_u.append("#1a1a2e")
            elif n == "PRavg":       node_colors_u.append("#c0392b")
            elif n in ["USEavg", "ICUavg"]: node_colors_u.append("#2d6a4f")
            else:                    node_colors_u.append("#6b6b8a")

        edge_colors_u = []
        edge_widths_u = []
        for u, v in G_u.edges():
            if u in indep_vars and v == "USEavg":
                edge_colors_u.append("#1a1a2e"); edge_widths_u.append(2.2)
            elif u == "USEavg" and v == "ICUavg":
                edge_colors_u.append("#2d6a4f"); edge_widths_u.append(2.5)
            elif u == "PRavg":
                edge_colors_u.append("#c0392b"); edge_widths_u.append(1.5)
            else:
                edge_colors_u.append("#8a8a8a"); edge_widths_u.append(1.0)

        nx.draw_networkx(
            G_u, pos=pos_u, ax=ax,
            labels=node_labels_u,
            node_color=node_colors_u,
            node_size=2400,
            font_color="white",
            font_size=7,
            font_weight="bold",
            arrows=True,
            arrowsize=18,
            edge_color=edge_colors_u,
            width=edge_widths_u,
        )
        legend_u = [
            plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#1a1a2e", markersize=9, label="T (Treatment candidates)"),
            plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#c0392b", markersize=9, label="W (Confounder: PRavg)"),
            plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#2d6a4f", markersize=9, label="Y (Outcome)"),
            plt.Line2D([0],[0], color="#1a1a2e", lw=2.2, label="H1-H4: Indep.Var → USEavg"),
            plt.Line2D([0],[0], color="#2d6a4f", lw=2.5, label="H5: USEavg → ICUavg"),
            plt.Line2D([0],[0], color="#c0392b", lw=1.5, label="Confounding path (PRavg)"),
        ]
        ax.legend(handles=legend_u, loc="lower left", fontsize=7.5, framealpha=0.9)
        ax.set_title("Unified Causal DAG  /  전체 인과 구조", fontsize=11, pad=10)
        ax.axis("off")

    fig_unified, ax_unified = plt.subplots(figsize=(12, 6))
    _draw_unified_dag(ax_unified)
    plt.tight_layout()
    st.pyplot(fig_unified)
    plt.close()

    st.markdown(
        '<div class="info-box" style="font-size:0.8rem">'
        '<b>변수 설명:</b> '
        'Uavg=유용성 &nbsp;|&nbsp; SQavg=시스템품질 &nbsp;|&nbsp; IFavg=인터페이스품질 &nbsp;|&nbsp; '
        'SIavg=사회적영향 &nbsp;|&nbsp; PRavg=인지된위험(교란변수) &nbsp;|&nbsp; '
        'USEavg=사용의도 &nbsp;|&nbsp; ICUavg=지속사용의도'
        '</div>', unsafe_allow_html=True,
    )
    st.divider()

    # ------------------------------------------------------------------
    # DAG 시각화 (H1-H5 개별)
    # ------------------------------------------------------------------
    st.markdown("### 가설별 DAG  /  Per-Hypothesis DAG  (H1-H5 개별)")
    st.markdown(
        '<div class="info-box" style="font-size:0.82rem">'
        '각 가설별 DAG: PRavg(빨강)는 T와 Y 양쪽에 화살표 — 교란변수 구조.<br>'
        '진한 화살표(굵음): 검증 대상 인과 경로 (T → Y).  회색 노드: 추가 통제변수(W).'
        '</div>', unsafe_allow_html=True,
    )

    # H1~H5 DAG 정의
    DAG_DEFS = [
        dict(hyp="H1", t="Uavg",   y="USEavg", title="H1\nUavg → USEavg",
             others=["SQavg","IFavg","SIavg"], other_y="USEavg"),
        dict(hyp="H2", t="SQavg",  y="USEavg", title="H2\nSQavg → USEavg",
             others=["Uavg","IFavg","SIavg"],  other_y="USEavg"),
        dict(hyp="H3", t="IFavg",  y="USEavg", title="H3\nIFavg → USEavg",
             others=["Uavg","SQavg","SIavg"],  other_y="USEavg"),
        dict(hyp="H4", t="SIavg",  y="USEavg", title="H4\nSIavg → USEavg",
             others=["Uavg","SQavg","IFavg"],  other_y="USEavg"),
        dict(hyp="H5", t="USEavg", y="ICUavg", title="H5\nUSEavg → ICUavg",
             others=["Uavg","SQavg","IFavg","SIavg"], other_y="USEavg"),
    ]

    # 2행: H1-H4 상단, H5 하단 단독
    fig_dag1, axes_dag1 = plt.subplots(1, 4, figsize=(18, 4.2))
    fig_dag2, ax_dag2   = plt.subplots(1, 1, figsize=(5.5, 4.2))

    def _draw_dag(ax, d):
        G = nx.DiGraph()
        t, y = d["t"], d["y"]
        # PR → T (교란: T에 영향)
        G.add_edge("PRavg", t)
        # PR → Y (교란: Y에 직접 영향)
        G.add_edge("PRavg", y)
        # T → Y (핵심 인과 경로)
        G.add_edge(t, y)
        # 나머지 W → Y (추가 통제)
        for o in d["others"]:
            G.add_edge(o, d["other_y"])
        # H5 특수: 나머지 W → USEavg(T)에도 영향
        if d["hyp"] == "H5":
            for o in d["others"]:
                G.add_edge(o, t)

        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G, seed=hash(d["hyp"]) % 100, k=2.2)

        node_colors = []
        for n in G.nodes():
            if n == t:         node_colors.append("#1a1a2e")   # T: 처치 (검정)
            elif n == y and n != t: node_colors.append("#2d6a4f")  # Y: 결과 (녹색)
            elif n == "PRavg": node_colors.append("#c0392b")   # PR: 교란 (빨강)
            else:              node_colors.append("#6b6b8a")   # W: 추가통제 (회색)

        # 엣지 색상: T→Y는 강조, 나머지는 회색
        edge_colors = []
        edge_widths = []
        for u, v in G.edges():
            if u == t and v == y:
                edge_colors.append("#1a1a2e")
                edge_widths.append(3.0)
            elif u == "PRavg":
                edge_colors.append("#c0392b")
                edge_widths.append(1.8)
            else:
                edge_colors.append("#8a8a8a")
                edge_widths.append(1.0)

        nx.draw_networkx(
            G, pos=pos, ax=ax,
            node_color=node_colors,
            node_size=1200,
            font_color="white",
            font_size=7,
            font_weight="bold",
            arrows=True,
            arrowsize=16,
            edge_color=edge_colors,
            width=edge_widths,
        )
        ax.set_title(d["title"], fontsize=9, pad=6)
        ax.axis("off")

    for ax_i, dd in zip(axes_dag1, DAG_DEFS[:4]):
        _draw_dag(ax_i, dd)

    _draw_dag(ax_dag2, DAG_DEFS[4])

    legend_h = [
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#1a1a2e", markersize=9, label="T: 처치변수 (Treatment)"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#2d6a4f", markersize=9, label="Y: 결과변수 (Outcome)"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#c0392b", markersize=9, label="W: PRavg 교란변수 (Confounder)"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#6b6b8a", markersize=9, label="W: 추가 통제변수 (Control)"),
        plt.Line2D([0],[0], color="#1a1a2e", lw=3,   label="T → Y: 검증 대상 인과 경로"),
        plt.Line2D([0],[0], color="#c0392b", lw=1.8, label="PR → T/Y: 교란 경로"),
    ]
    fig_dag1.legend(handles=legend_h, loc="lower center", ncol=3, fontsize=8,
                    bbox_to_anchor=(0.5, -0.10), framealpha=0.9)
    fig_dag1.suptitle("H1~H4: 독립변수 → 사용의도 (PRavg 교란변수 구조)", fontsize=10, y=1.01)
    plt.tight_layout()
    st.pyplot(fig_dag1)
    plt.close()

    fig_dag2.legend(handles=legend_h, loc="lower center", ncol=3, fontsize=8,
                    bbox_to_anchor=(0.5, -0.18), framealpha=0.9)
    fig_dag2.suptitle("H5: 사용의도 → 지속사용의도 (PRavg 교란변수 구조)", fontsize=10, y=1.02)
    plt.tight_layout()
    st.pyplot(fig_dag2)
    plt.close()

    # ------------------------------------------------------------------
    # 파라미터 설정
    # ------------------------------------------------------------------
    st.markdown("### 추정 파라미터  /  Estimation Parameters")
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        n_est7 = st.slider("n_estimators", 50, 500, 200, 50, key="n_est7")
    with pc2:
        min_s7 = st.slider("min_samples_leaf", 5, 50, 10, 5, key="min_s7")
    with pc3:
        cv7 = st.slider("cv (cross-fit)", 2, 5, 3, key="cv7")

    # ------------------------------------------------------------------
    # 5개 분석 일괄 실행
    # ------------------------------------------------------------------
    if st.button("Run All H1-H5 Analyses  /  H1-H5 전체 분석 실행"):
        from econml.dml import CausalForestDML
        from sklearn.ensemble import GradientBoostingRegressor

        ANALYSES = [
            dict(hyp="H1", t="Uavg",   t_label="유용성",          w=["PRavg","SQavg","IFavg","SIavg"], y="USEavg"),
            dict(hyp="H2", t="SQavg",  t_label="시스템품질",       w=["PRavg","Uavg","IFavg","SIavg"],  y="USEavg"),
            dict(hyp="H3", t="IFavg",  t_label="인터페이스품질",   w=["PRavg","Uavg","SQavg","SIavg"],  y="USEavg"),
            dict(hyp="H4", t="SIavg",  t_label="사회적영향",       w=["PRavg","Uavg","SQavg","IFavg"],  y="USEavg"),
            dict(hyp="H5", t="USEavg", t_label="사용의도",          w=["PRavg","Uavg","SQavg","IFavg","SIavg"], y="ICUavg"),
        ]

        results7 = []
        cate_map = {}
        progress = st.progress(0, text="분석 실행 중...")

        for i, an in enumerate(ANALYSES):
            try:
                t_vals7 = df[an["t"]].values.astype(float)
                y_vals7 = df[an["y"]].values.astype(float)
                X7 = df[an["w"]].values

                est7 = CausalForestDML(
                    model_y=GradientBoostingRegressor(n_estimators=100),
                    model_t=GradientBoostingRegressor(n_estimators=100),
                    n_estimators=n_est7,
                    min_samples_leaf=min_s7,
                    cv=cv7,
                    random_state=42,
                    discrete_treatment=False,
                )
                est7.fit(y_vals7, t_vals7, X=X7)
                ate7 = float(est7.ate(X7))
                lb7, ub7 = est7.ate_interval(X7, alpha=0.05)
                cate7 = est7.effect(X7)
                cate_map[an["hyp"]] = (an, cate7, ate7, float(lb7), float(ub7))

                sig = "유의" if lb7 * ub7 > 0 else "비유의"
                direction = "+" if ate7 >= 0 else "-"
                results7.append({
                    "가설": an["hyp"],
                    "T (처치)": f"{an['t']} ({an['t_label']})",
                    "Y (결과)": an["y"],
                    "ATE": round(ate7, 4),
                    "CI 하한": round(float(lb7), 4),
                    "CI 상한": round(float(ub7), 4),
                    "방향": direction,
                    "유의성 (95%)": sig,
                    "CATE 평균": round(float(cate7.mean()), 4),
                    "CATE SD": round(float(cate7.std()), 4),
                })
            except Exception as e:
                results7.append({
                    "가설": an["hyp"],
                    "T (처치)": an["t"],
                    "Y (결과)": an["y"],
                    "ATE": "ERROR",
                    "CI 하한": str(e)[:30],
                    "CI 상한": "", "방향": "", "유의성 (95%)": "", "CATE 평균": "", "CATE SD": "",
                })
            progress.progress((i + 1) / len(ANALYSES), text=f"{an['hyp']} 완료 ({i+1}/5)")

        st.session_state["paper_results7"] = results7
        st.session_state["paper_cate7"] = cate_map
        progress.empty()
        st.markdown('<span class="status-ok">전체 분석 완료 / All analyses complete</span>', unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 결과 출력
    # ------------------------------------------------------------------
    if "paper_results7" in st.session_state:
        results7 = st.session_state["paper_results7"]
        cate_map = st.session_state.get("paper_cate7", {})

        st.markdown("### 가설별 ATE 비교  /  ATE Comparison Table")
        res_df = pd.DataFrame(results7)

        def highlight_sig(row):
            color = "background-color: #d1efe0" if row["유의성 (95%)"] == "유의" else "background-color: #fae3d0"
            return [color] * len(row)

        try:
            st.dataframe(res_df.style.apply(highlight_sig, axis=1), use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(res_df, use_container_width=True, hide_index=True)

        st.markdown(
            '<div class="info-box">'
            '<b>ATE 해석 기준 (연속형 T)</b><br>'
            'ATE = T 변수 1단위 증가 시 Y의 평균 변화량<br>'
            '예) H1 ATE=0.45: 유용성(Uavg)이 1점 높아질 때 사용의도(USEavg)가 평균 0.45점 증가<br>'
            '95% CI가 0을 포함하지 않으면 통계적으로 유의 (녹색 행)<br>'
            'PRavg를 교란변수로 통제한 후의 순수 인과 효과 추정치'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### ATE 비교 그래프  /  ATE Bar Chart")
        valid = [r for r in results7 if isinstance(r["ATE"], float)]
        if valid:
            fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
            labels = [f"{r['가설']}\n({r['T (처치)'].split('(')[0].strip()})" for r in valid]
            ates = [r["ATE"] for r in valid]
            lbs  = [r["CI 하한"] for r in valid]
            ubs  = [r["CI 상한"] for r in valid]
            bar_colors = ["#2d6a4f" if lb * ub > 0 else "#b07040" for lb, ub in zip(lbs, ubs)]
            bars = ax_bar.bar(labels, ates, color=bar_colors, edgecolor="white", width=0.55)
            for bar_r, lb, ub in zip(bars, lbs, ubs):
                x = bar_r.get_x() + bar_r.get_width() / 2
                ax_bar.plot([x, x], [lb, ub], color="#444", lw=2)
            ax_bar.axhline(0, color="#888", lw=1, ls="--")
            ax_bar.set_ylabel("ATE (T 1단위 → Y 변화량)")
            ax_bar.set_title("H1-H5 가설별 ATE 비교  (PRavg 교란변수 통제)", fontsize=10)
            ax_bar.tick_params(axis="x", labelsize=8)
            green_p = plt.Rectangle((0,0),1,1, color="#2d6a4f", label="95% CI 유의")
            amber_p = plt.Rectangle((0,0),1,1, color="#b07040", label="95% CI 비유의")
            ax_bar.legend(handles=[green_p, amber_p], fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close()

        if cate_map:
            st.markdown("### CATE 분포  /  Individual Treatment Effects")
            n_valid = len(cate_map)
            fig_c, axes_c = plt.subplots(1, n_valid, figsize=(3.5 * n_valid, 3.8))
            if n_valid == 1:
                axes_c = [axes_c]
            palette = ["#1a1a2e","#2d6a4f","#5855A0","#B07040","#3A6B9A"]
            for idx, (hyp, (an, cate, ate, lb, ub)) in enumerate(cate_map.items()):
                axes_c[idx].hist(cate, bins=28, color=palette[idx % len(palette)], edgecolor="white", alpha=0.85)
                axes_c[idx].axvline(ate, color="#c0392b", lw=2, ls="--", label=f"ATE={ate:.3f}")
                axes_c[idx].axvline(0, color="#aaa", lw=1, ls=":")
                axes_c[idx].set_title(f"{hyp}: {an['t']}→{an['y']}", fontsize=8.5)
                axes_c[idx].set_xlabel("CATE", fontsize=8)
                axes_c[idx].legend(fontsize=7.5)
                neg_pct = (cate < 0).mean() * 100
                axes_c[idx].set_xlabel(f"CATE  (음수비율 {neg_pct:.0f}%)", fontsize=8)
            plt.suptitle("H1-H5 CATE 분포  (PRavg 교란변수 통제)", fontsize=10, y=1.02)
            plt.tight_layout()
            st.pyplot(fig_c)
            plt.close()

        if cate_map:
            st.markdown("### PRavg 수준별 CATE  /  CATE vs PR (교란변수)")
            st.markdown(
                '<div class="info-box" style="font-size:0.82rem">'
                'PRavg(인지된 위험) 수준에 따라 각 가설의 처치 효과(CATE)가 어떻게 달라지는지 확인.<br>'
                '기울기가 있으면 PR이 해당 경로를 조절(moderate)하고 있음을 시사.'
                '</div>', unsafe_allow_html=True
            )
            n_valid = len(cate_map)
            fig_pr, axes_pr = plt.subplots(1, n_valid, figsize=(3.5 * n_valid, 3.8))
            if n_valid == 1:
                axes_pr = [axes_pr]
            pr_vals = df["PRavg"].values
            for idx, (hyp, (an, cate, ate, lb, ub)) in enumerate(cate_map.items()):
                axes_pr[idx].scatter(pr_vals, cate, alpha=0.35, s=14, color=palette[idx % len(palette)])
                axes_pr[idx].axhline(0, color="#aaa", lw=1, ls=":")
                axes_pr[idx].axhline(ate, color="#c0392b", lw=1.5, ls="--", label=f"ATE={ate:.3f}")
                z = np.polyfit(pr_vals, cate, 1)
                p_fit = np.poly1d(z)
                x_line = np.linspace(pr_vals.min(), pr_vals.max(), 100)
                axes_pr[idx].plot(x_line, p_fit(x_line), color="#1a1a2e", lw=1.5, alpha=0.7)
                axes_pr[idx].set_xlabel("PRavg (인지된 위험)", fontsize=8)
                axes_pr[idx].set_ylabel("CATE", fontsize=8)
                axes_pr[idx].set_title(f"{hyp}: {an['t']}→{an['y']}", fontsize=8.5)
                axes_pr[idx].legend(fontsize=7.5)
            plt.suptitle("PRavg 수준별 CATE — PR의 조절 효과 확인", fontsize=10, y=1.02)
            plt.tight_layout()
            st.pyplot(fig_pr)
            plt.close()

        st.markdown("### 가설 채택/기각 요약  /  Hypothesis Supported?")
        hyp_summary = []
        for r in results7:
            if not isinstance(r["ATE"], float):
                continue
            ate_v = r["ATE"]
            lb_v = r["CI 하한"]
            ub_v = r["CI 상한"]
            sig = lb_v * ub_v > 0
            expected_pos = r["가설"] in ["H1","H2","H3","H4","H5"]
            direction_ok = ate_v > 0 if expected_pos else True
            verdict = "지지 (Supported)" if sig and direction_ok else ("방향 반대 (Reversed)" if sig and not direction_ok else "기각 (Not supported)")
            hyp_summary.append({
                "가설": r["가설"],
                "T → Y": f"{r['T (처치)'].split('(')[0].strip()} → {r['Y (결과)']}",
                "ATE": r["ATE"],
                "95% CI": f"[{lb_v:.4f}, {ub_v:.4f}]",
                "예측 방향": "양(+)",
                "실제 방향": r["방향"],
                "판정": verdict,
            })
        if hyp_summary:
            verdict_df = pd.DataFrame(hyp_summary)
            st.dataframe(verdict_df, use_container_width=True, hide_index=True)
            st.markdown(
                '<div class="info-box">'
                '<b>판정 기준:</b> 95% CI가 0을 미포함 + 양의 방향 = 가설 지지<br>'
                '데이터: LBS 설문 단면 데이터(n=332) — 통계 검정력에 한계 있음.<br>'
                'PRavg를 교란변수로 통제한 후의 순수 인과 효과 기준.',
                unsafe_allow_html=True,
            )

            # --- AI 해석 ---
            def _hyp_kr(s):
                # "Uavg → USEavg" 형태를 한글 라벨로
                parts = s.replace(" ", "").split("→")
                return " → ".join(explain.vlabel(pp) for pp in parts) if len(parts) == 2 else s
            _hyp = [(h["가설"], _hyp_kr(h["T → Y"]), h["판정"]) for h in hyp_summary]
            explain.show(
                f"step7_{_hyp}",
                f"위치정보(LBS) 앱 설문으로 연구가설 H1~H5를 인과추론으로 종합 검증했다. "
                f"(가설, 원인→결과, 판정) 목록은 {_hyp} 이다. "
                f"'지지'는 가설대로 효과가 확인됨, '기각'은 확인 안 됨, '방향 반대'는 예상과 반대 효과란 뜻이다. "
                f"위 변수 이름만 써서, 어떤 가설이 데이터로 뒷받침되고 어떤 것은 아닌지, 전체적으로 무엇을 알게 됐는지 3문장 이내로 설명하라.",
            )
