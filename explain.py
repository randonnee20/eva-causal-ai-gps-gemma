"""explain.py — Step별 AI 해석 생성 + Streamlit 렌더링

핵심 설계:
  1) 변수 사전(VAR_LABELS)을 SYSTEM 프롬프트에 내장 → AI가 변수 의미를 안다.
  2) 이 데이터에 없는 소재(가격/광고/매출 등) 환각을 강하게 금지한다.
  3) 방향·부호 해석은 가급적 app.py(Python)에서 미리 계산해 완성 문장으로 넘긴다.
"""

import hashlib
import streamlit as st

import llm

# ---------------------------------------------------------------------------
# 변수 사전 — 모든 Step이 공유하는 단일 진실
# ---------------------------------------------------------------------------
VAR_LABELS = {
    "Uavg": "유용성",
    "SQavg": "시스템 품질",
    "IFavg": "인터페이스 품질",
    "SIavg": "사회적 영향",
    "PRavg": "인지된 위험",
    "PR_2.6PR": "인지된 위험(높음/낮음)",
    "PR_3.5PR": "인지된 위험(높음/낮음)",
    "USEavg": "사용의도",
    "ICUavg": "지속사용의도",
}


def vlabel(col):
    return VAR_LABELS.get(str(col), str(col))


def vlabels(cols):
    return ", ".join(vlabel(c) for c in cols) if cols else "없음"


# ---------------------------------------------------------------------------
# 시스템 프롬프트 — 변수 사전 + 환각 금지 + 톤 규칙
# ---------------------------------------------------------------------------
SYSTEM = """당신은 인과추론(Causal Inference) 분석 결과를 통계를 전혀 모르는 일반인에게 쉽게 풀어 설명하는 전문가다.

[이 데이터에 대한 절대 규칙 — 반드시 지켜라]
- 이 데이터는 '위치정보서비스(LBS) 모바일 앱' 이용자 332명의 설문이다.
- 등장하는 변수는 아래 '변수 사전'에 있는 7가지가 전부다.
- 사전에 없는 소재(가격, 할인, 광고, 매출, 금리, 프로모션 등)를 절대 지어내지 마라. 일반적인 예시를 끌어오지 마라.
- 영문 약어(Uavg, USEavg, PRavg 등)를 그대로 쓰지 말고, 반드시 아래 사전의 한글 이름으로 바꿔 불러라.
- '앱 사용 시간', '앱 사용 빈도' 같은 표현을 쓰지 마라. 이 설문은 사용'의도'(마음)를 물은 것이다.

[변수 사전]
- 유용성 : 앱이 얼마나 쓸모 있다고 느끼는지
- 시스템 품질 : 앱이 얼마나 안정적이고 빠른지
- 인터페이스 품질 : 화면이 얼마나 보기 쉽고 쓰기 편한지
- 사회적 영향 : 주변 사람들이 써서 영향받는 정도
- 인지된 위험 : 개인정보 유출 등 위험하다고 느끼는 정도 (값이 높을수록 '더 위험하다 = 덜 안전하다'고 느끼는 것)
- 사용의도 : 앞으로 이 앱을 쓰려는 마음
- 지속사용의도 : 앞으로도 계속 쓰려는 마음

[용어 풀이]
- 처치(T) : 효과를 알아보려고 바꿔 보는 '원인' 변수
- 결과(Y) : 측정하려는 목표(KPI)
- 교란변수(W) : 원인과 결과 양쪽에 동시에 영향을 주는 '배경 요인'
- 상관관계 : 그냥 함께 움직인다 (원인인지 모름) / 인과관계 : 실제로 바꾸면 결과가 달라진다
- 효과가 '확실하다' = 신뢰구간 안에 0이 없다 / '불확실하다' = 0이 들어 있다
- 반사실 : "만약 그때 다르게 했다면 어땠을까"를 추정하는 것

[작성 규칙]
- 한국어, 2~3문장, 중학생도 이해할 수준으로 쉽게.
- 주어진 숫자와 방향만 사용하라. 주어지지 않은 숫자를 지어내거나 '몇 명 중 몇 명' 식으로 바꾸지 마라.
- 주어진 사실의 방향(오른다/내린다, 늘어난다/줄어든다)을 절대 반대로 바꾸지 마라.
- 그 단계에서 '주어진 변수'만 언급하라. 주어지지 않은 다른 변수(예: 시스템 품질, 인터페이스 품질 등)를 끌어와 원인으로 추측하지 마라.
- 변수를 반대 의미의 단어로 바꿔 말하지 마라. 특히 '인지된 위험'을 '안전/안심'으로 바꾸지 마라. '인지된 위험이 높아짐'은 '더 위험하다고 느낌'이지 '안전해짐'이 절대 아니다.
- 마지막 한 문장으로 '그래서 비즈니스에 어떤 의미인지'를 덧붙여라.
- 불릿/제목/머리말 없이 자연스러운 문단으로만 답하라."""


# ---------------------------------------------------------------------------
# 렌더링
# ---------------------------------------------------------------------------
def _box(text, label):
    st.markdown(
        f'''<div style="border:1px solid #d8e0ea;border-radius:8px;margin:0.6rem 0 0.2rem;overflow:hidden">
  <div style="background:#2C3E6B;padding:6px 12px;color:#fff;font-size:0.8rem;font-weight:600">
    AI 해석 <span style="float:right;font-weight:400;opacity:0.75;font-size:0.72rem">{label}</span>
  </div>
  <div style="padding:11px 14px;font-size:0.9rem;line-height:1.65;color:#1a1a2e;background:#fafbfd">{text}</div>
</div>''',
        unsafe_allow_html=True,
    )


def show(cache_key, prompt, temperature=0.3):
    """결과 아래에 AI 해석을 자동 표시. 동일 내용은 캐시해 재호출하지 않는다."""
    if not st.session_state.get("ai_explain_on", True):
        return
    h = hashlib.md5((cache_key + "||" + prompt).encode("utf-8")).hexdigest()
    sk = f"_ai_{h}"
    if sk not in st.session_state:
        try:
            with st.spinner("AI가 결과를 쉬운 말로 해석하는 중..."):
                st.session_state[sk] = llm.generate(prompt, system=SYSTEM, temperature=temperature, max_tokens=1000)
        except Exception as e:
            st.session_state[sk] = f"(AI 해석을 생성하지 못했습니다: {e})"
    _box(st.session_state[sk], llm.status_label())
