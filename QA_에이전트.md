# QA 에이전트 체크리스트
# Causal AI Pipeline — app.py 수정 후 필수 검증 절차

---

## 실행 방법

app.py 수정 후 아래 항목을 순서대로 확인한다.
자동 검증(터미널)과 수동 검증(Streamlit 화면)으로 구분된다.

---

## 1. 자동 검증 (터미널 실행)

### 1-1. Python 문법 검사
```bash
python -c "import ast; ast.parse(open('app.py').read()); print('Syntax OK')"
```
- 기대 결과: `Syntax OK`
- 실패 시: SyntaxError 위치 확인 후 수정

### 1-2. Import 및 데이터 로딩 검사
```bash
python -c "
import pandas as pd, numpy as np
df = pd.read_csv('data/lbs_survey_332.csv', encoding='cp949')
df.columns = df.columns.str.strip()
for c in ['Uavg','IFavg','SIavg','PRavg','SQavg','USEavg','ICUavg']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=['Uavg','IFavg','SIavg','PRavg','SQavg','USEavg','ICUavg'])
print('Rows:', len(df), '/ Cols:', len(df.columns))
assert len(df) >= 300, 'Data too small'
print('Data OK')
"
```
- 기대 결과: `Rows: 332 / Cols: ... / Data OK`

### 1-3. 핵심 변수 존재 확인
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/lbs_survey_332.csv', encoding='cp949')
df.columns = df.columns.str.strip()
required = ['Uavg','SQavg','IFavg','SIavg','PRavg','USEavg','ICUavg','PR_2.6PR','PR_3.5PR']
missing = [c for c in required if c not in df.columns]
print('Missing:', missing if missing else 'None')
assert not missing, f'Missing columns: {missing}'
print('Variables OK')
"
```

### 1-4. Step 7 분석 로직 검증 (H1, H5 대표 실행)
```bash
python -c "
import pandas as pd, numpy as np
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor
df = pd.read_csv('data/lbs_survey_332.csv', encoding='cp949')
df.columns = df.columns.str.strip()
for c in ['Uavg','SQavg','IFavg','SIavg','PRavg','USEavg','ICUavg']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna()
tests = [
    ('H1', 'Uavg',   ['PRavg','SQavg','IFavg','SIavg'], 'USEavg'),
    ('H5', 'USEavg', ['PRavg','Uavg','SQavg','IFavg','SIavg'], 'ICUavg'),
]
for hyp, t, w, y in tests:
    est = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=50),
        model_t=GradientBoostingRegressor(n_estimators=50),
        n_estimators=100, min_samples_leaf=10, cv=3, random_state=42
    )
    est.fit(df[y].values, df[t].values, X=df[w].values)
    ate = float(est.ate(df[w].values))
    lb, ub = est.ate_interval(df[w].values, alpha=0.05)
    sig = 'sig' if lb*ub > 0 else 'not sig'
    print(f'{hyp}: ATE={ate:.4f} [{float(lb):.4f},{float(ub):.4f}] {sig}')
print('Step7 logic OK')
"
```

### 1-5. "논문" 문구 잔존 검사 (수정 후 필수 실행)
```bash
python -c "
with open('app.py', encoding='utf-8') as f:
    lines = f.readlines()
hits = [(i+1, l.rstrip()) for i, l in enumerate(lines) if '논문' in l]
print(f'논문 잔존: {len(hits)}건')
for lineno, text in hits:
    print(f'  L{lineno}: {text[:80]}')
"
```
- 기대 결과: `논문 잔존: 0건`

### 1-6. T 하드코딩 레이블 검사
```bash
grep -n "인지된 위험을\|인지된위험(T=\|인지된 위험(T" app.py
```
- 기대 결과: 출력 없음 (0건)

---

## 2. UI 수동 검증 항목 (Streamlit 실행 후 화면 확인)

```bash
streamlit run app.py
```

### Step 0 - Data Overview
- [ ] Variable Map 테이블에서 PRavg Causal Role이 `Confounder W`로 표시되는가
- [ ] Uavg/SQavg/IFavg/SIavg의 Causal Role이 `Treatment T (H1~H4)` 형태로 표시되는가
- [ ] 상관행렬 7개 변수 모두 표시되는가

### Step 1 - Variable Definition
- [ ] T 선택 목록에 Uavg, SQavg, IFavg, SIavg, USEavg 가 포함되어 있는가
- [ ] T 선택 목록에 PR_2.6PR, PR_3.5PR, PRavg 도 포함되어 있는가
- [ ] T 선택 목록 라벨에 "논문" 문구가 없는가 (H1, H2 등만 표시)
- [ ] W(혼란변수) 목록에 PRavg 체크박스가 있는가
- [ ] T=Uavg 선택 시 W에서 Uavg 자동 제외되는가

### Step 2 - DAG Construction
- [ ] T=Uavg, W=[PRavg, SQavg, IFavg, SIavg] 선택 시 DAG가 올바르게 그려지는가
- [ ] PRavg → Uavg, PRavg → USEavg 화살표가 존재하는가 (W→T 모드)

### Step 3 - Identification
- [ ] T=Uavg (연속형) 설정에서 DoWhy 식별이 완료되는가
- [ ] Backdoor set에 PRavg가 포함되는가

### Step 4 - Estimation
- [ ] ATE 결과 설명 문장이 "T={선택한변수}가 Y를..." 형태로 표시되는가 (하드코딩 아님)
- [ ] binary T(PR_2.6PR) 선택 시도 정상 작동하는가

### Step 5 - Refutation
- [ ] 판정 기준 테이블이 실행 버튼 위에 표시되는가
- [ ] 결과 테이블에 `p_value` 컬럼이 있는가
- [ ] 결과 테이블에 `판정 기준` 컬럼이 있는가 (어떤 기준으로 PASS/FAIL 결정됐는지)
- [ ] p_value 미지원 시 대체 기준으로 자동 전환되는가

### Step 6 - Counterfactual
- [ ] 상단 설명 문구가 "T={선택한변수}이 달랐다면?" 형태로 표시되는가
- [ ] Factual T 입력 라벨이 "실제 {선택한변수} 값" 형태로 표시되는가
- [ ] 결과 설명 문구가 "{선택한변수}를 ...에서 ...으로 바꿨을 때" 형태로 표시되는가
- [ ] T=PR_2.6PR (이진) 선택 시 레이블이 "1 (높음) / 0 (낮음)" 형태로 표시되는가

### Step 7 - H1-H5 인과 분석
- [ ] 사이드바 탭 이름이 "H1-H5 인과 분석"으로 표시되는가 ("논문" 없음)
- [ ] 헤더에 "논문" 문구가 없는가
- [ ] 전체 인과 구조 DAG (통합 DAG) 1개가 상단에 표시되는가
- [ ] 통합 DAG에 Uavg/SQavg/IFavg/SIavg, PRavg, USEavg, ICUavg 7개 노드가 있는가
- [ ] 통합 DAG에 PRavg → 각 독립변수 빨간 화살표가 있는가
- [ ] 통합 DAG에 USEavg → ICUavg 녹색 화살표가 있는가
- [ ] 구분선(divider) 아래에 H1~H4 개별 DAG 4개가 표시되는가
- [ ] H5 DAG가 별도로 표시되는가
- [ ] 각 개별 DAG에서 PRavg → T 빨간 화살표가 있는가
- [ ] "Run All H1-H5 Analyses" 버튼 클릭 후 프로그레스바가 표시되는가
- [ ] 5개 분석 완료 후 ATE 비교 테이블이 표시되는가 (5행)
- [ ] 가설 채택/기각 판정 테이블이 표시되는가

---

## 3. 변수 역할 일관성 검증

```bash
# "인지된 위험"이 T로 고정 표현된 부분 찾기
grep -n "처치 변수.*인지된 위험\|Treatment T.*인지된 위험\|인지된 위험을\|인지된위험(T=" app.py

# 논문 문구 잔존 확인
grep -n "논문" app.py

# Step 7 분석 정의 확인
grep -n "ANALYSES\|hyp.*H[1-5]" app.py
```

---

## 3-B. AI 해석 기능 검증 (신규 — llm.py / explain.py)

### 3-B-1. LLM 모듈 문법·모드 감지
```bash
python -c "import ast; [ast.parse(open(f,encoding='utf-8').read()) for f in ['app.py','llm.py','explain.py']]; print('Syntax OK')"
python -c "import llm; print('mode:', llm.detect_mode(), '/', llm.status_label())"
```
- 기대: `Syntax OK` / 로컬은 `ollama / 로컬 gemma3:4b`, 클라우드는 `gemini / Gemini API`

### 3-B-2. 변수 사전 일관성
```bash
python -c "import explain; print(explain.vlabels(['Uavg','SQavg','IFavg','SIavg','PRavg','USEavg','ICUavg']))"
```
- 기대: `유용성, 시스템 품질, 인터페이스 품질, 사회적 영향, 인지된 위험, 사용의도, 지속사용의도`

### 3-B-3. 해석 정확도 체크리스트 (실제 생성 결과 육안 검토)
- [ ] 영문 약어(Uavg 등)가 그대로 노출되지 않는가 (한글 라벨로 변환)
- [ ] 데이터에 없는 소재(가격/할인/광고/매출)를 지어내지 않는가
- [ ] 원인(T)과 결과(Y)를 뒤바꿔 설명하지 않는가
- [ ] 효과 방향(올라간다/내려간다)이 실제 ATE 부호와 일치하는가
- [ ] 통계적으로 비유의한 효과를 "뚜렷한 효과 없음"으로 정직하게 설명하는가
- [ ] Step 6 서브그룹 필터(낮은편/높은편)를 올바르게 반영하는가
- [ ] '사용의도'를 '앱 사용 시간/빈도'로 잘못 바꾸지 않는가

### 3-B-4. AI 해석 삽입 지점 (8곳) 존재 확인
```bash
grep -c "explain.show(" app.py
```
- 기대: 8 (Step0,1,2,3,4-multi,4-single,5,6,7 중 Step4가 2분기이므로 총 8~9)

---

## 4. 수정 이력 기록

| 날짜 | 수정 내용 | 검증 결과 | 담당 |
|------|-----------|-----------|------|
| 2026-06-14 | AI 자동 해석 기능 추가 (llm.py/explain.py, Step0~7 삽입) | Syntax OK, Ollama 종단 OK, 부팅 HTTP 200 | Claude |
| 2026-06-14 | AI 해석 정확도 2·3차 보완 (변수사전 내장, 방향 Python 확정, 유의성/서브그룹 프레이밍) | 6개 문제 케이스 해소 확인 | Claude |
| 2026-06-14 | QA 1-6 위반 수정: Step4 매개분석 info-box "인지된 위험" 하드코딩 → 동적 라벨(explain.vlabel(T)) | 1-6 0건, 부팅 HTTP 200 | Claude |
| 2026-06-14 | AI 해석 4차: '인지된 위험 증가'→'안전 증가' 반대의미 의역 차단 (SYSTEM 규칙+사전 명시) | 위험↑/↓ 두 케이스 재검증 OK | Claude |
| 2026-06-10 | Step 7 H1-H5 인과 분석 탭 추가 | Syntax OK, H1/H5 추정 테스트 통과 | Claude |
| 2026-06-10 | H1-H5 개별 DAG (PR→T 화살표 추가) | 시각 확인 완료 | Claude |
| 2026-06-10 | Step 1 변수 정의 UI 수정 (T/W 재설계) | Syntax OK, Label 검증 통과 | Claude |
| 2026-06-10 | Step 0 Variable Map Causal Role 이중 표시 추가 | Syntax OK | Claude |
| 2026-06-10 | Step 5: p_value 기반 PASS/FAIL 판정 교체 + 기준 테이블 추가 | Syntax OK | Claude |
| 2026-06-10 | Step 6: T 레이블 동적화 (하드코딩 "인지된 위험" 제거) | Syntax OK, 논문 문구 0건 | Claude |
| 2026-06-10 | Step 7: 통합 DAG 추가 (전체 인과 구조 단일 그래프) | Syntax OK | Claude |
| 2026-06-10 | Step 1: T 복수 체크박스 재설계, 패러다임 테이블 제거 | Syntax OK, T_list 구조 확인 | Claude |
| 2026-06-10 | Step 4: 다중 T 루프 — ATE 비교 테이블/바차트/CATE 분포 추가 | Syntax OK | Claude |
| 2026-06-10 | Steps 2/3/5/6: T_list 대응, 다중 T 안내 문구 추가 | Syntax OK | Claude |
| 2026-06-10 | 통합 DAG: 한글 노드 레이블 영문 전용으로 교체 | 한글 잔존 0건 | Claude |
| 2026-06-10 | 전체 "논문" 문구 제거 (Causal AI 프레임으로 통일) | 논문 잔존 0건 확인 | Claude |

---

## 5. 주요 설계 원칙 (변경 금지 사항)

| 원칙 | 내용 |
|------|------|
| PR 역할 | Step 1~6: 선택 가능 T 또는 W / Step 7: 고정 교란변수(W) |
| T 연속형 처리 | t_is_binary 감지 → False이면 GradientBoostingRegressor 사용 |
| W-T 중복 방지 | T로 선택된 변수는 W 체크박스에서 비활성화 |
| DAG PR 화살표 | Step 7 개별 DAG에서 PRavg → T 및 PRavg → Y 반드시 존재 |
| 통합 DAG | Step 7 상단에 7-노드 전체 구조 DAG 항상 표시 |
| 가설 방향 | H1~H5 모두 양(+)의 ATE 예측 방향 |
| T 레이블 | Step 4/6 설명 텍스트는 반드시 session_state T 값 동적 참조 |
| Step 5 판정 | p_value > 0.05 = PASS (p_value 미지원 시 비율 기반 대체) |
| "논문" 문구 | UI 전체에서 사용 금지 — H1/H2 등 가설 번호로 대체 |
