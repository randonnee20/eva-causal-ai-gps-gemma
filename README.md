# Causal AI Pipeline - LBS App Privacy Risk
# 위치정보서비스 앱 인지된 위험 인과 추론

## Setup / 설치

```bash
pip install -r requirements.txt
```

## Run / 실행

```bash
streamlit run app.py
```

브라우저에서 http://localhost:8501 열림

---

## Pipeline Steps / 파이프라인 단계

| Step | Name (EN) | Name (KR) | Action Required |
|------|-----------|-----------|-----------------|
| 0 | Data Overview | 데이터 개요 | 자동 (확인만) |
| 1 | Variable Definition | 변수 정의 | T / Y / W 선택 |
| 2 | DAG Construction | DAG 구성 | 화살표 검토/수정 |
| 3 | Identification | 식별 (DoWhy) | 버튼 실행 |streamlit run app.py

| 4 | Estimation | 추정 (EconML) | 버튼 실행 |
| 5 | Refutation | 반증 검증 | 버튼 실행 |
| 6 | Counterfactual | 반사실 추론 | 시나리오 설정 후 실행 |

## Default Variable Assignment / 기본 변수 설정

- T (Treatment): PR_2.6PR - 인지된 위험 이진 (0=low, 1=high)
- Y (Outcome): USEavg - 사용의도 평균
- W (Confounders): Uavg, SQavg, IFavg, SIavg

## Data / 데이터

- data/lbs_survey_332.csv : LBS 앱 설문 n=332 (2013)

## Notes / 참고

- Step 1 에서 선택한 T/Y/W가 이후 모든 단계에 적용됨
- Step 2 DAG는 연구 가설(H1~H6) 기반 기본값 제공, 사용자 수정 가능
- Step 3 식별 결과가 없으면 Step 5 Refutation은 자체 실행
- Step 4 추정 완료 후 Step 6 Counterfactual 사용 가능
