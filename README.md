# 🤰 fertility-predictor

> 난임 환자 시술 데이터를 기반으로 임신 성공 여부를 예측하는 AI 모델  
> Dacon 해커톤 — 난임 환자 대상 임신 성공 여부 예측 AI 모델 개발

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Latest-green.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 프로젝트 개요

난임 시술 환자 데이터를 분석하여 **임신 성공 여부(출산까지 성공적으로 진행된 임신)**를 예측합니다.  
최소한의 시술로 임신 성공 가능성을 높이기 위해 주요 영향 요인을 도출하고, ROC-AUC 기준 최적 예측 모델을 개발합니다.

- **평가 지표**: ROC-AUC
- **타겟 변수**: `임신 성공 여부` (0: 실패, 1: 출산 성공)
- **데이터 규모**: Train 256,351행 / Test 90,067행 / 68개 컬럼

---

## 👥 팀 구성

| 이름 | 역할 |
|------|------|
| - | EDA, 피처 엔지니어링 |
| - | 모델링, 하이퍼파라미터 튜닝 |
| - | 전처리, 앙상블 |

---

## 📂 프로젝트 구조

```
fertility-predictor/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── notebooks/
│   ├── 01_eda.ipynb              # 탐색적 데이터 분석
│   ├── 02_preprocessing.ipynb    # 전처리 및 피처 엔지니어링
│   ├── 03_modeling.ipynb         # 모델 학습 및 실험
│   └── 04_ensemble.ipynb         # 앙상블 및 최종 제출
│
├── src/
│   ├── preprocessing.py          # 전처리 함수 모음
│   ├── features.py               # 피처 엔지니어링 함수
│   ├── train.py                  # 모델 학습
│   └── predict.py                # 예측 생성
│
├── submissions/                  # 제출 파일 저장
├── requirements.txt
└── README.md
```

---

## 📊 데이터 설명

### 주요 변수

| 변수 | 타입 | 설명 |
|------|------|------|
| `시술 당시 나이` | 범주형 | 만18-34세 ~ 만45-50세 |
| `시술 유형` | 범주형 | IVF (체외수정) / DI (인공수정) |
| `특정 시술 유형` | 범주형 | ICSI, IUI, FER, BLASTOCYST 등 복합 기재 |
| `배란 자극 여부` | 이진 | 배란 자극 치료 사용 여부 |
| `총 생성 배아 수` | 수치형 | 해당 시술에서 생성된 배아 총 수 |
| `미세주입된 난자 수` | 수치형 | ICSI로 처리된 난자 수 |
| `이식된 배아 수` | 수치형 | 자궁에 이식된 배아 총 수 |
| `저장된 배아 수` | 수치형 | 동결 보관된 배아 수 |
| `해동된 배아 수` | 수치형 | FER 시술 시 해동된 배아 수 |
| `수집된 신선 난자 수` | 수치형 | 채취된 신선 난자 수 |
| `난자 채취 경과일` | 수치형 | 기준일로부터 난자 채취까지 일수 |
| `배아 이식 경과일` | 수치형 | 기준일로부터 배아 이식까지 일수 |
| `PGS 시술 여부` | 이진 | 착상 전 유전 선별 검사 여부 |
| `PGD 시술 여부` | 이진 | 착상 전 유전 진단 여부 |
| `동결 배아 사용 여부` | 이진 | FER 시술 여부 간접 지표 |
| `불임 원인` (17개) | 이진 | 남성/여성/부부 요인별 불임 원인 |
| `총 시술 횟수` 외 | 범주형 | 0회 ~ 6회 이상, 시술 이력 |
| **`임신 성공 여부`** | **이진** | **타겟 변수 (0: 실패, 1: 성공)** |

### 시술 경로 구조

```
시술 유형
├── IVF (체외수정)
│   ├── 신선 배아 이식  → 난자채취 → 수정(IVF/ICSI) → 배양 → 이식
│   └── FER (동결 배아 이식) → 해동 → 이식
└── DI (인공수정 / IUI)
    └── 정자 주입 → 체내 수정 (배아 관련 컬럼 대부분 결측 = 정상)
```

> ⚠️ **구조적 결측치 주의**: DI 시술에서 배아/난자 수치 컬럼의 결측은 랜덤 결측이 아닌 정상 결측입니다.  
> 시술 유형별로 분리하여 처리해야 합니다.

---

## 🔧 데이터 전처리

### 1. 범주형 인코딩

- **시술 횟수 컬럼** (0회~6회 이상): `6회 이상 → 6` 으로 수치 변환 후 Label Encoding
- **나이 컬럼**: 순서형 인코딩 (만18-34세=0, ..., 만45-50세=5, 알 수 없음=6)
- **`특정 시술 유형`**: `/`, `:` 구분자 기준 멀티핫 인코딩

```python
keywords = ['IVF', 'ICSI', 'IUI', 'ICI', 'FER', 'BLASTOCYST', 'AH', 'GIFT', 'IVI']
for kw in keywords:
    df[f'proc_{kw}'] = df['특정 시술 유형'].str.contains(kw, na=False).astype(int)
```

### 2. 결측치 처리

- **DI 시술 배아/난자 컬럼**: 0으로 채우지 않고 시술 유형 그룹별 처리
- **수치형 결측치**: 시술 유형 그룹 내 중앙값으로 대체
- **test 데이터 결측치**: test 데이터셋 통계값만 활용 (Data Leakage 방지)

### 3. 파생 피처

| 피처명 | 계산식 | 의미 |
|--------|--------|------|
| `수정률` | 총 생성 배아 수 / 혼합된 난자 수 | 수정 성공률 |
| `이식률` | 이식된 배아 수 / 총 생성 배아 수 | 이식 선택률 |
| `배양기간` | 배아 이식 경과일 - 난자 혼합 경과일 | 5일↑ = 배반포 이식 가능성 |
| `has_blastocyst` | 특정 시술 유형에 BLASTOCYST 포함 | 배반포 이식 여부 |
| `icsi_수정률` | ICSI 생성 배아 수 / 미세주입 난자 수 | ICSI 효율 |
| `transfer_type` | 신선/동결 배아 사용 여부 조합 | 시술 경로 분류 |
| `genetic_test_combo` | PGS×2 + PGD | 유전 검사 조합 (0~3) |
| `시술_임신_비율` | 총 임신 횟수 / (총 시술 횟수 + 1) | 과거 성공률 |
| `proc_complexity` | 시술 키워드 개수 합산 | 시술 복잡도 |

---

## 🤖 모델링

### 사용 모델

- **LightGBM** — 기본 베이스라인
- **XGBoost** — 비교 실험
- **CatBoost** — 범주형 변수 처리 특화
- **TabNet** — 정형 데이터 딥러닝 (앙상블 재료)
- **Voting/Stacking Ensemble** — 최종 제출

### 학습 전략

- Stratified K-Fold (5-fold) 교차 검증
- Optuna 하이퍼파라미터 최적화
- 클래스 불균형 처리: `scale_pos_weight` 또는 `class_weight`

---

## 📈 실험 결과

| 모델 | CV AUC | Public AUC | 비고 |
|------|--------|------------|------|
| LightGBM (baseline) | - | - | 추후 업데이트 |
| XGBoost | - | - | 추후 업데이트 |
| CatBoost | - | - | 추후 업데이트 |
| LightGBM + 피처 엔지니어링 | - | - | 추후 업데이트 |
| Stacking Ensemble | - | - | 추후 업데이트 |

---

## 🏆 최종 모델

- **선택 모델**: 추후 업데이트
- **선택 이유**: 추후 업데이트
- **최종 Public AUC**: -

---

## ⚙️ 실행 방법

### 환경 설치

```bash
git clone https://github.com/YOUR_USERNAME/fertility-predictor.git
cd fertility-predictor
pip install -r requirements.txt
```

### 데이터 준비

```
data/ 폴더에 아래 파일을 넣어주세요:
- train.csv
- test.csv
- sample_submission.csv
```

### 모델 학습

```bash
python src/train.py
```

### 예측 생성

```bash
python src/predict.py
# submissions/ 폴더에 제출 파일 생성됩니다
```

---

## 🛠️ 사용 기술

| 분류 | 라이브러리 |
|------|-----------|
| 데이터 처리 | `pandas`, `numpy` |
| 시각화 | `matplotlib`, `seaborn` |
| 머신러닝 | `lightgbm`, `xgboost`, `catboost` |
| 딥러닝 | `pytorch-tabnet` |
| 하이퍼파라미터 최적화 | `optuna` |
| 평가 | `scikit-learn` |

---

## 🔭 향후 개선 방향

- [ ] 시술 유형별 서브모델 학습 후 앙상블
- [ ] 불임 원인 그룹화 피처 추가
- [ ] 나이 × 시술 횟수 상호작용 피처 실험
- [ ] Pseudo Labeling 적용
- [ ] Neural Network 기반 임베딩 피처 활용

---

## 📜 라이선스

MIT License
