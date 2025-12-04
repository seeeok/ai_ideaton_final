# ============================================================
# XGBoost_model.py
# - 메뉴 비율 예측 모델 (카테고리 · 기온 · 학사일정 기반)
# - ai_ideaton.csv 기반 안정화 버전
# - NaN/inf 처리 · 문자열 입력 지원 · 안정적 one-hot 적용
# ============================================================

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
import os

CSV_PATH = "ai_ideaton.csv"
XGB_MODEL_PATH = "xgb_models.pkl"
FEATURE_COLS_PATH = "xgb_feature_cols.pkl"

TEST_SIZE = 0.2
NUM_BOOST_ROUND = 200
SEED = 42

TARGET_COLS = [
    "korean_ratio",
    "chinese_ratio",
    "japanese_ratio",
    "western_ratio",
]


# ============================================================
# 1) CSV 로드 + 전처리
# ============================================================
def load_data(path=CSV_PATH):
    """ai_ideaton.csv로부터 학습용 데이터셋 생성."""

    df = pd.read_csv(path)

    # 숫자형 강제 변환
    numeric_cols = [
        "korean_sales",
        "chinese_sales",
        "japanese_sales",
        "western_sales",
        "total_sales",
        "temperature",
        "is_exam",
        "is_festival",
        "is_vacation",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # total_sales가 0이거나 NaN이면 학습 제외
    df = df[df["total_sales"].notna() & (df["total_sales"] > 0)].copy()

    # NaN / inf 제거
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.dropna(subset=numeric_cols)

    # 비율 계산
    df["korean_ratio"] = df["korean_sales"] / df["total_sales"]
    df["chinese_ratio"] = df["chinese_sales"] / df["total_sales"]
    df["japanese_ratio"] = df["japanese_sales"] / df["total_sales"]
    df["western_ratio"] = df["western_sales"] / df["total_sales"]

    df[TARGET_COLS] = df[TARGET_COLS].clip(0, 1)

    # Feature 구성
    cat_cols = [
        "korean_category",
        "chinese_category",
        "japanese_category",
        "western_category",
    ]
    num_cols = ["temperature", "is_exam", "is_festival", "is_vacation"]

    X_num = df[num_cols].astype(float)
    X_cat = pd.get_dummies(df[cat_cols])

    X = pd.concat([X_num, X_cat], axis=1)
    y = df[TARGET_COLS].copy()

    feature_cols = list(X.columns)

    return X, y, feature_cols


# ============================================================
# 2) XGBoost 모델 학습
# ============================================================
def train_xgb_models():
    print("▶ Training XGBoost models...")

    X, y, feature_cols = load_data(CSV_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=True, random_state=SEED
    )

    models = {}

    for t in TARGET_COLS:
        print(f"  - Training {t}...")

        dtrain = xgb.DMatrix(X_train.values, label=y_train[t].values)
        dtest = xgb.DMatrix(X_test.values, label=y_test[t].values)

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "seed": SEED,
        }

        model = xgb.train(params, dtrain, NUM_BOOST_ROUND)

        pred = model.predict(dtest)
        mae = mean_absolute_error(y_test[t].values, pred)
        print(f"    → MAE = {mae:.4f}")

        models[t] = model

    # 저장
    with open(XGB_MODEL_PATH, "wb") as f:
        pickle.dump(models, f)

    with open(FEATURE_COLS_PATH, "wb") as f:
        pickle.dump(feature_cols, f)

    print("✔ XGBoost models saved.")

    return models, feature_cols


# ============================================================
# 3) 모델 불러오기
# ============================================================
def load_xgb_models():
    if not os.path.exists(XGB_MODEL_PATH):
        return None, None

    with open(XGB_MODEL_PATH, "rb") as f:
        models = pickle.load(f)
    with open(FEATURE_COLS_PATH, "rb") as f:
        feature_cols = pickle.load(f)

    return models, feature_cols


# ============================================================
# 4) 문자열 입력 → feature 벡터 변환 → 예측
# ============================================================
def predict_with_strings(
    total_sales,
    korean,
    chinese,
    japanese,
    western,
    temperature,
    is_exam,
    is_festival,
    is_vacation,
    models,
    feature_cols,
):
    """메뉴 이름 + 컨텍스트를 받아 비율 → 판매량 예측"""

    cat_cols = [
        "korean_category",
        "chinese_category",
        "japanese_category",
        "western_category",
    ]
    num_cols = ["temperature", "is_exam", "is_festival", "is_vacation"]

    row = pd.DataFrame([{
        "korean_category": korean,
        "chinese_category": chinese,
        "japanese_category": japanese,
        "western_category": western,
        "temperature": float(temperature),
        "is_exam": int(is_exam),
        "is_festival": int(is_festival),
        "is_vacation": int(is_vacation),
    }])

    X_num = row[num_cols].astype(float)
    X_cat = pd.get_dummies(row[cat_cols])

    X_row = pd.concat([X_num, X_cat], axis=1)
    X_row = X_row.reindex(columns=feature_cols, fill_value=0)

    drow = xgb.DMatrix(X_row.values)

    result = {}
    for t in TARGET_COLS:
        r = float(models[t].predict(drow)[0])
        r = max(0.0, min(1.0, r))  # 안정화

        key = t.replace("_ratio", "_sales")
        result[key] = int(round(r * total_sales))

    return result


# ============================================================
# 5) 단독 실행 방지
# ============================================================
if __name__ == "__main__":
    print("이 파일은 직접 실행하지 않습니다.")
    print("Streamlit 또는 final.py에서 import하여 사용하세요.")
