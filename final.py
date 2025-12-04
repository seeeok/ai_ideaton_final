# ============================================================
# final.py  (Prophet + XGB + 종합 평가 시스템)
# ============================================================

import pandas as pd
from datetime import datetime, timedelta
from typing import Union, Dict, Any

from prophet_model import (
    load_prophet_model,
    train_prophet_model,
)

from XGBoost_model import (
    load_xgb_models,
    train_xgb_models,
    predict_with_strings,
)


# ============================================================
# 학사일정 헬퍼
# ============================================================
def is_exam_day(dt: pd.Timestamp) -> int:
    exams = [(4, 15, 21), (6, 10, 16), (10, 15, 21), (12, 10, 16)]
    for m, s, e in exams:
        if dt.month == m and s <= dt.day <= e:
            return 1
    return 0


def is_festival_day(dt: pd.Timestamp) -> int:
    if dt.month not in [5, 9]:
        return 0

    next_month = dt.replace(
        year=dt.year + (1 if dt.month == 12 else 0),
        month=1 if dt.month == 12 else dt.month + 1,
        day=1,
    )
    last_day = next_month - timedelta(days=1)

    d = last_day
    while d.weekday() != 4:  # 금요일
        d -= timedelta(days=1)

    wed, thu, fri = d - timedelta(days=2), d - timedelta(days=1), d
    return 1 if dt.date() in [wed.date(), thu.date(), fri.date()] else 0


def is_vacation_day(dt: pd.Timestamp) -> int:
    m, d = dt.month, dt.day
    if (m == 6 and 1 <= d <= 20) or (m == 12 and 1 <= d <= 20):
        return 0
    if m in [3, 9, 4, 5, 10, 11]:
        return 0
    return 1


# ============================================================
# Prophet / XGB 로드 헬퍼
# ============================================================
def _get_prophet_model_and_forecast():
    model, forecast = load_prophet_model()
    if model is None or forecast is None:
        model, forecast = train_prophet_model()
    return model, forecast


def _get_xgb_models_and_features():
    models, feature_cols = load_xgb_models()
    if models is None or feature_cols is None:
        models, feature_cols = train_xgb_models()
    return models, feature_cols


# ============================================================
# 예측 함수
# ============================================================
def predict_total_and_menus(
    date_value: Union[str, datetime, pd.Timestamp],
    korean_menu: str,
    chinese_menu: str,
    japanese_menu: str,
    western_menu: str,
    temperature: Union[int, float],
) -> Dict[str, Any]:

    try:
        target_ts = pd.to_datetime(date_value)
    except Exception:
        return {"error": f"날짜 오류: {date_value}"}

    date_str = target_ts.strftime("%Y-%m-%d")

    _, forecast = _get_prophet_model_and_forecast()
    row = forecast[forecast["ds"] == target_ts]

    if row.empty:
        return {"error": f"{date_str} 예측값 없음"}

    yhat = round(row["yhat"].values[0])
    total_sales = max(0, int(yhat))

    if target_ts.weekday() >= 5:  # 주말
        total_sales = 0

    models, feature_cols = _get_xgb_models_and_features()

    exam_flag = is_exam_day(target_ts)
    fest_flag = is_festival_day(target_ts)
    vac_flag = is_vacation_day(target_ts)

    result = predict_with_strings(
        total_sales=total_sales,
        korean=korean_menu,
        chinese=chinese_menu,
        japanese=japanese_menu,
        western=western_menu,
        temperature=temperature,
        is_exam=exam_flag,
        is_festival=fest_flag,
        is_vacation=vac_flag,
        models=models,
        feature_cols=feature_cols,
    )

    return {
        "date": date_str,
        "total_sales": total_sales,
        "korean_sales": result.get("korean_sales", 0),
        "chinese_sales": result.get("chinese_sales", 0),
        "japanese_sales": result.get("japanese_sales", 0),
        "western_sales": result.get("western_sales", 0),
        "is_exam": exam_flag,
        "is_festival": fest_flag,
        "is_vacation": vac_flag,
    }


# ============================================================
# 메뉴 절대 MAE
# ============================================================
def compute_menu_mae(csv_path="ai_ideaton.csv"):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    cutoff = df["date"].max() - timedelta(days=60)
    df_use = df[(df["date"] >= cutoff) & (df["total_sales"] > 0)]

    if df_use.empty:
        return None

    diff = {"k": [], "c": [], "j": [], "w": []}

    for _, row in df_use.iterrows():
        pred = predict_total_and_menus(
            row["date"],
            row["korean_category"],
            row["chinese_category"],
            row["japanese_category"],
            row["western_category"],
            row["temperature"],
        )
        if "error" in pred:
            continue

        diff["k"].append(abs(row["korean_sales"] - pred["korean_sales"]))
        diff["c"].append(abs(row["chinese_sales"] - pred["chinese_sales"]))
        diff["j"].append(abs(row["japanese_sales"] - pred["japanese_sales"]))
        diff["w"].append(abs(row["western_sales"] - pred["western_sales"]))

    return {
        "korean": sum(diff["k"]) / len(diff["k"]),
        "chinese": sum(diff["c"]) / len(diff["c"]),
        "japanese": sum(diff["j"]) / len(diff["j"]),
        "western": sum(diff["w"]) / len(diff["w"]),
    }


# ============================================================
# 메뉴 비율 MAE
# ============================================================
def compute_menu_ratio_mae(csv_path="ai_ideaton.csv"):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    cutoff = df["date"].max() - timedelta(days=60)
    df_use = df[(df["date"] >= cutoff) & (df["total_sales"] > 0)]

    if df_use.empty:
        return None

    diff = {"k": [], "c": [], "j": [], "w": []}

    for _, row in df_use.iterrows():
        pred = predict_total_and_menus(
            row["date"],
            row["korean_category"],
            row["chinese_category"],
            row["japanese_category"],
            row["western_category"],
            row["temperature"],
        )
        if "error" in pred:
            continue

        act_total = row["total_sales"]
        pred_total = max(pred["total_sales"], 1)

        act_ratio = {
            "k": row["korean_sales"] / act_total,
            "c": row["chinese_sales"] / act_total,
            "j": row["japanese_sales"] / act_total,
            "w": row["western_sales"] / act_total,
        }
        pred_ratio = {
            "k": pred["korean_sales"] / pred_total,
            "c": pred["chinese_sales"] / pred_total,
            "j": pred["japanese_sales"] / pred_total,
            "w": pred["western_sales"] / pred_total,
        }

        for key in diff:
            diff[key].append(abs(act_ratio[key] - pred_ratio[key]))

    return {
        "korean_ratio_mae": sum(diff["k"]) / len(diff["k"]),
        "chinese_ratio_mae": sum(diff["c"]) / len(diff["c"]),
        "japanese_ratio_mae": sum(diff["j"]) / len(diff["j"]),
        "western_ratio_mae": sum(diff["w"]) / len(diff["w"]),
    }


# ============================================================
# 총 판매량 MAE / MAPE
# ============================================================
def compute_total_mae(csv_path="ai_ideaton.csv"):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    cutoff = df["date"].max() - timedelta(days=60)
    df_use = df[(df["date"] >= cutoff) & (df["total_sales"] > 0)]

    if df_use.empty:
        return None

    _, forecast = _get_prophet_model_and_forecast()

    mae_list, mape_list = [], []

    for _, row in df_use.iterrows():
        pred_row = forecast[forecast["ds"] == row["date"]]
        if pred_row.empty:
            continue

        yhat = max(pred_row["yhat"].values[0], 0)
        actual = row["total_sales"]

        mae_list.append(abs(actual - yhat))
        mape_list.append(abs(actual - yhat) / actual)

    return {
        "total_mae": sum(mae_list) / len(mae_list),
        "total_mape": 100 * sum(mape_list) / len(mape_list),
    }


# ========================================================
