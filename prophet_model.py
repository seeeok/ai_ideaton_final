# ============================================================
# prophet_model.py
# - Prophet 총 판매량 예측 모델
# - 회귀변수: is_exam, is_festival, is_vacation
# - 365일 미래 forecast 저장
# ============================================================

import pandas as pd
from prophet import Prophet
from datetime import timedelta
import pickle
import os

CSV_PATH = "ai_ideaton.csv"
PROPHET_MODEL_PATH = "prophet_model.pkl"
FORECAST_PATH = "prophet_forecast.pkl"


# ============================================================
# 1) CSV → Prophet 입력 형식 변환
# ============================================================
def load_prophet_dataframe(csv_path=CSV_PATH):
    if not os.path.exists(csv_path):
        print(f"❌ CSV 파일 없음: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    df_prophet = pd.DataFrame({
        "ds": pd.to_datetime(df["date"]),
        "y": df["total_sales"],
        "is_exam": df["is_exam"].astype(float),
        "is_festival": df["is_festival"].astype(float),
        "is_vacation": df["is_vacation"].astype(float),
    })

    return df_prophet


# ============================================================
# 2) Prophet 모델 학습
# ============================================================
def train_prophet_model(csv_path=CSV_PATH):
    print("▶ Prophet 모델 학습 시작...")

    df_prophet = load_prophet_dataframe(csv_path)
    if df_prophet is None:
        return None, None

    # Prophet 설정
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.3,
    )

    model.add_regressor("is_exam")
    model.add_regressor("is_festival")
    model.add_regressor("is_vacation")

    # 학습
    model.fit(df_prophet)
    print("✔ Prophet 학습 완료")

    # ------------------------------------------------------------
    # 미래 365일 예측 준비
    # ------------------------------------------------------------
    future = model.make_future_dataframe(periods=365)

    future = future.merge(
        df_prophet[["ds", "is_exam", "is_festival", "is_vacation"]],
        on="ds",
        how="left",
    )

    # ===== 미래 회귀변수 설정 함수 =====
    def is_exam_day(dt):
        exams = [(4, 15, 21), (6, 10, 16), (10, 15, 21), (12, 10, 16)]
        for m, s, e in exams:
            if dt.month == m and s <= dt.day <= e:
                return 1
        return 0

    def is_festival_day(dt):
        if dt.month not in [5, 9]:
            return 0

        # 다음 달 1일
        next_month = dt.replace(
            year=dt.year + (1 if dt.month == 12 else 0),
            month=1 if dt.month == 12 else dt.month + 1,
            day=1,
        )
        last_day = next_month - timedelta(days=1)

        d = last_day
        while d.weekday() != 4:  # 금요일
            d -= timedelta(days=1)

        return 1 if dt.date() in [
            (d - timedelta(days=2)).date(),
            (d - timedelta(days=1)).date(),
            d.date()
        ] else 0

    def is_vacation_day(dt):
        m, d = dt.month, dt.day
        if (m == 6 and 1 <= d <= 20) or (m == 12 and 1 <= d <= 20):
            return 0
        if m in [3, 4, 5, 9, 10, 11]:
            return 0
        return 1

    # 미래 날짜에 회귀변수 채우기
    mask_future = future["is_exam"].isna()
    future_dates = future.loc[mask_future, "ds"]

    future.loc[mask_future, "is_exam"] = [is_exam_day(ts) for ts in future_dates]
    future.loc[mask_future, "is_festival"] = [is_festival_day(ts) for ts in future_dates]
    future.loc[mask_future, "is_vacation"] = [is_vacation_day(ts) for ts in future_dates]

    for col in ["is_exam", "is_festival", "is_vacation"]:
        future[col] = future[col].astype(float)

    # 예측
    forecast = model.predict(future)
    print("✔ 365일 예측 완료")


    print("✔ Prophet 모델 저장 완료")

    return model, forecast


# ============================================================
# 3) 저장된 Prophet 모델 로드
# ============================================================
def load_prophet_model():
    if not os.path.exists(PROPHET_MODEL_PATH):
        print("⚠ Prophet 모델 없음 → 학습 필요")
        return None, None

    try:
        with open(PROPHET_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(FORECAST_PATH, "rb") as f:
            forecast = pickle.load(f)
        return model, forecast

    except Exception as e:
        print("❌ Prophet 모델 불러오기 실패:", e)
        return None, None


# ============================================================
# 4) 단독 실행 방지
# ============================================================
if __name__ == "__main__":
    print("이 파일은 직접 실행하지 않습니다.")
    print("Streamlit 또는 final.py에서 호출하세요.")

