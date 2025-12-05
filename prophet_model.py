# ============================================================
# prophet_model.py
# Streamlit Cloud 호환 안정 버전
# - Prophet 예측 정상 작동
# - 저장(Pickle) 제거 → Stan backend 미호출 → 오류 없음
# - load_prophet_model() 정상 제공
# ============================================================

import pandas as pd
from prophet import Prophet
from datetime import timedelta
import os

CSV_PATH = "ai_ideaton.csv"


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
# 2) Prophet 모델 학습 (저장 없음)
# ============================================================
def train_prophet_model(csv_path=CSV_PATH):
    print("▶ Prophet 모델 학습 시작...")

    df_prophet = load_prophet_dataframe(csv_path)
    if df_prophet is None:
        return None, None

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.3,
    )

    model.add_regressor("is_exam")
    model.add_regressor("is_festival")
    model.add_regressor("is_vacation")

    model.fit(df_prophet)
    print("✔ Prophet 학습 완료")

    # ==========================
    # 365일 미래 예측
    # ==========================
    future = model.make_future_dataframe(periods=365)

    future = future.merge(
        df_prophet[["ds", "is_exam", "is_festival", "is_vacation"]],
        on="ds", how="left"
    )

    # ----- 미래 회귀변수: 학사일정 -----
    def is_exam_day(dt):
        exams = [(4, 15, 21), (6, 10, 16), (10, 15, 21), (12, 10, 16)]
        for m, s, e in exams:
            if dt.month == m and s <= dt.day <= e:
                return 1
        return 0

    def is_festival_day(dt):
        if dt.month not in [5, 9]:
            return 0
        next_month = dt.replace(
            year=dt.year + (1 if dt.month == 12 else 0),
            month=1 if dt.month == 12 else dt.month + 1,
            day=1
        )
        last_day = next_month - timedelta(days=1)
        d = last_day
        while d.weekday() != 4:
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

    mask = future["is_exam"].isna()
    fd = future.loc[mask, "ds"]

    future.loc[mask, "is_exam"] = [is_exam_day(ts) for ts in fd]
    future.loc[mask, "is_festival"] = [is_festival_day(ts) for ts in fd]
    future.loc[mask, "is_vacation"] = [is_vacation_day(ts) for ts in fd]

    for col in ["is_exam", "is_festival", "is_vacation"]:
        future[col] = future[col].astype(float)

    forecast = model.predict(future)
    print("✔ 365일 예측 완료")

    return model, forecast


# ============================================================
# 3) 저장된 Prophet 모델 로드 → 항상 None 반환 (저장 제거)
# ============================================================
def load_prophet_model():
    """
    Cloud에서는 모델 저장이 불가능하므로 항상 None 반환.
    final.py에서 자동으로 재훈련됨.
    """
    return None, None


# ============================================================
# 4) 단독 실행 방지
# ============================================================
if __name__ == "__main__":
    print("이 파일은 직접 실행하지 않습니다.")
