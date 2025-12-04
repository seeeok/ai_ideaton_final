import streamlit as st
import pandas as pd
import requests
from datetime import date, timedelta, datetime

from final import (
    predict_total_and_menus,
    compute_menu_mae,
    compute_total_mae,
)

from prophet_model import train_prophet_model
from XGBoost_model import train_xgb_models


CSV_PATH = "ai_ideaton.csv"
WEEKDAY_KR = ["월", "화", "수", "목", "금", "토", "일"]


# --------------------------------------------------------------
# 페이지 설정
# --------------------------------------------------------------
st.set_page_config(page_title="AI Cafeteria", layout="wide")

if "page" not in st.session_state:
    st.session_state["page"] = "main"


# =======================================================
# 기온 API
# =======================================================
def fetch_temperature(dt):
    """오늘 기준 16일까지는 실제 예측 값, 그 이후는 fallback"""
    if isinstance(dt, datetime):
        dt = dt.date()

    LAT, LON = 37.275, 127.132
    today = date.today()
    ds = dt.strftime("%Y-%m-%d")

    # 과거 기온
    if dt < today:
        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={LAT}&longitude={LON}"
            f"&start_date={ds}&end_date={ds}"
            "&daily=temperature_2m_mean"
            "&timezone=Asia%2FSeoul"
        )
        try:
            r = requests.get(url, timeout=5).json()
            return float(r["daily"]["temperature_2m_mean"][0])
        except:
            return 10.0

    # 미래 (최대 16일)
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={LAT}&longitude={LON}"
        "&daily=temperature_2m_min,temperature_2m_max"
        "&forecast_days=16"
        "&timezone=Asia%2FSeoul"
    )

    try:
        r = requests.get(url, timeout=5).json()
        dates = r["daily"]["time"]

        if ds not in dates:
            return 10.0  # fallback

        idx = dates.index(ds)
        tmin = r["daily"]["temperature_2m_min"][idx]
        tmax = r["daily"]["temperature_2m_max"][idx]
        return float((tmin + tmax) / 2)
    except:
        return 10.0


# =======================================================
# 메뉴 카테고리 로드
# =======================================================
@st.cache_data
def load_menus():
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return (
        df,
        sorted(df["korean_category"].unique()),
        sorted(df["chinese_category"].unique()),
        sorted(df["japanese_category"].unique()),
        sorted(df["western_category"].unique()),
    )


# =======================================================
# 📌 선택한 날짜 → 그 주 월~금 반환
# =======================================================
def get_week_dates(ref_date):
    wd = ref_date.weekday()  # 0=월 ~ 6=일
    monday = ref_date - timedelta(days=wd)
    return [monday + timedelta(days=i) for i in range(5)]


# =======================================================
# 날짜 카드 UI
# =======================================================
def day_card(title, k_opts, c_opts, j_opts, w_opts, dt):

    st.markdown(
        f"""
        <div style='background:#F3F3F3;padding:8px;border-radius:10px;
        text-align:center;font-weight:700;border:1px solid #DDD;margin-bottom:8px;'>
            {title}
        </div>
        """,
        unsafe_allow_html=True,
    )

    kor = st.selectbox("한식 메뉴", k_opts, key=f"kor_{title}")
    chi = st.selectbox("중식 메뉴", c_opts, key=f"chi_{title}")
    jap = st.selectbox("일식 메뉴", j_opts, key=f"jap_{title}")
    wes = st.selectbox("양식 메뉴", w_opts, key=f"wes_{title}")

    auto_temp = fetch_temperature(dt)
    temp = st.number_input("기온(℃)", value=float(auto_temp), key=f"temp_{title}")

    return kor, chi, jap, wes, temp


# =======================================================
# 신뢰도 요약
# =======================================================
def readable_error_summary():
    try:
        mae = compute_menu_mae()
        tot = compute_total_mae()

        return f"""
최근 판매 데이터를 기준으로 보면, **평균적으로**

- 한식 약 **{int(mae['korean'])}그릇**
- 중식 약 **{int(mae['chinese'])}그릇**
- 일식 약 **{int(mae['japanese'])}그릇**
- 양식 약 **{int(mae['western'])}그릇**

정도의 예측 오차가 있습니다.

총판매량 기준으로도 평균 약 **{int(tot['total_mae'])}그릇** 정도 차이가 발생합니다.
"""
    except:
        return "최근 데이터가 부족하여 신뢰도 요약을 제공할 수 없습니다."


# =======================================================
# 메인 페이지
# =======================================================
def show_main():
    df_hist, k_opts, c_opts, j_opts, w_opts = load_menus()

    st.title("🍽 AI 식당 판매량 예측")
    st.caption("영양사 · 운영팀을 위한 간단하고 직관적인 예측 도구")

    # ----------------------------------------
    # 📅 주 선택 UI
    # ----------------------------------------
    selected_day = st.date_input("예측할 주 선택 (해당 날짜가 포함된 주가 자동 선택됩니다)", value=date.today())
    dates = get_week_dates(selected_day)

    header = st.columns([8, 2])
    with header[1]:
        if st.button("판매 기록 입력"):
            st.session_state["page"] = "record"
            st.rerun()

    st.markdown("---")

    cols = st.columns(5)
    inputs = {}

    for i, col in enumerate(cols):
        dt = dates[i]
        title = f"{dt.strftime('%m/%d')} ({WEEKDAY_KR[dt.weekday()]})"

        with col:
            kor, chi, jap, wes, temp = day_card(title, k_opts, c_opts, j_opts, w_opts, dt)

            inputs[title] = {
                "date": dt,
                "kor": kor,
                "chi": chi,
                "jap": jap,
                "wes": wes,
                "temp": temp,
            }

    st.markdown("---")

    if st.button("📈 선택한 주 예측하기", type="primary"):
        rows = []

        for title, info in inputs.items():
            res = predict_total_and_menus(
                info["date"],
                info["kor"],
                info["chi"],
                info["jap"],
                info["wes"],
                info["temp"],
            )

            rows.append({
                "날짜(요일)": title,
                "한식": res["korean_sales"],
                "중식": res["chinese_sales"],
                "일식": res["japanese_sales"],
                "양식": res["western_sales"],
                "총판매량": res["total_sales"],
            })

        df = pd.DataFrame(rows)
        st.subheader("📋 예측 결과")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("📘 예측 신뢰도 요약")
        st.markdown(readable_error_summary())


# =======================================================
# 판매 기록 입력 페이지
# =======================================================
def show_record():

    if st.button("← 돌아가기"):
        st.session_state["page"] = "main"
        st.rerun()

    st.title("📥 실제 판매량 기록하기")

    df_hist, k_opts, c_opts, j_opts, w_opts = load_menus()

    rec_date = st.date_input("날짜")
    rec_k = st.selectbox("한식 메뉴", k_opts)
    rec_c = st.selectbox("중식 메뉴", c_opts)
    rec_j = st.selectbox("일식 메뉴", j_opts)
    rec_w = st.selectbox("양식 메뉴", w_opts)

    s_k = st.number_input("한식 판매량", min_value=0)
    s_c = st.number_input("중식 판매량", min_value=0)
    s_j = st.number_input("일식 판매량", min_value=0)
    s_w = st.number_input("양식 판매량", min_value=0)

    temp = st.number_input("기온(℃)", value=10.0)

    if st.button("저장하기", type="primary"):
        from final import is_exam_day, is_festival_day, is_vacation_day

        df = pd.read_csv(CSV_PATH)
        ts = pd.to_datetime(rec_date)
        ds = ts.strftime("%Y-%m-%d")

        # 기존 날짜 삭제 후 새 거래 입력
        df = df[df["date"] != ds]

        new = {
            "date": ds,
            "weekday": ts.weekday(),
            "korean_category": rec_k,
            "chinese_category": rec_c,
            "japanese_category": rec_j,
            "western_category": rec_w,
            "korean_sales": int(s_k),
            "chinese_sales": int(s_c),
            "japanese_sales": int(s_j),
            "western_sales": int(s_w),
            "total_sales": int(s_k + s_c + s_j + s_w),
            "temperature": float(temp),
            "is_exam": int(is_exam_day(ts)),
            "is_festival": int(is_festival_day(ts)),
            "is_vacation": int(is_vacation_day(ts)),
        }

        new = {col: new[col] for col in df.columns}

        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)

        # 모델 재학습
        train_prophet_model()
        train_xgb_models()

        st.success(f"{ds} 데이터 저장 완료!")


# =======================================================
# 라우팅
# =======================================================
if st.session_state["page"] == "main":
    show_main()
else:
    show_record()
