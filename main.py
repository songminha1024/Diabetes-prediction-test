
import streamlit as st
import pandas as pd

# 페이지 설정
st.set_page_config(
    page_title="당뇨병 예측 실습실",
    page_icon="🩺",
    layout="wide"
)

# 제목 및 프로젝트 소개
st.title("🩺 당뇨병 예측 실습실")
st.subheader("건강 데이터를 활용한 AI 당뇨병 예측 모델 개발")

st.write(
    "나이, BMI, 당화혈색소, 혈당 등의 건강 데이터를 활용하여 "
    "당뇨병 여부를 알아보는 프로젝트입니다."
)

st.divider()

# GitHub에 저장된 CSV 데이터 불러오기
df = pd.read_csv("diabetes_prediction_dataset.csv")

# 데이터 기본 정보
st.subheader("📊 데이터 기본 정보")

col1, col2 = st.columns(2)

with col1:
    st.metric("전체 행 개수", f"{df.shape[0]:,}개")

with col2:
    st.metric("전체 열 개수", f"{df.shape[1]}개")

# 데이터 미리보기
st.divider()
st.subheader("📋 데이터 미리보기")
st.dataframe(df.head(), use_container_width=True)

# 컬럼 설명
st.divider()
st.subheader("📖 데이터 컬럼 설명")

column_info = {
    "gender": "대상자의 성별",
    "age": "대상자의 나이",
    "hypertension": "고혈압 여부 (0: 없음, 1: 있음)",
    "heart_disease": "심장질환 여부 (0: 없음, 1: 있음)",
    "smoking_history": "대상자의 흡연 이력",
    "bmi": "체질량지수",
    "HbA1c_level": "당화혈색소 수치 (%)",
    "blood_glucose_level": "혈당 수치 (mg/dL)",
    "diabetes": "당뇨병 여부 (0: 없음, 1: 있음)"
}

for column, description in column_info.items():
    if column in df.columns:
        st.write(f"**{column}**: {description}")

# 데이터 전처리
processed_df = df.copy()

# 중복 데이터 제거
processed_df = processed_df.drop_duplicates()

# BMI 결측치를 중앙값으로 대체
processed_df["bmi"] = processed_df["bmi"].fillna(
    processed_df["bmi"].median()
)

# 문자형 데이터를 숫자로 변환
processed_df = pd.get_dummies(
    processed_df,
    columns=["gender", "smoking_history"],
    dtype=int
)

# 전처리된 데이터 저장
processed_df.to_csv(
    "preprocessed_diabetes.csv",
    index=False,
    encoding="utf-8-sig"
)
