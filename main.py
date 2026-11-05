
import streamlit as st
import pandas as pd
import os

# 페이지 설정
st.set_page_config(
    page_title="당뇨병 예측 실습실",
    page_icon="🩺",
    layout="wide"
)

# 데이터 불러오기
st.title("🩺 당뇨병 예측 실습실")
st.subheader("건강 데이터를 활용한 AI 당뇨병 예측 모델 개발")

st.write("""
나이, BMI, 당화혈색소, 혈당 등의 건강 데이터를 살펴보고
당뇨병 여부와 어떤 관계가 있는지 알아보는 프로젝트입니다.
""")

st.divider()

# CSV 파일 불러오기
file = st.file_uploader("당뇨병 데이터 CSV 파일을 업로드하세요.", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
elif os.path.exists("diabetes_prediction_dataset.csv"):
    df = pd.read_csv("diabetes_prediction_dataset.csv")
else:
    st.info("GitHub 저장소에 CSV 파일을 올리거나 위에서 파일을 업로드해 주세요.")
    st.stop()

# 데이터 크기 표시
st.subheader("📊 데이터 기본 정보")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("전체 데이터 수", f"{df.shape[0]:,}개")

with col2:
    st.metric("전체 행 개수", f"{df.shape[0]:,}개")

with col3:
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
        st.write(f"**{column}** : {description}")

# 데이터 전처리
processed_df = df.copy()

# 중복 데이터 제거
processed_df = processed_df.drop_duplicates()

# BMI 결측치를 중앙값으로 대체
if "bmi" in processed_df.columns:
    processed_df["bmi"] = processed_df["bmi"].fillna(
        processed_df["bmi"].median()
    )

# 문자형 데이터를 숫자로 변환
category_columns = ["gender", "smoking_history"]
existing_columns = [
    col for col in category_columns
    if col in processed_df.columns
]

if existing_columns:
    processed_df = pd.get_dummies(
        processed_df,
        columns=existing_columns,
        dtype=int
    )

# 전처리된 데이터 저장
processed_df.to_csv(
    "preprocessed_diabetes.csv",
    index=False,
    encoding="utf-8-sig"
)
