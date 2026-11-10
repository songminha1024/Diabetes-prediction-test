
import streamlit as st
import pandas as pd
import plotly.express as px

# 페이지 설정
st.set_page_config(page_title="데이터 탐색", page_icon="📊", layout="wide")

st.title("📊 당뇨병 데이터 탐색")
st.write("건강 관련 데이터와 당뇨병 여부 사이의 관계를 살펴봅니다.")

# 데이터 불러오기
url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/diabetes_prediction_dataset.csv"
df = pd.read_csv(url, encoding="utf-8")

# 결측치가 있는 BMI 처리
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

# 중복 데이터 제거
df = df.drop_duplicates()

# 당뇨병 여부 이름 변경
df["당뇨병 여부"] = df["diabetes"].map({
    0: "당뇨병 없음",
    1: "당뇨병 있음"
})

# 1. 나이와 혈당 분포
st.header("1. 나이와 혈당의 분포")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={"age": "나이", "count": "인원수"}
    )
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_glucose = px.histogram(
        df,
        x="blood_glucose_level",
        nbins=30,
        title="혈당 수치 분포",
        labels={"blood_glucose_level": "혈당 수치 (mg/dL)", "count": "인원수"}
    )
    st.plotly_chart(fig_glucose, use_container_width=True)

# 2. 당뇨병 유무에 따른 건강 지표 비교
st.header("2. 당뇨병 유무에 따른 건강 지표 비교")

health_columns = {
    "age": "나이",
    "bmi": "BMI",
    "HbA1c_level": "당화혈색소",
    "blood_glucose_level": "혈당"
}

for column, label in health_columns.items():
    fig = px.box(
        df,
        x="당뇨병 여부",
        y=column,
        color="당뇨병 여부",
        title=f"당뇨병 유무에 따른 {label} 비교",
        labels={column: label}
    )
    st.plotly_chart(fig, use_container_width=True)

# 평균값 비교
st.subheader("건강 지표 평균값 비교")

mean_table = df.groupby("당뇨병 여부")[
    list(health_columns.keys())
].mean().rename(columns=health_columns).round(2)

st.dataframe(mean_table, use_container_width=True)

# 3. 고혈압에 따른 당뇨병 비율
st.header("3. 고혈압에 따른 당뇨병 비율")

hypertension_rate = (
    df.groupby("hypertension")["diabetes"].mean() * 100
).reset_index()

hypertension_rate["고혈압 여부"] = hypertension_rate["hypertension"].map({
    0: "고혈압 없음",
    1: "고혈압 있음"
})

fig_hyper = px.bar(
    hypertension_rate,
    x="고혈압 여부",
    y="diabetes",
    title="고혈압 여부에 따른 당뇨병 비율",
    labels={"diabetes": "당뇨병 비율 (%)", "고혈압 여부": "고혈압 여부"},
    text=hypertension_rate["diabetes"].round(1)
)

fig_hyper.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
st.plotly_chart(fig_hyper, use_container_width=True)

# 4. 심장질환에 따른 당뇨병 비율
st.header("4. 심장질환에 따른 당뇨병 비율")

heart_rate = (
    df.groupby("heart_disease")["diabetes"].mean() * 100
).reset_index()

heart_rate["심장질환 여부"] = heart_rate["heart_disease"].map({
    0: "심장질환 없음",
    1: "심장질환 있음"
})

fig_heart = px.bar(
    heart_rate,
    x="심장질환 여부",
    y="diabetes",
    title="심장질환 여부에 따른 당뇨병 비율",
    labels={"diabetes": "당뇨병 비율 (%)", "심장질환 여부": "심장질환 여부"},
    text=heart_rate["diabetes"].round(1)
)

fig_heart.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
st.plotly_chart(fig_heart, use_container_width=True)

# 5. 성별에 따른 당뇨병 비율
st.header("5. 성별에 따른 당뇨병 비율")

gender_rate = (
    df.groupby("gender")["diabetes"].mean() * 100
).reset_index()

fig_gender = px.bar(
    gender_rate,
    x="gender",
    y="diabetes",
    title="성별에 따른 당뇨병 비율",
    labels={"gender": "성별", "diabetes": "당뇨병 비율 (%)"},
    text=gender_rate["diabetes"].round(1)
)

fig_gender.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
st.plotly_chart(fig_gender, use_container_width=True)

# 6. 흡연 이력에 따른 당뇨병 환자 수
st.header("6. 흡연 이력에 따른 당뇨병 환자 수")

smoking_table = (
    df[df["diabetes"] == 1]
    .groupby("smoking_history")
    .size()
    .reset_index(name="당뇨병 환자 수")
)

st.dataframe(smoking_table, use_container_width=True)
