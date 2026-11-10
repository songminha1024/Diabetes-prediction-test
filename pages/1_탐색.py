
import streamlit as st
import pandas as pd
import plotly.express as px

# 페이지 설정
st.set_page_config(
    page_title="당뇨병 데이터 탐색",
    page_icon="📊",
    layout="wide"
)

st.title("📊 당뇨병 데이터 탐색")
st.write("건강 데이터의 분포와 당뇨병 여부 사이의 관계를 분석합니다.")

# 1. GitHub에서 데이터 불러오기
url = "https://raw.githubusercontent.com/songminha1024/Diabetes-prediction-test/refs/heads/main/diabetes_prediction_dataset.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(url, encoding="utf-8")

    # 중복 데이터 제거
    data = data.drop_duplicates()

    # BMI 결측치를 중앙값으로 대체
    data["bmi"] = data["bmi"].fillna(data["bmi"].median())

    return data

try:
    df = load_data()
except Exception as e:
    st.error("CSV 파일을 불러오지 못했습니다. GitHub 주소와 파일을 확인해 주세요.")
    st.exception(e)
    st.stop()

st.success("데이터를 성공적으로 불러왔습니다.")

# 당뇨병 여부를 문자로 변환
df["당뇨병 여부"] = df["diabetes"].map({
    0: "당뇨병 없음",
    1: "당뇨병 있음"
})

# 2. 데이터 기본 정보
st.header("1. 데이터 기본 정보")

col1, col2, col3 = st.columns(3)

col1.metric("전체 데이터 수", f"{len(df):,}개")
col2.metric("전체 변수 수", f"{df.shape[1] - 1}개")
col3.metric("당뇨병 환자 수", f"{df['diabetes'].sum():,}명")

# 3. 나이와 혈당 분포
st.header("2. 나이와 혈당 수치의 분포")

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
        labels={
            "blood_glucose_level": "혈당 수치 (mg/dL)",
            "count": "인원수"
        }
    )
    st.plotly_chart(fig_glucose, use_container_width=True)

# 4. 당뇨병 유무에 따른 건강 지표 비교
st.header("3. 당뇨병 유무에 따른 건강 지표 비교")

health_columns = {
    "age": "나이",
    "bmi": "BMI",
    "HbA1c_level": "당화혈색소",
    "blood_glucose_level": "혈당 수치"
}

for column, label in health_columns.items():
    fig = px.box(
        df,
        x="당뇨병 여부",
        y=column,
        color="당뇨병 여부",
        title=f"당뇨병 유무에 따른 {label} 비교",
        labels={
            column: label,
            "당뇨병 여부": "당뇨병 여부"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

# 5. 건강 지표 평균값 비교
st.subheader("건강 지표 평균값 비교")

mean_table = (
    df.groupby("당뇨병 여부")[list(health_columns.keys())]
    .mean()
    .rename(columns=health_columns)
    .round(2)
)

st.dataframe(mean_table, use_container_width=True)

# 6. 고혈압에 따른 당뇨병 비율
st.header("4. 고혈압에 따른 당뇨병 비율")

hypertension_rate = (
    df.groupby("hypertension")["diabetes"]
    .mean()
    .mul(100)
    .reset_index(name="당뇨병 비율")
)

hypertension_rate["고혈압 여부"] = hypertension_rate["hypertension"].map({
    0: "고혈압 없음",
    1: "고혈압 있음"
})

fig_hyper = px.bar(
    hypertension_rate,
    x="고혈압 여부",
    y="당뇨병 비율",
    title="고혈압 여부에 따른 당뇨병 비율",
    text="당뇨병 비율",
    labels={
        "고혈압 여부": "고혈압 여부",
        "당뇨병 비율": "당뇨병 비율 (%)"
    }
)

fig_hyper.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

st.plotly_chart(fig_hyper, use_container_width=True)

# 7. 심장질환에 따른 당뇨병 비율
st.header("5. 심장질환에 따른 당뇨병 비율")

heart_rate = (
    df.groupby("heart_disease")["diabetes"]
    .mean()
    .mul(100)
    .reset_index(name="당뇨병 비율")
)

heart_rate["심장질환 여부"] = heart_rate["heart_disease"].map({
    0: "심장질환 없음",
    1: "심장질환 있음"
})

fig_heart = px.bar(
    heart_rate,
    x="심장질환 여부",
    y="당뇨병 비율",
    title="심장질환 여부에 따른 당뇨병 비율",
    text="당뇨병 비율",
    labels={
        "심장질환 여부": "심장질환 여부",
        "당뇨병 비율": "당뇨병 비율 (%)"
    }
)

fig_heart.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

st.plotly_chart(fig_heart, use_container_width=True)

# 8. 성별에 따른 당뇨병 비율
st.header("6. 성별에 따른 당뇨병 비율")

gender_rate = (
    df.groupby("gender")["diabetes"]
    .mean()
    .mul(100)
    .reset_index(name="당뇨병 비율")
)

fig_gender = px.bar(
    gender_rate,
    x="gender",
    y="당뇨병 비율",
    title="성별에 따른 당뇨병 비율",
    text="당뇨병 비율",
    labels={
        "gender": "성별",
        "당뇨병 비율": "당뇨병 비율 (%)"
    }
)

fig_gender.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

st.plotly_chart(fig_gender, use_container_width=True)

# 9. 흡연 이력에 따른 당뇨병 환자 수
st.header("7. 흡연 이력에 따른 당뇨병 환자 수")

smoking_table = (
    df[df["diabetes"] == 1]
    .groupby("smoking_history")
    .size()
    .reset_index(name="당뇨병 환자 수")
    .rename(columns={"smoking_history": "흡연 이력"})
)

st.dataframe(smoking_table, use_container_width=True)

st.caption("※ 당뇨병 비율은 각 그룹의 전체 인원 중 당뇨병이 있는 인원의 비율입니다.")
