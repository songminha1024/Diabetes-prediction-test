
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.utils import resample


# 페이지 설정
st.set_page_config(
    page_title="당뇨병 예측 채점표",
    page_icon="📊",
    layout="wide"
)

st.title("📊 당뇨병 예측 채점표")
st.write("모델의 예측 결과를 실제 정답과 비교해 성능을 평가합니다.")


# 데이터 불러오기
url = "https://raw.githubusercontent.com/songminha1024/Diabetes-prediction-test/refs/heads/main/diabetes_prediction_dataset.csv"

try:
    df = pd.read_csv(url, encoding="utf-8")
    df = df.drop_duplicates().reset_index(drop=True)

except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


# 사용할 항목 선택
feature_names = {
    "age": "나이",
    "bmi": "BMI",
    "HbA1c_level": "당화혈색소",
    "blood_glucose_level": "혈당",
    "hypertension": "고혈압",
    "heart_disease": "심장질환"
}

st.subheader("1. 예측에 사용할 항목")

selected_features = st.multiselect(
    "예측에 사용할 항목을 선택하세요.",
    options=list(feature_names.keys()),
    default=[
        "age",
        "HbA1c_level",
        "blood_glucose_level",
        "hypertension",
        "heart_disease"
    ],
    format_func=lambda x: feature_names[x]
)

if len(selected_features) == 0:
    st.warning("예측에 사용할 항목을 하나 이상 선택해 주세요.")
    st.stop()


# 데이터 분리
X = df[selected_features].copy()
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# 결측값 처리
medians = X_train.median()

X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)


# 훈련 데이터의 불균형 조정
train_data = X_train.copy()
train_data["diabetes"] = y_train.values

majority = train_data[train_data["diabetes"] == 0]
minority = train_data[train_data["diabetes"] == 1]

minority_upsampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

balanced_data = pd.concat(
    [majority, minority_upsampled]
)

X_balanced = balanced_data[selected_features]
y_balanced = balanced_data["diabetes"]


# 로지스틱 회귀
scaler = StandardScaler()

X_balanced_scaled = scaler.fit_transform(X_balanced)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_balanced_scaled, y_balanced)


# 의사결정트리
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(X_balanced, y_balanced)


# 기준 모델
baseline_model = DummyClassifier(
    strategy="most_frequent"
)

baseline_model.fit(X_train, y_train)


# 테스트 데이터 예측
logistic_pred = logistic_model.predict(X_test_scaled)
tree_pred = tree_model.predict(X_test)
baseline_pred = baseline_model.predict(X_test)


# 혼동행렬 계산
def get_confusion_values(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)

    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tp = int(((actual == 1) & (predicted == 1)).sum())

    return tn, fp, fn, tp


# 평가 지표 계산
def calculate_metrics(tn, fp, fn, tp):
    total = tn + fp + fn + tp

    accuracy = (tn + tp) / total if total > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None

    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return accuracy, recall, precision, f1


def format_metric(value):
    if value is None:
        return "계산할 수 없음"
    return f"{value * 100:.2f}%"


# 각 모델의 결과
model_results = {
    "로지스틱 회귀(확률로 답하는 모델)": logistic_pred,
    "의사결정트리(질문으로 답하는 모델)": tree_pred,
    "훈련용에서 많은 쪽으로만 답하는 모델": baseline_pred
}

confusion_results = {}

for name, predictions in model_results.items():
    tn, fp, fn, tp = get_confusion_values(
        y_test,
        predictions
    )

    confusion_results[name] = {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


# 테스트 데이터 정보
st.subheader("2. 테스트 데이터")

test_total = len(y_test)
test_positive = int((y_test == 1).sum())

st.write(
    f"테스트용 전체 인원: **{test_total:,}명** "
    f"| 당뇨병 환자: **{test_positive:,}명**"
)


# 혼동행렬
st.subheader("3. 모델별 혼동행렬")

st.caption(
    "행은 실제 결과, 열은 모델의 예측 결과입니다. "
    "당뇨병을 양성(1)으로 설정했습니다."
)

confusion_names = [
    "로지스틱 회귀(확률로 답하는 모델)",
    "의사결정트리(질문으로 답하는 모델)"
]

for name in confusion_names:
    result = confusion_results[name]

    tn = result["TN"]
    fp = result["FP"]
    fn = result["FN"]
    tp = result["TP"]

    matrix = pd.DataFrame(
        [
            [
                f"TN\n{tn:,}명",
                f"FP\n{fp:,}명",
                f"{tn + fp:,}명"
            ],
            [
                f"FN\n{fn:,}명",
                f"TP\n{tp:,}명",
                f"{fn + tp:,}명"
            ],
            [
                f"{tn + fn:,}명",
                f"{fp + tp:,}명",
                f"{tn + fp + fn + tp:,}명"
            ]
        ],
        index=["실제 비당뇨병", "실제 당뇨병", "열 합계"],
        columns=["비당뇨병 예측", "당뇨병 예측", "행 합계"]
    )

    st.markdown(f"**{name}**")
    st.table(matrix)


# 평가 지표 비교
st.subheader("4. 모델 성능 비교")

metric_rows = []

for name, result in confusion_results.items():
    tn = result["TN"]
    fp = result["FP"]
    fn = result["FN"]
    tp = result["TP"]

    accuracy, recall, precision, f1 = calculate_metrics(
        tn, fp, fn, tp
    )

    metric_rows.append({
        "모델": name,
        "정확도": format_metric(accuracy),
        "재현율": format_metric(recall),
        "정밀도": format_metric(precision),
        "F1": format_metric(f1)
    })

metric_df = pd.DataFrame(metric_rows)

st.dataframe(
    metric_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "정확도: 전체 예측 중 정답 비율 | "
    "재현율: 실제 당뇨병 환자 중 찾아낸 비율 | "
    "정밀도: 당뇨병이라고 예측한 사람 중 실제 환자 비율 | "
    "F1: 정밀도와 재현율의 조화평균"
)


# Plotly 그래프
st.subheader("5. 모델 성능 그래프")

chart_data = []

for name, result in confusion_results.items():
    accuracy, recall, precision, f1 = calculate_metrics(
        result["TN"],
        result["FP"],
        result["FN"],
        result["TP"]
    )

    chart_data.append({
        "모델": name,
        "정확도": accuracy * 100 if accuracy is not None else None,
        "재현율": recall * 100 if recall is not None else None,
        "정밀도": precision * 100 if precision is not None else None,
        "F1": f1 * 100 if f1 is not None else None
    })

chart_df = pd.DataFrame(chart_data)

chart_long = chart_df.melt(
    id_vars="모델",
    var_name="평가 지표",
    value_name="점수"
)

fig = px.bar(
    chart_long,
    x="평가 지표",
    y="점수",
    color="모델",
    barmode="group",
    text_auto=".1f",
    title="모델별 평가 지표 비교",
    labels={
        "점수": "점수(%)",
        "평가 지표": "평가 지표",
        "모델": "모델"
    }
)

fig.update_layout(
    yaxis_range=[0, 100],
    legend_title_text="모델"
)

st.plotly_chart(fig, use_container_width=True)


# 놓친 환자와 헛짚은 사람
st.subheader("6. 잘못 예측한 사람 확인")

selected_model = st.selectbox(
    "확인할 모델을 선택하세요.",
    options=confusion_names
)

selected_predictions = model_results[selected_model]

# 테스트 데이터의 원래 행 정보 가져오기
detail_columns = [
    "age",
    "blood_glucose_level",
    "HbA1c_level",
    "bmi",
    "hypertension",
    "heart_disease"
]

detail_data = df.loc[X_test.index, detail_columns].copy()
detail_data["실제 결과"] = y_test
detail_data["예측 결과"] = selected_predictions

# FN: 실제 당뇨병인데 비당뇨병으로 예측
missed = detail_data[
    (detail_data["실제 결과"] == 1) &
    (detail_data["예측 결과"] == 0)
].copy()

# FP: 실제 비당뇨병인데 당뇨병으로 예측
false_alarm = detail_data[
    (detail_data["실제 결과"] == 0) &
    (detail_data["예측 결과"] == 1)
].copy()

missed = missed.sort_values(
    by="age",
    ascending=False
)

false_alarm = false_alarm.sort_values(
    by="age",
    ascending=False
)


# 목록에 보여 줄 열 이름
display_names = {
    "age": "나이",
    "blood_glucose_level": "평균 혈당",
    "HbA1c_level": "당화혈색소",
    "bmi": "BMI",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부"
}


def prepare_display(data):
    result = data[detail_columns].head(20).copy()

    result.insert(
        0,
        "번호",
        range(1, len(result) + 1)
    )

    result = result.rename(columns=display_names)

    result["고혈압 여부"] = result["고혈압 여부"].map({
        0: "아니요",
        1: "예"
    })

    result["심장병 여부"] = result["심장병 여부"].map({
        0: "아니요",
        1: "예"
    })

    return result


left, right = st.columns(2)

with left:
    st.markdown(f"**놓친 환자: {len(missed):,}명**")
    st.caption("실제 당뇨병 환자이지만 비당뇨병으로 예측한 사람")

    if len(missed) > 0:
        st.dataframe(
            prepare_display(missed),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("놓친 환자가 없습니다.")

with right:
    st.markdown(f"**헛짚은 사람: {len(false_alarm):,}명**")
    st.caption("실제 비당뇨병이지만 당뇨병으로 예측한 사람")

    if len(false_alarm) > 0:
        st.dataframe(
            prepare_display(false_alarm),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("잘못 당뇨병으로 예측한 사람이 없습니다.")


# 목록과 혼동행렬 비교
selected_result = confusion_results[selected_model]

st.write(
    f"목록의 놓친 환자 수 **{len(missed):,}명** = "
    f"혼동행렬 FN **{selected_result['FN']:,}명**"
)

st.write(
    f"목록의 헛짚은 사람 수 **{len(false_alarm):,}명** = "
    f"혼동행렬 FP **{selected_result['FP']:,}명**"
)
