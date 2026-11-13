import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)


# 페이지 설정
st.set_page_config(
    page_title="성능 높이기",
    page_icon="📈",
    layout="wide"
)

st.title("📈 성능 높이기")
st.write("가중치를 조정하기 전과 후의 당뇨병 예측 성능을 비교합니다.")


# 데이터 불러오기
DATA_URL = "https://raw.githubusercontent.com/songminha1024/Diabetes-prediction-test/refs/heads/main/diabetes_prediction_dataset.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df = df.drop_duplicates()
    return df


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.write(e)
    st.stop()


# 사용할 속성
feature_names = {
    "age": "나이",
    "bmi": "체질량지수(BMI)",
    "HbA1c_level": "당화혈색소",
    "blood_glucose_level": "혈당",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부"
}

# 속성 선택
st.subheader("1. 예측에 사용할 속성 선택")

selected_features = st.multiselect(
    "사용할 속성을 선택하세요.",
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
    st.warning("예측에 사용할 속성을 하나 이상 선택해 주세요.")
    st.stop()


# 데이터 나누기
X = df[selected_features].copy()
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# BMI 결측치 처리
if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()

    X_train.loc[:, "bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test.loc[:, "bmi"] = X_test["bmi"].fillna(bmi_median)


# 모델 학습
# 가중치를 조정하기 전
logistic_before = LogisticRegression(max_iter=1000)
tree_before = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

# 가중치를 조정한 후
logistic_after = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

tree_after = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced"
)


# 로지스틱 회귀는 표준화 진행
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_before.fit(X_train_scaled, y_train)
logistic_after.fit(X_train_scaled, y_train)

tree_before.fit(X_train, y_train)
tree_after.fit(X_train, y_train)


# 예측 결과
logistic_pred_before = logistic_before.predict(X_test_scaled)
logistic_pred_after = logistic_after.predict(X_test_scaled)

tree_pred_before = tree_before.predict(X_test)
tree_pred_after = tree_after.predict(X_test)


# 성능 평가 함수
def evaluate_model(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "정확도": accuracy_score(y_true, y_pred),
        "재현율": recall_score(y_true, y_pred, zero_division=0),
        "정밀도": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0)
    }


# 평가 결과 계산
logistic_result_before = evaluate_model(y_test, logistic_pred_before)
logistic_result_after = evaluate_model(y_test, logistic_pred_after)

tree_result_before = evaluate_model(y_test, tree_pred_before)
tree_result_after = evaluate_model(y_test, tree_pred_after)


# 결과 비교 함수
def show_comparison(title, before, after):
    st.subheader(title)

    result_df = pd.DataFrame({
        "가중치 조정 전": before,
        "가중치 조정 후": after
    })

    st.dataframe(
        result_df.style.format({
            "가중치 조정 전": "{:.4f}",
            "가중치 조정 후": "{:.4f}"
        }),
        use_container_width=True
    )

    st.markdown("**예측 인원 변화**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "찾아낸 당뇨병 환자",
            f"{after['TP']:,}명",
            delta=f"{after['TP'] - before['TP']:+,}명"
        )

    with col2:
        st.metric(
            "놓친 당뇨병 환자",
            f"{after['FN']:,}명",
            delta=f"{after['FN'] - before['FN']:+,}명",
            delta_color="inverse"
        )

    with col3:
        st.metric(
            "헛짚은 사람",
            f"{after['FP']:,}명",
            delta=f"{after['FP'] - before['FP']:+,}명",
            delta_color="inverse"
        )


# 성능 비교
st.divider()
st.subheader("2. 가중치 조정 전후 성능 비교")

st.caption(
    "TN: 당뇨병이 없는 사람을 맞게 예측한 수 / "
    "FP: 당뇨병이 없는 사람을 당뇨병으로 잘못 예측한 수"
)

st.caption(
    "FN: 당뇨병이 있는 사람을 놓친 수 / "
    "TP: 당뇨병이 있는 사람을 맞게 예측한 수"
)

show_comparison(
    "① 로지스틱 회귀",
    logistic_result_before,
    logistic_result_after
)

st.divider()

show_comparison(
    "② 의사결정트리",
    tree_result_before,
    tree_result_after
)


# 그래프에 사용할 한글 이름
selected_korean = [
    feature_names[feature] for feature in selected_features
]


# 실제 당뇨병 여부에 따른 색상
def get_actual_colors(y_values):
    return [
        "당뇨병 있음" if value == 1 else "당뇨병 없음"
        for value in y_values
    ]


# 2차원 그래프
def draw_2d_graph(model_before, model_after, model_name):

    x_feature = selected_features[0]
    y_feature = selected_features[1]

    x_min = X_test[x_feature].min()
    x_max = X_test[x_feature].max()
    y_min = X_test[y_feature].min()
    y_max = X_test[y_feature].max()

    x_range = np.linspace(x_min, x_max, 100)
    y_range = np.linspace(y_min, y_max, 100)

    xx, yy = np.meshgrid(x_range, y_range)

    grid = pd.DataFrame({
        x_feature: xx.ravel(),
        y_feature: yy.ravel()
    })

    # 예측 확률 계산
    if model_name == "로지스틱 회귀":
        grid_scaled = scaler.transform(grid)

        prob_before = model_before.predict_proba(grid_scaled)[:, 1]
        prob_after = model_after.predict_proba(grid_scaled)[:, 1]

    else:
        prob_before = model_before.predict_proba(grid)[:, 1]
        prob_after = model_after.predict_proba(grid)[:, 1]

    prob_before = prob_before.reshape(xx.shape)
    prob_after = prob_after.reshape(xx.shape)

    fig = go.Figure()

    # 실제 데이터 표시
    actual_colors = get_actual_colors(y_test)

    fig.add_trace(go.Scatter(
        x=X_test[x_feature],
        y=X_test[y_feature],
        mode="markers",
        name="실제 데이터",
        marker=dict(
            size=6,
            color=[
                "red" if value == 1 else "blue"
                for value in y_test
            ],
            opacity=0.5
        ),
        text=actual_colors,
        hovertemplate=(
            "실제 결과: %{text}<br>"
            + selected_korean[0] + ": %{x}<br>"
            + selected_korean[1] + ": %{y}"
            + "<extra></extra>"
        )
    ))

    # 가중치 조정 전 경계선
    fig.add_trace(go.Contour(
        x=x_range,
        y=y_range,
        z=prob_before,
        contours=dict(
            start=0.5,
            end=0.5,
            size=1,
            coloring="none"
        ),
        line=dict(color="green", width=3),
        showscale=False,
        name="가중치 조정 전 0.5 경계선",
        hoverinfo="skip"
    ))

    # 가중치 조정 후 경계선
    fig.add_trace(go.Contour(
        x=x_range,
        y=y_range,
        z=prob_after,
        contours=dict(
            start=0.5,
            end=0.5,
            size=1,
            coloring="none"
        ),
        line=dict(color="orange", width=3, dash="dash"),
        showscale=False,
        name="가중치 조정 후 0.5 경계선",
        hoverinfo="skip"
    ))

    fig.update_layout(
        title=f"{model_name}의 예측 경계선 비교",
        xaxis_title=selected_korean[0],
        yaxis_title=selected_korean[1],
        legend_title="구분",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # 경계선이 나타나지 않는 경우 안내
    before_min = prob_before.min()
    before_max = prob_before.max()
    after_min = prob_after.min()
    after_max = prob_after.max()

    if not (before_min <= 0.5 <= before_max):
        st.info("가중치 조정 전에는 그래프 범위 안에 0.5 경계선이 없습니다.")

    if not (after_min <= 0.5 <= after_max):
        st.info("가중치 조정 후에는 그래프 범위 안에 0.5 경계선이 없습니다.")


# 3차원 그래프
def draw_3d_graph(model_before, model_after, model_name):

    features_3d = selected_features[:3]

    x_feature = features_3d[0]
    y_feature = features_3d[1]
    z_feature = features_3d[2]

    # 그래프에 표시할 실제 데이터
    display_data = X_test.copy()

    if len(display_data) > 2500:
        display_data = display_data.sample(
            n=2500,
            random_state=42
        )

    display_y = y_test.loc[display_data.index]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=display_data[x_feature],
        y=display_data[y_feature],
        z=display_data[z_feature],
        mode="markers",
        name="실제 데이터",
        marker=dict(
            size=3,
            color=[
                "red" if value == 1 else "blue"
                for value in display_y
            ],
            opacity=0.6
        ),
        text=get_actual_colors(display_y),
        hovertemplate=(
            "실제 결과: %{text}<br>"
            + feature_names[x_feature] + ": %{x}<br>"
            + feature_names[y_feature] + ": %{y}<br>"
            + feature_names[z_feature] + ": %{z}"
            + "<extra></extra>"
        )
    ))

    # 3차원 예측 범위 만들기
    x_range = np.linspace(
        X_train[x_feature].min(),
        X_train[x_feature].max(),
        22
    )

    y_range = np.linspace(
        X_train[y_feature].min(),
        X_train[y_feature].max(),
        22
    )

    z_range = np.linspace(
        X_train[z_feature].min(),
        X_train[z_feature].max(),
        22
    )

    xx, yy, zz = np.meshgrid(
        x_range,
        y_range,
        z_range,
        indexing="ij"
    )

    grid = pd.DataFrame(
        np.tile(X_train.median().values, (xx.size, 1)),
        columns=selected_features
    )

    grid.loc[:, x_feature] = xx.ravel()
    grid.loc[:, y_feature] = yy.ravel()
    grid.loc[:, z_feature] = zz.ravel()

    # 나머지 선택 속성은 훈련 데이터의 중앙값으로 유지
    if "bmi" in selected_features:
        grid.loc[:, "bmi"] = grid["bmi"].fillna(
            X_train["bmi"].median()
        )

    # 예측 확률 계산
    if model_name == "로지스틱 회귀":
        grid_scaled = scaler.transform(grid)

        prob_before = model_before.predict_proba(grid_scaled)[:, 1]
        prob_after = model_after.predict_proba(grid_scaled)[:, 1]

    else:
        prob_before = model_before.predict_proba(grid)[:, 1]
        prob_after = model_after.predict_proba(grid)[:, 1]

    prob_before = prob_before.reshape(xx.shape)
    prob_after = prob_after.reshape(xx.shape)

    # 0.5 경계면 표시
    if prob_before.min() <= 0.5 <= prob_before.max():
        fig.add_trace(go.Isosurface(
            x=xx.ravel(),
            y=yy.ravel(),
            z=zz.ravel(),
            value=prob_before.ravel(),
            isomin=0.499,
            isomax=0.501,
            surface_count=1,
            colorscale=[[0, "green"], [1, "green"]],
            opacity=0.35,
            showscale=False,
            caps=dict(x_show=False, y_show=False, z_show=False),
            name="가중치 조정 전 경계면",
            hoverinfo="skip"
        ))
    else:
        st.info("가중치 조정 전에는 그래프 범위 안에 0.5 경계면이 없습니다.")

    if prob_after.min() <= 0.5 <= prob_after.max():
        fig.add_trace(go.Isosurface(
            x=xx.ravel(),
            y=yy.ravel(),
            z=zz.ravel(),
            value=prob_after.ravel(),
            isomin=0.499,
            isomax=0.501,
            surface_count=1,
            colorscale=[[0, "orange"], [1, "orange"]],
            opacity=0.35,
            showscale=False,
            caps=dict(x_show=False, y_show=False, z_show=False),
            name="가중치 조정 후 경계면",
            hoverinfo="skip"
        ))
    else:
        st.info("가중치 조정 후에는 그래프 범위 안에 0.5 경계면이 없습니다.")

    fig.update_layout(
        title=f"{model_name}의 3차원 예측 경계면 비교",
        scene=dict(
            xaxis_title=feature_names[x_feature],
            yaxis_title=feature_names[y_feature],
            zaxis_title=feature_names[z_feature]
        ),
        height=750
    )

    st.plotly_chart(fig, use_container_width=True)


# 그래프 표시
st.divider()
st.subheader("3. 예측 경계선 및 경계면 비교")

if len(selected_features) == 1:
    st.info("경계선을 비교하려면 속성을 두 개 이상 선택해 주세요.")

elif len(selected_features) == 2:
    st.write("선택한 두 속성을 이용해 2차원 예측 경계선을 비교합니다.")

    draw_2d_graph(
        logistic_before,
        logistic_after,
        "로지스틱 회귀"
    )

    draw_2d_graph(
        tree_before,
        tree_after,
        "의사결정트리"
    )

else:
    st.write("선택한 속성 중 앞의 세 개를 이용해 3차원 예측 경계면을 비교합니다.")

    draw_3d_graph(
        logistic_before,
        logistic_after,
        "로지스틱 회귀"
    )

    draw_3d_graph(
        tree_before,
        tree_after,
        "의사결정트리"
    )
