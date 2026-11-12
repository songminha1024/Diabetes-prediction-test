
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# -----------------------------
# 1. 페이지 설정
# -----------------------------

st.set_page_config(
    page_title="성능 높이기",
    page_icon="📈",
    layout="wide"
)

st.title("📈 성능 높이기")
st.write("가중치를 맞추기 전과 후의 뇌졸중 예측 결과를 비교합니다.")


# -----------------------------
# 2. 데이터 불러오기
# -----------------------------

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

try:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()

df = df.drop_duplicates()

# 사용할 속성의 우리말 이름
feature_names = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부",
    "bmi": "체질량지수(BMI)"
}


# -----------------------------
# 3. 속성 선택
# -----------------------------

st.subheader("1. 예측에 사용할 속성 선택")

selected_features = st.multiselect(
    "사용할 속성을 선택하세요.",
    options=list(feature_names.keys()),
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease"
    ],
    format_func=lambda x: feature_names[x]
)

if len(selected_features) == 0:
    st.warning("속성을 하나 이상 선택해 주세요.")
    st.stop()


# -----------------------------
# 4. 훈련용과 테스트용 데이터 나누기
# -----------------------------

X = df[selected_features].copy()
y = df["stroke"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# BMI가 선택되었을 때 훈련용 중앙값으로 결측치 처리
if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

# 다른 선택 속성에 결측치가 있을 경우에도 훈련용 중앙값 사용
medians = X_train.median(numeric_only=True)
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)


# -----------------------------
# 5. 데이터 표준화
# -----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------
# 6. 모델 만들기
# -----------------------------

# 가중치를 맞추기 전
logistic_before = LogisticRegression(
    max_iter=1000,
    random_state=42
)

tree_before = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

# 가중치를 맞춘 뒤
logistic_after = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

tree_after = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced"
)

# 모델 학습
logistic_before.fit(X_train_scaled, y_train)
logistic_after.fit(X_train_scaled, y_train)

tree_before.fit(X_train, y_train)
tree_after.fit(X_train, y_train)


# -----------------------------
# 7. 테스트 데이터 예측
# -----------------------------

predictions = {
    "로지스틱 회귀(확률로 답하는 모델)": {
        "전": logistic_before.predict(X_test_scaled),
        "후": logistic_after.predict(X_test_scaled)
    },
    "의사결정트리(질문으로 답하는 모델)": {
        "전": tree_before.predict(X_test),
        "후": tree_after.predict(X_test)
    }
}

models = {
    "로지스틱 회귀(확률로 답하는 모델)": {
        "전": logistic_before,
        "후": logistic_after
    },
    "의사결정트리(질문으로 답하는 모델)": {
        "전": tree_before,
        "후": tree_after
    }
}


# -----------------------------
# 8. 혼동행렬과 성능 지표 계산
# -----------------------------

def calculate_result(actual, predicted):

    actual = np.array(actual)
    predicted = np.array(predicted)

    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tp = int(((actual == 1) & (predicted == 1)).sum())

    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None

    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1
    }


results = {}

for model_name in predictions:
    results[model_name] = {}

    for stage in ["전", "후"]:
        results[model_name][stage] = calculate_result(
            y_test,
            predictions[model_name][stage]
        )


# -----------------------------
# 9. 테스트 데이터 정보
# -----------------------------

st.divider()
st.subheader("2. 가중치를 맞추기 전과 후의 결과")

st.write(
    f"테스트용 데이터: **{len(y_test):,}명** "
    f"/ 실제 뇌졸중 환자: **{int(y_test.sum()):,}명**"
)


# -----------------------------
# 10. 성능 비교 표
# -----------------------------

def format_metric(value):
    if value is None:
        return "계산할 수 없음"
    return f"{value:.3f}"


for model_name in results:

    st.markdown(f"### {model_name}")

    before = results[model_name]["전"]
    after = results[model_name]["후"]

    comparison = pd.DataFrame({
        "항목": [
            "TP (찾아낸 환자)",
            "FN (놓친 환자)",
            "FP (헛짚은 사람)",
            "TN (정상으로 맞힌 사람)",
            "정확도",
            "재현율",
            "정밀도",
            "F1"
        ],
        "가중치 맞추기 전": [
            before["TP"],
            before["FN"],
            before["FP"],
            before["TN"],
            format_metric(before["정확도"]),
            format_metric(before["재현율"]),
            format_metric(before["정밀도"]),
            format_metric(before["F1"])
        ],
        "가중치 맞춘 뒤": [
            after["TP"],
            after["FN"],
            after["FP"],
            after["TN"],
            format_metric(after["정확도"]),
            format_metric(after["재현율"]),
            format_metric(after["정밀도"]),
            format_metric(after["F1"])
        ]
    })

    st.dataframe(comparison, hide_index=True, use_container_width=True)

    st.write(
        f"**찾아낸 환자 수:** {before['TP']}명 → {after['TP']}명  \n"
        f"**놓친 환자 수:** {before['FN']}명 → {after['FN']}명  \n"
        f"**헛짚은 사람 수:** {before['FP']}명 → {after['FP']}명"
    )


# -----------------------------
# 11. 그래프에 사용할 데이터
# -----------------------------

st.divider()
st.subheader("3. 예측 경계 확인")

if len(selected_features) == 1:
    st.info("산점도와 예측 경계를 확인하려면 속성을 두 개 이상 선택해 주세요.")

else:

    # 실제 데이터가 너무 많으면 그래프에는 최대 2000개만 표시
    if len(X_test) > 2000:
        plot_data = X_test.sample(n=2000, random_state=42)
        plot_y = y_test.loc[plot_data.index]
    else:
        plot_data = X_test.copy()
        plot_y = y_test.loc[plot_data.index]

    # 선택한 속성이 2개일 때
    if len(selected_features) == 2:

        x_feature = selected_features[0]
        y_feature = selected_features[1]

        x_label = feature_names[x_feature]
        y_label = feature_names[y_feature]

        x_min = float(X_train[x_feature].min())
        x_max = float(X_train[x_feature].max())
        y_min = float(X_train[y_feature].min())
        y_max = float(X_train[y_feature].max())

        x_values = np.linspace(x_min, x_max, 100)
        y_values = np.linspace(y_min, y_max, 100)

        xx, yy = np.meshgrid(x_values, y_values)

        grid = pd.DataFrame(
            np.tile(X_train.median().values, (xx.size, 1)),
            columns=selected_features
        )

        grid[x_feature] = xx.ravel()
        grid[y_feature] = yy.ravel()

        fig = go.Figure()

        # 실제 뇌졸중 여부에 따른 점
        for label, color, name in [
            (0, "blue", "뇌졸중 없음"),
            (1, "red", "뇌졸중 있음")
        ]:
            mask = plot_y == label

            fig.add_trace(go.Scatter(
                x=plot_data.loc[mask, x_feature],
                y=plot_data.loc[mask, y_feature],
                mode="markers",
                name=name,
                marker=dict(color=color, size=6, opacity=0.6)
            ))

        # 두 모델의 전후 경계선
        boundary_info = []

        for model_name in models:

            for stage, color in [
                ("전", "green"),
                ("후", "orange")
            ]:

                model = models[model_name][stage]

                if "로지스틱 회귀" in model_name:
                    grid_scaled = scaler.transform(grid)
                    probabilities = model.predict_proba(grid_scaled)[:, 1]
                else:
                    probabilities = model.predict_proba(grid)[:, 1]

                zz = probabilities.reshape(xx.shape)

                # 0.5 경계선이 그래프 범위 안에 있는지 확인
                if zz.min() <= 0.5 <= zz.max():

                    fig.add_trace(go.Contour(
                        x=x_values,
                        y=y_values,
                        z=zz,
                        contours=dict(
                            start=0.5,
                            end=0.5,
                            size=1,
                            coloring="none"
                        ),
                        line=dict(color=color, width=3),
                        showscale=False,
                        name=f"{model_name} {stage}"
                    ))

                    boundary_info.append(
                        f"{model_name} {stage}: 경계선 표시"
                    )

                else:
                    boundary_info.append(
                        f"{model_name} {stage}: 0.5 경계선이 그래프 범위 밖에 있음"
                    )

        fig.update_layout(
            title="실제 뇌졸중 여부와 예측 경계선",
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        for info in boundary_info:
            st.caption(info)

        st.caption("초록색은 가중치를 맞추기 전, 주황색은 가중치를 맞춘 뒤의 경계선입니다.")


    # -----------------------------
    # 12. 선택한 속성이 3개 이상일 때
    # -----------------------------

    else:

        st.write("선택한 속성 중 앞의 세 가지를 그래프의 축으로 사용합니다.")

        x_feature = selected_features[0]
        y_feature = selected_features[1]
        z_feature = selected_features[2]

        x_label = feature_names[x_feature]
        y_label = feature_names[y_feature]
        z_label = feature_names[z_feature]

        x_min, x_max = X_train[x_feature].min(), X_train[x_feature].max()
        y_min, y_max = X_train[y_feature].min(), X_train[y_feature].max()
        z_min, z_max = X_train[z_feature].min(), X_train[z_feature].max()

        x_values = np.linspace(x_min, x_max, 20)
        y_values = np.linspace(y_min, y_max, 20)
        z_values = np.linspace(z_min, z_max, 20)

        xx, yy, zz = np.meshgrid(
            x_values,
            y_values,
            z_values,
            indexing="ij"
        )

        grid = pd.DataFrame(
            np.tile(X_train.median().values, (xx.size, 1)),
            columns=selected_features
        )

        grid[x_feature] = xx.ravel()
        grid[y_feature] = yy.ravel()
        grid[z_feature] = zz.ravel()

        fig = go.Figure()

        # 실제 뇌졸중 여부에 따른 점
        for label, color, name in [
            (0, "blue", "뇌졸중 없음"),
            (1, "red", "뇌졸중 있음")
        ]:
            mask = plot_y == label

            fig.add_trace(go.Scatter3d(
                x=plot_data.loc[mask, x_feature],
                y=plot_data.loc[mask, y_feature],
                z=plot_data.loc[mask, z_feature],
                mode="markers",
                name=name,
                marker=dict(
                    color=color,
                    size=3,
                    opacity=0.6
                )
            ))

        # 모델별 전후 0.5 경계면
        for model_name in models:

            for stage, color in [
                ("전", "green"),
                ("후", "orange")
            ]:

                model = models[model_name][stage]

                if "로지스틱 회귀" in model_name:
                    grid_scaled = scaler.transform(grid)
                    probabilities = model.predict_proba(grid_scaled)[:, 1]
                else:
                    probabilities = model.predict_proba(grid)[:, 1]

                probability_grid = probabilities.reshape(xx.shape)

                if probability_grid.min() <= 0.5 <= probability_grid.max():

                    fig.add_trace(go.Isosurface(
                        x=xx.ravel(),
                        y=yy.ravel(),
                        z=zz.ravel(),
                        value=probabilities,
                        isomin=0.5,
                        isomax=0.5,
                        surface_count=1,
                        opacity=0.25,
                        caps=dict(
                            x_show=False,
                            y_show=False,
                            z_show=False
                        ),
                        colorscale=[
                            [0, color],
                            [1, color]
                        ],
                        showscale=False,
                        name=f"{model_name} {stage}"
                    ))

                else:
                    st.caption(
                        f"{model_name} {stage}: "
                        "0.5 경계면이 그래프 범위 안에 나타나지 않습니다."
                    )

        fig.update_layout(
            title="3차원 실제 데이터와 예측 경계면",
            scene=dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title=z_label
            ),
            height=750
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption("초록색은 가중치를 맞추기 전, 주황색은 가중치를 맞춘 뒤의 경계면입니다.")
