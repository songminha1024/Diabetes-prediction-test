
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import setuptools
import koreanize_matplotlib


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.utils import resample


# 페이지 설정
st.set_page_config(
    page_title="당뇨병 분류 모델",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 당뇨병 예측 분류 모델")
st.write("여러 분류 모델을 이용해 당뇨병 여부를 예측합니다.")


# 데이터 불러오기
url = "https://raw.githubusercontent.com/songminha1024/Diabetes-prediction-test/refs/heads/main/diabetes_prediction_dataset.csv"

try:
    df = pd.read_csv(url, encoding="utf-8")
    df = df.drop_duplicates()
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

if len(selected_features) < 2:
    st.warning("그래프를 만들려면 항목을 두 개 이상 선택해야 합니다.")
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


# 훈련 데이터의 클래스 불균형 조정
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

balanced_data = pd.concat([majority, minority_upsampled])

X_balanced = balanced_data[selected_features]
y_balanced = balanced_data["diabetes"]


# 로지스틱 회귀
scaler = StandardScaler()

X_balanced_scaled = scaler.fit_transform(X_balanced)
X_train_scaled = scaler.transform(X_train)
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
baseline_model = DummyClassifier(strategy="most_frequent")
baseline_model.fit(X_train, y_train)


# 모델 정확도 비교
models = {
    "기준 모델": baseline_model,
    "로지스틱 회귀": logistic_model,
    "의사결정트리": tree_model
}

train_scores = {}
test_scores = {}

for name, model in models.items():

    if name == "로지스틱 회귀":
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
    else:
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

    train_scores[name] = np.mean(train_pred == y_train)
    test_scores[name] = np.mean(test_pred == y_test)


# 정확도 결과
st.subheader("2. 모델 성능 비교")

score_df = pd.DataFrame({
    "모델": list(models.keys()),
    "훈련 데이터 정확도": [
        train_scores[name] for name in models
    ],
    "테스트 데이터 정확도": [
        test_scores[name] for name in models
    ]
})

score_df["훈련 데이터 정확도"] = (
    score_df["훈련 데이터 정확도"] * 100
).round(2)

score_df["테스트 데이터 정확도"] = (
    score_df["테스트 데이터 정확도"] * 100
).round(2)

st.dataframe(score_df, use_container_width=True)

st.caption(
    "정확도는 전체 예측 중 실제 정답과 일치한 비율입니다. "
    "테스트 데이터는 학습에 사용하지 않은 데이터입니다."
)


# 그래프에 사용할 두 항목 선택
st.subheader("3. 두 항목을 이용한 예측 그래프")

col1, col2 = st.columns(2)

with col1:
    x_feature = st.selectbox(
        "가로축 항목",
        selected_features,
        index=0,
        format_func=lambda x: feature_names[x]
    )

with col2:
    y_options = [x for x in selected_features if x != x_feature]

    y_feature = st.selectbox(
        "세로축 항목",
        y_options,
        index=0,
        format_func=lambda x: feature_names[x]
    )


# 나머지 항목은 테스트 데이터 중앙값으로 고정
base_values = X_test.median()

x_min = X_test[x_feature].min()
x_max = X_test[x_feature].max()
y_min = X_test[y_feature].min()
y_max = X_test[y_feature].max()

x_range = np.linspace(x_min, x_max, 120)
y_range = np.linspace(y_min, y_max, 120)

xx, yy = np.meshgrid(x_range, y_range)

grid = pd.DataFrame(
    np.tile(base_values.values, (xx.size, 1)),
    columns=selected_features
)

grid[x_feature] = xx.ravel()
grid[y_feature] = yy.ravel()


# 실제 데이터 일부 표시
plot_data = X_test.copy()
plot_data["diabetes"] = y_test

if len(plot_data) > 3000:
    plot_data = plot_data.sample(n=3000, random_state=42)


# 로지스틱 회귀 예측 그래프
grid_scaled = scaler.transform(grid)

logistic_probability = logistic_model.predict_proba(
    grid_scaled
)[:, 1].reshape(xx.shape)

fig1, ax1 = plt.subplots(figsize=(8, 6))

ax1.contourf(
    xx,
    yy,
    logistic_probability >= 0.5,
    levels=[-0.5, 0.5, 1.5],
    alpha=0.2
)

ax1.contour(
    xx,
    yy,
    logistic_probability,
    levels=[0.5],
    colors="black",
    linewidths=2
)

ax1.scatter(
    plot_data[x_feature],
    plot_data[y_feature],
    c=plot_data["diabetes"],
    cmap="coolwarm",
    alpha=0.5,
    s=12
)

ax1.set_title("Logistic Regression")
ax1.set_xlabel(feature_names[x_feature])
ax1.set_ylabel(feature_names[y_feature])

st.pyplot(fig1)
plt.close(fig1)


# 의사결정트리 예측 그래프
tree_predictions = tree_model.predict(grid)
tree_predictions = tree_predictions.reshape(xx.shape)

fig2, ax2 = plt.subplots(figsize=(8, 6))

ax2.contourf(
    xx,
    yy,
    tree_predictions,
    levels=[-0.5, 0.5, 1.5],
    alpha=0.2
)

ax2.scatter(
    plot_data[x_feature],
    plot_data[y_feature],
    c=plot_data["diabetes"],
    cmap="coolwarm",
    alpha=0.5,
    s=12
)

ax2.set_title("Decision Tree")
ax2.set_xlabel(feature_names[x_feature])
ax2.set_ylabel(feature_names[y_feature])

st.pyplot(fig2)
plt.close(fig2)

st.caption(
    "그래프에서 선택하지 않은 항목은 테스트 데이터의 중앙값으로 고정했습니다. "
    "두 항목의 변화에 따른 예측 결과만 비교하기 위해서입니다."
)


# 의사결정트리 그림
st.subheader("4. 의사결정트리의 판단 과정")

tree = tree_model.tree_

# 원래 훈련 데이터가 각 마디에 몇 개 도달하는지 확인
decision_paths = tree_model.decision_path(X_train)

node_counts = {}
node_positive = {}

for node_id in range(tree.node_count):
    reached = decision_paths[:, node_id].toarray().ravel().astype(bool)

    node_counts[node_id] = int(reached.sum())
    node_positive[node_id] = int(y_train.iloc[np.where(reached)[0]].sum())


# 트리의 마디 위치 계산
positions = {}
leaf_number = [0]

def set_positions(node_id, depth=0):

    left = tree.children_left[node_id]
    right = tree.children_right[node_id]

    if left == right:
        x = leaf_number[0]
        leaf_number[0] += 1
    else:
        left_x = set_positions(left, depth + 1)
        right_x = set_positions(right, depth + 1)
        x = (left_x + right_x) / 2

    positions[node_id] = (x, -depth)
    return x


set_positions(0)

leaf_count = max(leaf_number[0], 1)

fig3, ax3 = plt.subplots(
    figsize=(max(12, leaf_count * 3), 9)
)


# 가지 그리기
for node_id in range(tree.node_count):

    left = tree.children_left[node_id]
    right = tree.children_right[node_id]

    if left == right:
        continue

    x1, y1 = positions[node_id]
    x2, y2 = positions[left]
    x3, y3 = positions[right]

    ax3.annotate(
        "",
        xy=(x2, y2 + 0.12),
        xytext=(x1, y1 - 0.12),
        arrowprops=dict(arrowstyle="->")
    )

    ax3.annotate(
        "",
        xy=(x3, y3 + 0.12),
        xytext=(x1, y1 - 0.12),
        arrowprops=dict(arrowstyle="->")
    )

    ax3.text(
        (x1 + x2) / 2 - 0.1,
        (y1 + y2) / 2,
        "Yes",
        fontsize=9
    )

    ax3.text(
        (x1 + x3) / 2 + 0.1,
        (y1 + y3) / 2,
        "No",
        fontsize=9
    )


# 마디 그리기
for node_id in range(tree.node_count):

    x, y = positions[node_id]

    left = tree.children_left[node_id]
    right = tree.children_right[node_id]

    count = node_counts[node_id]
    positive = node_positive[node_id]

    rate = positive / count * 100 if count > 0 else 0

    if left == right:
        prediction = int(np.argmax(tree.value[node_id][0]))
        result = "Prediction: Diabetes" if prediction == 1 else "Prediction: No diabetes"
        color = "#f4b6b6" if prediction == 1 else "#b7d8f5"

        label = (
            f"Leaf\n"
            f"{result}\n"
            f"Data: {count}\n"
            f"Diabetes: {positive} ({rate:.1f}%)"
        )

    else:
        feature_index = tree.feature[node_id]
        feature = selected_features[feature_index]
        threshold = tree.threshold[node_id]

        label = (
            f"{feature_names[feature]}\n"
            f"<= {threshold:.2f}?\n"
            f"Data: {count}\n"
            f"Diabetes: {positive} ({rate:.1f}%)"
        )

        color = "#dce9f7"

    ax3.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=color,
            edgecolor="gray"
        )
    )


ax3.set_xlim(-1, leaf_count)
ax3.set_ylim(-tree.max_depth - 0.8, 0.8)
ax3.axis("off")
ax3.set_title("Decision Tree Structure", fontsize=15)

st.pyplot(fig3)
plt.close(fig3)


# 트리 요약
leaf_nodes = [
    node_id
    for node_id in range(tree.node_count)
    if tree.children_left[node_id] == tree.children_right[node_id]
]

no_diabetes_leaves = sum(
    1
    for node_id in leaf_nodes
    if np.argmax(tree.value[node_id][0]) == 0
)

used_features = sorted({
    selected_features[tree.feature[node_id]]
    for node_id in range(tree.node_count)
    if tree.feature[node_id] >= 0
})

st.write(f"**전체 마디 수:** {tree.node_count}개")
st.write(f"**마지막 마디 수:** {len(leaf_nodes)}개")
st.write(f"**비당뇨병으로 예측한 마지막 마디 수:** {no_diabetes_leaves}개")
st.write(
    "**예측에 사용된 항목:** "
    + (", ".join(feature_names[x] for x in used_features) if used_features else "없음")
)

st.caption(
    "마디에 표시된 데이터 수와 당뇨병 환자 수는 "
    "균형 조정 전 원래 훈련 데이터를 기준으로 계산했습니다."
)
