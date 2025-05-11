import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


match_df = pd.read_csv(r"..\dataset\processed_data\match_df_20_24_symmetry.csv",low_memory=False)
y = match_df["result"]  # 目标变量
X = match_df.drop(["result", "tourney_id", "tourney_date","player1_id","player2_id",
                   "sym_ret",'sym_sets', 'sym_games'], axis=1)  # 移除无关特征

# 处理缺失值（示例：用均值填充）
X = X.fillna(X.mean())

# 标准化数据（对L1正则化等线性模型重要）
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==============================
# 逻辑回归特征选择 (需标准化数据)
# ==============================
scaler_lr = StandardScaler()
X_train_lr = scaler_lr.fit_transform(X_train)

# 训练逻辑回归模型
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_lr, y_train)

# 提取系数绝对值作为特征重要性
lr_importances = np.abs(lr_model.coef_[0])

# 创建特征重要性表格
feature_importances_lr = pd.DataFrame({
    "feature": X.columns,
    "importance": lr_importances
}).sort_values("importance", ascending=False)

# 选择重要性高于阈值的特征
threshold_lr = np.percentile(feature_importances_lr["importance"], 80)
selected_features_lr = feature_importances_lr[feature_importances_lr["importance"] > threshold_lr]["feature"].tolist()
print("\nLogistic Regression Selected Features:", selected_features_lr)

# ==============================
# 人工神经网络特征选择 (需标准化数据)
# ==============================
scaler_ann = StandardScaler()
X_train_ann = scaler_ann.fit_transform(X_train)

# 方法1: 基于输入层权重的重要性
# 训练MLP模型
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp_model.fit(X_train_ann, y_train)

# 计算输入层权重绝对值之和
weights_input_layer = mlp_model.coefs_[0]
ann_importances = np.sum(np.abs(weights_input_layer), axis=1)

# 创建特征重要性表格
feature_importances_ann = pd.DataFrame({
    "feature": X.columns,
    "importance": ann_importances
}).sort_values("importance", ascending=False)

# 选择特征
threshold_ann = np.percentile(feature_importances_ann["importance"], 80)
selected_features_ann = feature_importances_ann[feature_importances_ann["importance"] > threshold_ann]["feature"].tolist()
print("\nANN Selected Features (Weight-based):", selected_features_ann)

# 方法2: 基于排列重要性 (更耗时但更准确)
result = permutation_importance(mlp_model, X_train_ann, y_train, n_repeats=10, random_state=42)
ann_importances_pi = result.importances_mean

feature_importances_ann_pi = pd.DataFrame({
    "feature": X.columns,
    "importance": ann_importances_pi
}).sort_values("importance", ascending=False)

threshold_ann_pi = np.percentile(feature_importances_ann_pi["importance"], 80)
selected_features_ann_pi = feature_importances_ann_pi[feature_importances_ann_pi["importance"] > threshold_ann_pi]["feature"].tolist()
print("\nANN Selected Features (Permutation Importance):", selected_features_ann_pi)

# ==============================
# 决策树特征选择 (无需标准化)
# ==============================
# 训练决策树模型
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# 提取特征重要性
dt_importances = dt_model.feature_importances_

# 创建特征重要性表格
feature_importances_dt = pd.DataFrame({
    "feature": X.columns,
    "importance": dt_importances
}).sort_values("importance", ascending=False)

# 选择特征
threshold_dt = np.percentile(feature_importances_dt["importance"], 80)
selected_features_dt = feature_importances_dt[feature_importances_dt["importance"] > threshold_dt]["feature"].tolist()
print("\nDecision Tree Selected Features:", selected_features_dt)