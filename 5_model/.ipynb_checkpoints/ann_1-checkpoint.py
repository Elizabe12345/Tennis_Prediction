import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# 设置 pandas 选项
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 读取数据
match_df = pd.read_csv(r"../dataset/processed_data_1/match_df_20_24.csv", low_memory=False).dropna()

# 特征选择
hist_features = [col for col in match_df.columns if col.endswith('_hist')]
hist_e_features = [col for col in match_df.columns if col.endswith('_hist_e')]
histo_features = [col for col in match_df.columns if col.endswith('_histo')]
elo_features = ['player1_elo', 'player2_elo']
rank_features = ['player1_rank', 'player1_rank_points', 'player2_rank', 'player2_rank_points']
supplement_features = ['round_code', 'best_of', 'player1_seed_bucket', 'player2_seed_bucket',
                       'player1_entry', 'player1_host', 'player1_hand', 'player1_ht', 'player1_age',
                       'player2_entry', 'player2_host', 'player2_hand', 'player2_ht', 'player2_age']

features = hist_features + hist_e_features + histo_features + elo_features + rank_features + supplement_features
continuous_features = hist_features + hist_e_features + histo_features + elo_features + rank_features + [
    'player1_ht', 'player1_age', 'player2_ht', 'player2_age']

# elo_features = ['sym_elo']
# rank_features = [ 'sym_rank', 'sym_rank_points']
# supplement_features = ['round_code', 'best_of', 'surface_Clay', 'surface_Grass',
#                        'surface_Hard','sym_seed_bucket',
#                        'sym_entry', 'sym_host', 'sym_hand', 'sym_ht', 'sym_age']
#
# features = hist_features + hist_e_features + histo_features + elo_features + rank_features + supplement_features
# continuous_features = hist_features + hist_e_features + histo_features + elo_features + rank_features + [
#     'sym_ht', 'sym_age']

# 目标变量
X = match_df[features].copy()
y = match_df['result']

# 数值特征归一化
scaler = StandardScaler()
X[continuous_features] = scaler.fit_transform(X[continuous_features])

# 训练集和测试集划分
match_df['year'] = pd.to_datetime(match_df['tourney_date'], format='%Y-%m-%d').dt.year
train_mask = match_df['year'].between(2020, 2023)
test_mask = match_df['year'] == 2024

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

# 人工神经网络模型训练
ann_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                          alpha=0.01, max_iter=500, random_state=42)
ann_model.fit(X_train, y_train)

# 计算输入层权重重要性
input_weights = np.abs(ann_model.coefs_[0]).sum(axis=1)
feature_importance_nn = pd.Series(input_weights, index=features).sort_values(ascending=False)
print("Neural Network Feature Importance (Input Layer Weights):")
print(feature_importance_nn)

# 预测与评估
y_pred_ann = ann_model.predict(X_test)
accuracy_ann = accuracy_score(y_test, y_pred_ann)
print(f'Artificial Neural Network Accuracy: {accuracy_ann:.4f}')
print(classification_report(y_test, y_pred_ann))

# 排列重要性检验
def permutation_importance(model, X, y, metric=accuracy_score, n_repeats=10):
    baseline_score = metric(y, model.predict(X))
    importances = {}
    for col in X.columns:
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col])
            scores.append(metric(y, model.predict(X_permuted)))
        importances[col] = baseline_score - np.mean(scores)
    return pd.Series(importances).sort_values(ascending=False)

perm_importance_nn = permutation_importance(ann_model, X_test, y_test)
print("Neural Network Feature Importance (Permutation Test):")
print(perm_importance_nn)

# 超参数调优
param_grid_ann = {
    'hidden_layer_sizes': [(64,), (64, 32), (128, 64)],
    'alpha': [0.01, 0.001, 0.0001],
    'solver': ['adam', 'sgd']
}

grid_ann = GridSearchCV(MLPClassifier(max_iter=1000, random_state=42), param_grid_ann, cv=3, scoring='accuracy')
grid_ann.fit(X_train, y_train)

print(f'Best ANN Accuracy: {grid_ann.best_score_:.4f}')
