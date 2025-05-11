import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 设置 pandas 选项
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 读取数据
match_df = pd.read_csv(r"..\dataset\processed_data_1\match_df_20_24_symmetry.csv", low_memory=False).dropna()

# 特征选择
hist_features = [col for col in match_df.columns if col.endswith('_hist')]
hist_e_features = [col for col in match_df.columns if col.endswith('_hist_e')]
histo_features = [col for col in match_df.columns if col.endswith('_histo')]
elo_features = ['sym_elo']
rank_features = [ 'sym_rank', 'sym_rank_points']
supplement_features = ['round_code', 'best_of', 'surface_Clay', 'surface_Grass',
                       'surface_Hard','sym_seed_bucket',
                       'sym_entry', 'sym_host', 'sym_hand', 'sym_ht', 'sym_age']

features = hist_features + hist_e_features + histo_features + elo_features + rank_features + supplement_features
continuous_features = hist_features + hist_e_features + histo_features + elo_features + rank_features + [
    'sym_ht', 'sym_age']

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

# 基学习器
lr_model = LogisticRegression(max_iter=500, random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)
mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                          alpha=0.01, max_iter=500, random_state=42)

# Stacking 集成模型
estimators = [('lr', lr_model), ('dt', dt_model), ('mlp', mlp_model)]
stk_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# 训练集成模型
stk_model.fit(X_train, y_train)

# 预测与评估
y_pred_stk = stk_model.predict(X_test)
accuracy_stk = accuracy_score(y_test, y_pred_stk)
print(f'Stacking Model Accuracy: {accuracy_stk:.4f}')
print(classification_report(y_test, y_pred_stk))
