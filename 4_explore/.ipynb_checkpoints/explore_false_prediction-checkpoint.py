import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. 加载数据
match_df = pd.read_csv(r"..\dataset\processed_data_1\match_df_20_24.csv", low_memory=False)

# 2. 特征选择
hist_features = [col for col in match_df.columns if col.endswith('_hist')]
hist_e_features = [col for col in match_df.columns if col.endswith('_hist_e')]
supplement_features = ['round_code', 'best_of', 'player1_seed_bucket', 'player2_seed_bucket',
                       'player1_entry', 'player1_host', 'player1_hand', 'player1_ht', 'player1_age',
                       'player2_entry', 'player2_host', 'player2_hand', 'player2_ht', 'player2_age']
elo_features = ['player1_elo', 'player2_elo']
rank_features = ['player1_rank', 'player1_rank_points', 'player2_rank', 'player2_rank_points']
histo_features = [col for col in match_df.columns if col.endswith('_histo')]

features = hist_features + hist_e_features + supplement_features + elo_features + rank_features + histo_features
continuous_features = hist_features + hist_e_features + rank_features + histo_features + elo_features + ['player1_ht', 'player1_age', 'player2_ht', 'player2_age']

# 3. 目标变量
X = match_df[features].copy().fillna(0)
y = match_df['result']

# 4. 归一化数值特征
scaler = StandardScaler()
X[continuous_features] = scaler.fit_transform(X[continuous_features])

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 定义 LightGBM 训练参数
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': -1,
    'verbose': -1
}

# 7. 训练 LightGBM 模型
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

lgb_model = lgb.train(
    lgb_params, train_data, num_boost_round=200,
    valid_sets=[test_data], early_stopping_rounds=50, verbose_eval=20
)

# 8. 预测 LightGBM 结果
y_pred_lgb = (lgb_model.predict(X_test) > 0.5).astype(int)

# 9. 训练 XGBoost 模型
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic", learning_rate=0.05,
    max_depth=6, n_estimators=200, subsample=0.8, colsample_bytree=0.8, random_state=42
)
xgb_model.fit(X_train, y_train)

# 10. 预测 XGBoost 结果
y_pred_xgb = xgb_model.predict(X_test)

# 11. 评估 LightGBM 和 XGBoost
print(f"LightGBM Accuracy: {accuracy_score(y_test, y_pred_lgb):.4f}")
print(classification_report(y_test, y_pred_lgb))

print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(classification_report(y_test, y_pred_xgb))

# ------------------------------ 误分类分析 ------------------------------ #

# 12. 计算分类错误样本索引
incorrect_idx_lgb = (y_pred_lgb != y_test)
incorrect_idx_xgb = (y_pred_xgb != y_test)

# 13. 误分类样本的统计分析
print("误分类样本类别分布 (LightGBM)：")
print(y_test[incorrect_idx_lgb].value_counts())

print("误分类样本类别分布 (XGBoost)：")
print(y_test[incorrect_idx_xgb].value_counts())

# 14. 误分类样本的特征分析
incorrect_samples_lgb = X_test[incorrect_idx_lgb]
incorrect_samples_xgb = X_test[incorrect_idx_xgb]

# 计算均值差异
feature_diff_lgb = incorrect_samples_lgb.mean() - X_test.mean()
feature_diff_xgb = incorrect_samples_xgb.mean() - X_test.mean()

print("误分类样本和整体样本的特征均值差异 (LightGBM)：")
print(feature_diff_lgb.sort_values(ascending=False).head(10))

print("误分类样本和整体样本的特征均值差异 (XGBoost)：")
print(feature_diff_xgb.sort_values(ascending=False).head(10))

# 15. 误分类样本的预测概率分布
y_prob_lgb = lgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

incorrect_prob_lgb = y_prob_lgb[incorrect_idx_lgb]
incorrect_prob_xgb = y_prob_xgb[incorrect_idx_xgb]

plt.figure(figsize=(12, 5))
plt.hist(incorrect_prob_lgb, bins=20, alpha=0.6, color='r', label='Misclassified (LightGBM)')
plt.hist(incorrect_prob_xgb, bins=20, alpha=0.6, color='b', label='Misclassified (XGBoost)')
plt.axvline(0.5, color='k', linestyle='dashed', linewidth=1)
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.title("Distribution of Misclassified Samples' Probabilities")
plt.legend()
plt.show()

# 16. 误分类比赛 Elo 评分差异分析
X_test['elo_diff'] = X_test['player1_elo'] - X_test['player2_elo']

plt.figure(figsize=(12, 5))
sns.histplot(X_test.loc[incorrect_idx_lgb, 'elo_diff'], color='red', label='Misclassified (LightGBM)', kde=True)
sns.histplot(X_test.loc[incorrect_idx_xgb, 'elo_diff'], color='blue', label='Misclassified (XGBoost)', kde=True)
sns.histplot(X_test['elo_diff'], color='gray', label='All Matches', kde=True, alpha=0.5)
plt.legend()
plt.title("Elo Score Difference Distribution (Misclassified vs Correct)")
plt.show()

# 17. 误分类样本示例
print("随机 5 个误分类的比赛 (LightGBM)：")
print(X_test[incorrect_idx_lgb].sample(5))

print("随机 5 个误分类的比赛 (XGBoost)：")
print(X_test[incorrect_idx_xgb].sample(5))
