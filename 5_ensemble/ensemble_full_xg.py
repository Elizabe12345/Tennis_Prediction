import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. 加载数据
match_df = pd.read_csv(r"..\dataset\processed_data_1\match_df_20_24.csv", low_memory=False)

hist_features = [col for col in match_df.columns if col.endswith('_hist')]
hist_e_features = [col for col in match_df.columns if col.endswith('_hist_e')]
supplement_features =['round_code','best_of',
                      'player1_seed_bucket','player2_seed_bucket',
                     'player1_entry', 'player1_host', 'player1_hand', 'player1_ht','player1_age',
                      'player2_entry', 'player2_host', 'player2_hand', 'player2_ht', 'player2_age',
                      ]
elo_features =['player1_elo','player2_elo']
rank_features =['player1_rank', 'player1_rank_points','player2_rank', 'player2_rank_points']
histo_features = [col for col in match_df.columns if col.endswith('_histo')]

features =hist_features + hist_e_features + supplement_features + elo_features +  rank_features + histo_features
# 数值型特征
continuous_features = hist_features+hist_e_features+rank_features+histo_features+elo_features+['player1_ht','player1_age','player2_ht', 'player2_age',]
# 目标变量
X = match_df[features]
y = match_df['result']

# 显式创建 X 的副本，防止 SettingWithCopyWarning
X = match_df[features].copy().fillna(0)

# 归一化数值特征
scaler = StandardScaler()
X[continuous_features] = scaler.fit_transform(X[continuous_features])
# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 训练 XGBoost 分类器
xgb_classifier = xgb.XGBClassifier(
    n_estimators=200,  # 设定 200 棵树
    learning_rate=0.05,  # 学习率
    max_depth=6,  # 树的最大深度
    subsample=0.8,  # 采样 80% 数据
    colsample_bytree=0.8,  # 采样 80% 特征
    objective='binary:logistic',  # 二分类任务
    eval_metric='logloss',  # 评估指标
    use_label_encoder=False,
    random_state=42
)
xgb_classifier.fit(X_train, y_train)

# 6. 预测
y_pred = xgb_classifier.predict(X_test)

# 7. 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
