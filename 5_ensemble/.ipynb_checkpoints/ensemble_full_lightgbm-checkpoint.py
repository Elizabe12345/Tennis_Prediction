import pandas as pd
import lightgbm as lgb
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

# 5. 定义 LightGBM 训练参数
lgb_params = {
    'objective': 'binary',  # 二分类
    'metric': 'binary_error',  # 误差率
    'boosting_type': 'gbdt',  # 传统 GBDT
    'num_leaves': 31,  # 叶子节点数
    'learning_rate': 0.05,  # 学习率
    'feature_fraction': 0.8,  # 80% 特征采样
    'bagging_fraction': 0.8,  # 80% 数据采样
    'bagging_freq': 5,  # 每 5 轮进行一次数据采样
    'max_depth': -1,  # 无限制树深度
    'verbose': -1
}

# 6. 转换为 LightGBM 数据格式
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 7. 训练 LightGBM 模型
lgb_model = lgb.train(
    lgb_params, train_data, num_boost_round=200,
    valid_sets=[test_data], early_stopping_rounds=50, verbose_eval=20
)

# 8. 预测
y_pred = (lgb_model.predict(X_test) > 0.5).astype(int)  # 预测概率 > 0.5 视为正类

# 9. 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"LightGBM Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
