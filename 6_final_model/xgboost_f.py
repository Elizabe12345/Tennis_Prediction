import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import os

# 数据集文件路径
file_paths = [
    f"../dataset/processed_data_1/match_df_{start:02d}_{end:02d}.csv"
    for start, end in zip(range(0, 25, 5), range(4, 29, 5))
]

# 遍历数据集进行训练和评估
for file_path in file_paths:
    print(f"Processing {file_path}...")
    match_df = pd.read_csv(file_path, low_memory=False)
    dataset_name = file_path.split("\\")[-1].split(".")[0]

    # 选择特征
    hist_features = [col for col in match_df.columns if col.endswith('_hist')]
    hist_e_features = [col for col in match_df.columns if col.endswith('_hist_e')]
    supplement_features = ['round_code', 'best_of', 'player1_seed_bucket', 'player2_seed_bucket',
                           'player1_entry', 'player1_host', 'player1_hand', 'player1_ht', 'player1_age',
                           'player2_entry', 'player2_host', 'player2_hand', 'player2_ht', 'player2_age']
    elo_features = ['player1_elo', 'player2_elo']
    rank_features = ['player1_rank', 'player1_rank_points', 'player2_rank', 'player2_rank_points']
    histo_features = [col for col in match_df.columns if col.endswith('_histo')]

    features = hist_features + hist_e_features + supplement_features + elo_features + rank_features + histo_features
    continuous_features = hist_features + hist_e_features + rank_features + histo_features + elo_features + [
        'player1_ht', 'player1_age', 'player2_ht', 'player2_age']

    # 目标变量
    X = match_df[features].copy().fillna(0)
    y = match_df['result']

    # 归一化数值特征
    scaler = StandardScaler()
    X[continuous_features] = scaler.fit_transform(X[continuous_features])

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost 训练
    model = xgb.XGBClassifier(
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
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred)
    feature_importance = model.feature_importances_
    feature_importance_dict = {feature: importance for feature, importance in zip(features, feature_importance)}
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    # 结果保存
    result = {
        "accuracy": accuracy,
        "AUC": auc,
        "classification_report": report,
        "feature_importance": dict(zip(features, feature_importance))
    }
    output_file = os.path.join(r"./xgboost", f"{dataset_name}_results.txt")
    with open(output_file, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nFeature Importance:\n")
        for feature, importance in sorted_features:
            f.write(f"{feature}: {importance:.6f}\n")

    print(f"Results saved to {output_file}")
