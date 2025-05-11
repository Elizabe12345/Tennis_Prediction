import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import itertools
import os

# 文件列表
data_files = [
    f"../dataset/processed_data_1/match_df_{start:02d}_{end:02d}.csv"
    for start, end in zip(range(0, 25, 5), range(4, 29, 5))
]

for file in data_files:
    match_df = pd.read_csv(file, low_memory=False)
    dataset_name = os.path.splitext(os.path.basename(file))[0] # 获取数据集名称

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
    continuous_features = hist_features + hist_e_features + rank_features + histo_features + elo_features + ['player1_ht', 'player1_age', 'player2_ht', 'player2_age']

    # 目标变量
    X = match_df[features].copy().fillna(0)
    y = match_df['result']

    # 归一化数值特征
    scaler = StandardScaler()
    X[continuous_features] = scaler.fit_transform(X[continuous_features])

    # 按年份划分训练集和测试集
    match_df['year'] = pd.to_datetime(match_df['tourney_date'], format='%Y-%m-%d').dt.year
    train_mask = match_df['year'] < match_df['year'].max()
    test_mask = match_df['year'] == match_df['year'].max()

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # 定义参数范围，寻找最佳 n_estimators
    n_estimators_values = [10, 50, 100, 200, 300]
    best_n = None
    best_accuracy = 0

    for n in n_estimators_values:
        model = AdaBoostClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_n = n

    # 使用最佳 n_estimators 训练最终模型
    final_model = AdaBoostClassifier(n_estimators=best_n, random_state=42)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred)

    # 输出结果到文件
    output_file = f"{dataset_name}_adaboost_results.txt"
    with open(output_file, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Best n_estimators value: {best_n}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"Results saved to {output_file}")
