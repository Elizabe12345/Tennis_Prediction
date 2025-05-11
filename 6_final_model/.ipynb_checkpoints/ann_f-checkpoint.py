import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.neural_network import MLPClassifier
import shap

# 文件列表
data_files = [
    f"../dataset/processed_data_1/match_df_{start:02d}_{end:02d}.csv"
    for start, end in zip(range(0, 25, 5), range(4, 29, 5))
]

for file in data_files:
    match_df = pd.read_csv(file, low_memory=False)
    dataset_name = file.split("\\")[-1].split(".")[0]  # 获取数据集名称

    # 选择特征（与原代码保持一致）
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
    X = match_df[features].copy()
    y = match_df['result']

    # 归一化数值特征
    scaler = StandardScaler()
    X[continuous_features] = scaler.fit_transform(X[continuous_features])

    # # 按年份划分训练集和测试集
    # match_df['year'] = pd.to_datetime(match_df['tourney_date'], format='%Y-%m-%d').dt.year
    # train_mask = match_df['year'] < match_df['year'].max()
    # test_mask = match_df['year'] == match_df['year'].max()
    #
    # X_train, X_test = X[train_mask], X[test_mask]
    # y_train, y_test = y[train_mask], y[test_mask]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 构建人工神经网络
    # 修改模型配置部分
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        batch_size=32,
        max_iter=200,  # 增加最大迭代次数
        learning_rate_init=0.005,  # 适当增大学习率
        early_stopping=True,  # 启用早停
        n_iter_no_change=10,  # 10次迭代无改进则停止
        tol=1e-4,  # 损失改进阈值
        random_state=42,
        verbose=True
    )

    # 训练模型
    model.fit(X_train, y_train)

    # 评估模型
    y_prob = model.predict_proba(X_test)[:, 1]  # 获取正类概率
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred)

    # 计算特征重要性
    background = X_train.iloc[:100]  # 使用前100个样本作为背景数据
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_test.iloc[:100])

    # 处理二分类SHAP值（取正类）
    if isinstance(shap_values, list):
        shap_values_positive = shap_values[1]
    else:
        shap_values_positive = shap_values

    feature_importance = np.abs(shap_values_positive).mean(axis=0)
    feature_importance_dict = dict(zip(features, feature_importance))
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    # 输出结果到文件
    output_file = os.path.join(r".\ann", f"{dataset_name}_results.txt")
    with open(output_file, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nFeature Importance:\n")
        for feature, importance in sorted_features:
            f.write(f"{feature}: {importance:.4f}\n")

    print(f"Results saved to {output_file}")