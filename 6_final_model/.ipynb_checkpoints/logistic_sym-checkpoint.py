import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import os

# 文件列表
data_files = [
    r"F:\大四\tennis_predicton\processed_data_1\match_df_20_24_symmetry.csv",
    r"F:\大四\tennis_predicton\processed_data_1\match_df_15_19_symmetry.csv",
    r"F:\大四\tennis_predicton\processed_data_1\match_df_10_14_symmetry.csv",
    r"F:\大四\tennis_predicton\processed_data_1\match_df_05_09_symmetry.csv",
    r"F:\大四\tennis_predicton\processed_data_1\match_df_00_04_symmetry.csv"
]

for file in data_files:
    match_df = pd.read_csv(file, low_memory=False)
    dataset_name = file.split("\\")[-1].split(".")[0]  # 获取数据集名称

    # 选择特征
    hist_features = [col for col in match_df.columns if col.endswith('_hist')]
    hist_e_features = [col for col in match_df.columns if col.endswith('_hist_e')]
    histo_features = [col for col in match_df.columns if col.endswith('_histo')]
    elo_features = ['sym_elo']
    rank_features = ['sym_rank', 'sym_rank_points']
    supplement_features = ['round_code', 'best_of', 'sym_seed_bucket',
                           'sym_entry', 'sym_host', 'sym_hand', 'sym_ht', 'sym_age']

    features = hist_features + hist_e_features + histo_features + elo_features + rank_features + supplement_features
    continuous_features = hist_features + hist_e_features + histo_features + elo_features + rank_features + [
        'sym_ht', 'sym_age']

    # 目标变量
    X = match_df[features].copy()
    y = match_df['result']
    # 目标变量
    X = match_df[features].copy()
    y = match_df['result']

    # 归一化数值特征
    scaler = StandardScaler()
    X[continuous_features] = scaler.fit_transform(X[continuous_features])

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 定义参数范围，寻找最佳C值
    C_values = np.logspace(-3, 1, 10)  # 0.001 到 10 之间的10个C值
    best_C = None
    best_accuracy = 0

    for C in C_values:
        model = LogisticRegression(solver='liblinear', penalty='l2', C=C)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_C = C

    # 使用最佳C值训练最终模型
    final_model = LogisticRegression(solver='liblinear', penalty='l2', C=best_C)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred)

    # 计算特征重要性
    feature_importance = abs(final_model.coef_[0])  # 取绝对值，使其更易解释
    feature_importance_dict = {feature: importance for feature, importance in zip(features, feature_importance)}
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    # 输出结果到文件
    output_file = os.path.join(r"F:\大四\tennis_predicton\final_model\logistic", f"{dataset_name}_results.txt")
    with open(output_file, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Best C value: {best_C}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nFeature Importance:\n")
        for feature, importance in sorted_features:
            f.write(f"{feature}: {importance:.6f}\n")

    print(f"Results saved to {output_file}")
