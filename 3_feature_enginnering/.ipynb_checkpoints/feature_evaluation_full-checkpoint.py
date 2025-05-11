import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pointbiserialr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif

# 读取数据集
match_df = pd.read_csv(r"..\dataset\processed_data\match_df_20_24_symmetry.csv", low_memory=False)

# 定义特征分类
categorical_features = [
    'round_code', 'best_of', 'surface_Clay', 'surface_Grass', 'surface_Hard',
    'sym_seed_bucket', 'sym_entry', 'sym_host', 'sym_hand'
]

continuous_features = [
    'sym_ace_hist','sym_df_hist', 'sym_svpt_hist', 'sym_fstIn_hist', 'sym_fstWon_hist',
    'sym_sndWon_hist', 'sym_SvGms_hist', 'sym_bpSaved_hist', 'sym_bpFaced_hist',
    'sym_baseline_rally_hist', 'sym_intensity_hist', 'sym_ace_rate_hist', 'sym_df_rate_hist',
    'sym_serve_win_rate_hist', 'sym_serve_efficiency_hist', 'sym_clutch_ability_hist',
    'sym_ht', 'sym_age', 'sym_elo_before',
    'sym_rank','sym_rank_points'
]
continuous1_features = [
    'sym_ht', 'sym_age', 'sym_elo_before', 'sym_ace', 'sym_df', 'sym_svpt',
    'sym_fstIn', 'sym_fstWon', 'sym_sndWon', 'sym_SvGms', 'sym_bpSaved',
    'sym_bpFaced', 'sym_ace_rate', 'sym_df_rate', 'sym_serve_win_rate',
    'sym_serve_efficiency', 'sym_clutch_ability',

]


y = match_df['result']  # 目标变量

# 合并所有特征
all_features = categorical_features + continuous_features

### 1. Spearman相关系数（适用于所有特征）
spearman_corr = {col: spearmanr(match_df[col], y)[0] for col in all_features}
spearman_df = pd.DataFrame(spearman_corr.items(), columns=['Feature', 'Spearman Correlation']).sort_values(
    by='Spearman Correlation', ascending=False)

### 2. 互信息（区分离散/连续特征）
X = match_df[all_features]
discrete_mask = [col in categorical_features for col in X.columns]  # 创建离散特征掩码
mi_scores = mutual_info_classif(X, y, discrete_features=discrete_mask)
mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi_scores}).sort_values(
    by='Mutual Information', ascending=False)

### 3. 点二列相关系数（仅连续特征）
point_biserial_corr = {col: pointbiserialr(match_df[col], y)[0] for col in continuous_features}
pb_df = pd.DataFrame(point_biserial_corr.items(), columns=['Feature', 'Point-Biserial Correlation']).sort_values(
    by='Point-Biserial Correlation', ascending=False)

### 4. 卡方检验（仅分类特征）
chi2_results = {}
for col in categorical_features:
    contingency_table = pd.crosstab(match_df[col], y)
    chi2_stat, *_ = chi2_contingency(contingency_table)
    chi2_results[col] = chi2_stat

chi2_df = pd.DataFrame(chi2_results.items(), columns=['Feature', 'Chi-Square Statistic']).sort_values(
    by='Chi-Square Statistic', ascending=False)

# 输出结果
print("Spearman 相关性:\n", spearman_df)
print("\n互信息（MI）:\n", mi_df)
print("\n点二列相关系数:\n", pb_df)
print("\n卡方检验:\n", chi2_df)