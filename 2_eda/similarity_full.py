import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import jaccard, correlation
from scipy.special import rel_entr


# 计算 SMC（Simple Matching Coefficient）
def smc_similarity(df1, df2):
    return np.sum(df1 == df2) / len(df1)


# 计算 Bregman divergence（KL 散度）
def bregman_divergence(p, q):
    p, q = np.array(p), np.array(q)
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(rel_entr(p, q))


# 处理数据并计算相似性
def calculate_similarity(df):
    numeric_cols = ['winner_seed', 'winner_ht', 'winner_age', 'loser_seed', 'loser_ht', 'loser_age',
                    'winner_rank', 'winner_rank_points', 'loser_rank', 'loser_rank_points',
                    'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
                    'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced']
    categorical_cols = ['winner_hand', 'loser_hand']
    surface_cols = ['surface_Carpet', 'surface_Clay', 'surface_Grass', 'surface_Hard']

    # 处理类别型变量
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])

    # 标准化数值型数据
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

    # 计算相似性
    cosine_sim = cosine_similarity(df_scaled)
    euclidean_dist = euclidean_distances(df_scaled)
    correlation_dist = 1 - np.corrcoef(df_scaled, rowvar=False)

    # 计算 Jaccard 相似度
    jaccard_sim = np.array([[1 - jaccard(df[surface_cols].iloc[i], df[surface_cols].iloc[j])
                             for j in range(len(df))] for i in range(len(df))])

    # 计算 Bregman（KL 散度）
    bregman_div = np.array([[bregman_divergence(df_scaled.iloc[i], df_scaled.iloc[j])
                             for j in range(len(df))] for i in range(len(df))])

    return {
        'cosine': cosine_sim,
        'euclidean': euclidean_dist,
        'correlation': correlation_dist,
        'jaccard': jaccard_sim,
        'bregman': bregman_div
    }


# 加载数据并计算不同年份的相似性
file_paths = [
    f"../dataset/processed_data/df_{start:02d}_{end:02d}_cleaned.csv"
    for start, end in zip(range(0, 25, 5), range(4, 29, 5))
]

# 读取数据并存入字典
dfs = {f"df_{i}": pd.read_csv(file,low_memory=False) for i, file in zip(["00_04", "05_09", "10_14", "15_19", "20_24"], file_paths)}

similarity_results = {}
for year, df_name in zip(["00_04", "05_09", "10_14", "15_19", "20_24"], dfs.keys()):
    similarity_results[year] = calculate_similarity(dfs[df_name])

# 输出某个相似度矩阵示例
print(similarity_results['10_14']['cosine'])
