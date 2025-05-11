import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from pylab import mpl

# 设置复合字体
mpl.rcParams["font.sans-serif"] = ["Helvetica", "Microsoft YaHei"]
mpl.rcParams["axes.unicode_minus"] = False

# 读取数据（路径保持原样）
file_paths = {
    "00_04": r"F:\大四\tennis_predicton\processed_data\df_00_04_cleaned.csv",
    "05_09": r"F:\大四\tennis_predicton\processed_data\df_05_09_cleaned.csv",
    "10_14": r"F:\大四\tennis_predicton\processed_data\df_10_14_cleaned.csv",
    "15_19": r"F:\大四\tennis_predicton\processed_data\df_15_19_cleaned.csv",
    "20_24": r"F:\大四\tennis_predicton\processed_data\df_20_24_cleaned.csv"
}

dfs = {key: pd.read_csv(path) for key, path in file_paths.items()}

# 创建输出目录
output_dir = r"F:\大四\tennis_predicton\Pure_EDA_Results"
os.makedirs(output_dir, exist_ok=True)

# 分析结果存储结构
analysis_data = {
    'missing_values': {},
    'desc_stats': {},
    'category_counts': {}
}

def format_period(key):
    """将00_04格式转换为2000-2004"""
    start = 2000 + int(key[:2])
    end = 2000 + int(key[3:])
    return f"{start}-{end}"


def pure_eda(df, period_name, file_prefix):
    print(f"\n=== 正在分析 {period_name} 数据集 ===")

    # 1. 基础信息分析
    analysis_data['missing_values'][period_name] = df.isna().sum()
    analysis_data['desc_stats'][period_name] = df.describe(include='all')

    # 2. 分类变量分布分析
    categorical_cols = ['surface', 'tourney_level', 'round', 'winner_hand', 'loser_hand']
    for col in categorical_cols:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=col, data=df, order=df[col].value_counts().index)
            plt.title(f"{period_name} - {col} 分布")
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(output_dir, f"{file_prefix}_{col}_dist.png"))
            plt.close()

    # 3. 数值变量分布分析
    numerical_cols = ['winner_age', 'loser_age', 'winner_ht', 'loser_ht',
                      'winner_rank', 'loser_rank', 'minutes']
    for col in numerical_cols:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f"{period_name} - {col} 分布")
            plt.savefig(os.path.join(output_dir, f"{file_prefix}_{col}_dist.png"))
            plt.close()

    # 4. 比赛特征分析
    # 4.1 赛事规模随时间变化
    if 'tourney_year' in df.columns:
        plt.figure(figsize=(12, 6))
        df.groupby('tourney_year')['draw_size'].mean().plot(kind='bar')
        plt.title(f"{period_name} - 年度平均赛事规模")
        plt.savefig(os.path.join(output_dir, f"{file_prefix}_drawsize_trend.png"))
        plt.close()

    # 4.2 发球数据对比
    ace_cols = ['w_ace', 'l_ace']
    if all(col in df.columns for col in ace_cols):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[ace_cols])
        plt.title(f"{period_name} - ACE球分布对比")
        plt.savefig(os.path.join(output_dir, f"{file_prefix}_ace_comparison.png"))
        plt.close()

    # 5. 选手特征分析
    # 5.1 左右手持拍分布
    if 'winner_hand' in df.columns:
        hand_dist = pd.DataFrame({
            'Winner': df['winner_hand'].value_counts(),
            'Loser': df['loser_hand'].value_counts()
        })
        hand_dist.plot(kind='bar', figsize=(10, 6))
        plt.title(f"{period_name} - 选手持手方式分布")
        plt.savefig(os.path.join(output_dir, f"{file_prefix}_hand_dist.png"))
        plt.close()

    # 5.2 国家分布分析
    for role in ['winner', 'loser']:
        col = f'{role}_ioc'
        if col in df.columns:
            plt.figure(figsize=(12, 6))
            df[col].value_counts().head(10).plot(kind='bar')
            plt.title(f"{period_name} - {role} 国家分布 Top10")
            plt.savefig(os.path.join(output_dir, f"{file_prefix}_{role}_ioc.png"))
            plt.close()

    # 6. 比赛时长分析
    if 'minutes' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='surface', y='minutes', data=df)
        plt.title(f"{period_name} - 不同场地比赛时长")
        plt.savefig(os.path.join(output_dir, f"{file_prefix}_minutes_by_surface.png"))
        plt.close()

    # 7. 破发点分析
    bp_cols = ['w_bpSaved', 'w_bpFaced', 'l_bpSaved', 'l_bpFaced']
    if all(col in df.columns for col in bp_cols):
        plt.figure(figsize=(12, 6))
        (df['w_bpSaved'] / df['w_bpFaced']).plot(kind='kde', label='Winner')
        (df['l_bpSaved'] / df['l_bpFaced']).plot(kind='kde', label='Loser')
        plt.title(f"{period_name} - 破发点挽救率分布")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{file_prefix}_bp_save_rate.png"))
        plt.close()


# 执行分析
for period_key, df in dfs.items():
    friendly_name = f"{format_period(period_key)}年数据"  # 转换为2000-2004年数据
    pure_eda(df.copy(), friendly_name, period_key)
    
# 保存分析结果
pd.DataFrame(analysis_data['missing_values']).to_csv(
    os.path.join(output_dir, "missing_values_report.csv"))
pd.concat(analysis_data['desc_stats'], axis=1).to_csv(
    os.path.join(output_dir, "desc_stats_report.csv"))

print(f"分析完成，结果已保存至：{output_dir}")