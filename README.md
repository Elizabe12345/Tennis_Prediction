# 网球比赛结果预测项目

## 项目概述
本仓库包含完整的机器学习流水线，用于预测ATP网球赛事比赛结果。项目涵盖数据预处理、探索性分析、特征工程、多模型训练和集成学习，通过模块化结构保证可复现性。
使用anaconda和pycharm实现

---

## 目录结构

### 核心模块
| 目录 | 描述 |
|-----------|-------------|
| **/preprocessing** | 数据合并、数据清洗、分类编码 |
| **/feature_enginnering** | 特征构造、特征选择、elo |
| **/eda** | 探索性数据分析、相似性 |
| **/explore** | 数据分析探索（对称性、历史估计、对比试验） |

### 数据存储
| 目录 | 内容 |
|-----------|----------|
| **/tennis_atp** | 原始ATP比赛数据(.csv) |
| **/processed_data** | 清洗后数据集<br/>└─/merged: 合并后的赛事数据 |
| **/processed_data_1** | 系统优化后数据集 |

### 机器学习模型
| 目录 | 内容 |
|-----------|----------|
| **/model** | 基础模型实现（随机森林/SVM等） |
| **/final_model** | 最终模型：<br/>- /ann: 神经网络<br/>- /logistic: 逻辑回归变体<br/>- /xgboost: 优化提升树模型 |
| **/ensemble** | 集成模型 |

### eda结果
| 目录 | 内容 |
|-----------|----------|
| **/EDA_results** | 可视化分析结果（比赛时长分布/场地统计） |
| **/Pure_EDA_Results** | eda结果 |

## 核心依赖环境

### 关键组件版本
| 类别              | 包名称             | 版本       | 安装源       |
|--------------------|--------------------|------------|--------------|
| **编程语言**      | Python             | 3.8.20     | conda-forge  |
| **机器学习框架**  | XGBoost            | 2.0.3      | conda-forge  |
|                   | CatBoost           | 1.2.5      | conda-forge  |
|                   | LightGBM           | 3.3.5      | conda-forge  |
|                   | scikit-learn       | 0.22.1     | conda-forge  |
| **数据处理**      | pandas             | 0.25.3     | conda-forge  |
|                   | NumPy              | 1.16.5     | conda-forge  |
|                   | SciPy              | 1.5.3      | conda-forge  |
| **可视化**        | matplotlib         | 3.3.3      | conda-forge  |
|                   | seaborn            | 0.11.2     | conda-forge  |
|                   | plotly             | 5.24.1     | conda-forge  |
| **特征工程**      | imbalanced-learn   | 0.7.0      | conda-forge  |
|                   | shap               | 0.40.0     | conda-forge  |
