# 基于 Spark MLlib 的电商交易金额回归预测与可视化分析系统

## 1. 项目简介
本项目为大数据处理与分析课程大作业（独立完成）。项目基于真实的电商交易数据集（UCI Online Retail II，超 100 万行记录），在 Linux (Ubuntu) 环境下搭建了 Hadoop 伪分布式集群与 Spark 计算框架。使用 PySpark 进行了数据清洗、特征工程，并对比了线性回归、随机森林与 GBDT 三种回归模型，最终将模型持久化至 HDFS。

## 2. 技术栈
- **操作系统**：Ubuntu 24.04 LTS
- **分布式存储**：Hadoop 3.3.6 (HDFS 伪分布式)
- **计算框架**：Apache Spark 3.5.1 (PySpark)
- **机器学习库**：Spark MLlib
- **可视化**：Python (Matplotlib, Seaborn)

## 3. 数据说明
- **数据源**：Online Retail II (UCI Machine Learning Repository)
- **原始数据规模**：1,067,371 行，8 列。
- **预处理**：剔除退货记录（Quantity<=0）、异常价格（Price<=0）、缺失客户 ID 的数据，并过滤掉单笔超过 5000 的极端金额。
- **清洗后规模**：805,534 行。
- **目标变量**：`TotalAmount = Quantity * Price`

## 4. 核心算法与优化策略
- **特征工程**：提取 InvoiceDate 的月份和星期特征，对 Country 进行 StringIndexer 编码。
- **数据变换**：对目标变量 `TotalAmount` 引入 `Log1p` 对数变换，缓解长尾分布问题。
- **模型训练**：使用 Spark ML Pipeline 训练了三种模型：
  1. 线性回归 (Linear Regression)：因对数变换导致梯度发散，实验失效。
  2. 随机森林 (Random Forest)：RMSE = 63.80, R² = 0.14。
  3. **GBDT (梯度提升树)**：**RMSE = 34.98, R² = 0.741（表现最优）**。

## 5. 运行指南
1. 启动 Hadoop 集群：
   `start-dfs.sh`
2. 将数据上传至 HDFS：
   `hdfs dfs -put retail_*.csv /user/bigdata/data/`
3. 提交 Spark 任务：
   `spark-submit retail_final_all.py`
4. 运行可视化脚本：
   `python3 plot_final.py`

## 6. 文件结构
- `retail_final_all.py`: 核心 PySpark 数据处理与模型训练代码。
- `plot_final.py`: 结果可视化画图代码。
- `README.md`: 项目说明文档。
