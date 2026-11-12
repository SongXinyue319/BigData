from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, month, dayofweek, log1p, expm1

spark = SparkSession.builder \
    .appName("FinalAllModels") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("1. 数据读取与清洗...")
df = spark.read.csv("hdfs://localhost:9000/user/bigdata/data/retail_*.csv", header=True, inferSchema=True)
df_clean = df.filter((col("Quantity") > 0) & (col("Price") > 0) & (col("Customer ID").isNotNull()))
df_clean = df_clean.withColumn("TotalAmount", col("Quantity") * col("Price")).filter(col("TotalAmount") < 5000)

print("2. 特征工程...")
# 提取时间特征
df_feat = df_clean.withColumn("Month", month(col("InvoiceDate"))).withColumn("DayOfWeek", dayofweek(col("InvoiceDate")))
# 【进阶技巧】：对目标变量做 Log1p 变换，缓解长尾分布，提升模型拟合效果
df_feat = df_feat.withColumn("LogTotalAmount", log1p(col("TotalAmount")))

train_data, test_data = df_feat.randomSplit([0.8, 0.2], seed=42)

indexer = StringIndexer(inputCol="Country", outputCol="Country_idx", handleInvalid="skip")
assembler = VectorAssembler(
    inputCols=["Quantity", "Price", "Month", "DayOfWeek", "Country_idx"],
    outputCol="features"
)

# 评估器（注意：因为训练时取了对数，评估时要预测值还原回来 expm1）
evaluator_rmse = RegressionEvaluator(labelCol="TotalAmount", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="TotalAmount", predictionCol="prediction", metricName="r2")

print("3. 训练线性回归...")
lr = LinearRegression(featuresCol="features", labelCol="LogTotalAmount", maxIter=50)
lr_pipeline = Pipeline(stages=[indexer, assembler, lr])
lr_model = lr_pipeline.fit(train_data)
# 预测后还原
lr_pred = lr_model.transform(test_data).withColumn("prediction", expm1(col("prediction")))
print(f"LinearRegression RMSE: {evaluator_rmse.evaluate(lr_pred)}")
print(f"LinearRegression R2: {evaluator_r2.evaluate(lr_pred)}")

print("4. 训练随机森林...")
rf = RandomForestRegressor(featuresCol="features", labelCol="LogTotalAmount", numTrees=20, maxDepth=5, maxBins=64, seed=42)
rf_pipeline = Pipeline(stages=[indexer, assembler, rf])
rf_model = rf_pipeline.fit(train_data)
rf_pred = rf_model.transform(test_data).withColumn("prediction", expm1(col("prediction")))
print(f"RandomForest RMSE: {evaluator_rmse.evaluate(rf_pred)}")
print(f"RandomForest R2: {evaluator_r2.evaluate(rf_pred)}")

print("5. 训练 GBDT...")
# GBDT 通常效果最好，这里设置 maxIter=20 防止跑太久
gbt = GBTRegressor(featuresCol="features", labelCol="LogTotalAmount", maxIter=20, maxDepth=5, maxBins=64, seed=42)
gbt_pipeline = Pipeline(stages=[indexer, assembler, gbt])
gbt_model = gbt_pipeline.fit(train_data)
gbt_pred = gbt_model.transform(test_data).withColumn("prediction", expm1(col("prediction")))
print(f"GBDT RMSE: {evaluator_rmse.evaluate(gbt_pred)}")
print(f"GBDT R2: {evaluator_r2.evaluate(gbt_pred)}")

# 保存 GBDT 的抽样预测结果，用于画最后的散点图
gbt_pred.select("TotalAmount", "prediction").sample(False, 0.001, seed=42) \
    .toPandas().to_csv("/home/bigdata/predictions_gbdt.csv", index=False)

print("所有模型运行完毕！")
spark.stop()
