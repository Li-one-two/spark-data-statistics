# spark-data-statistics
基于 Spark 的电商用户行为大数据分析系统。使用 PySpark/Spark SQL 对 20,000 条用户行为日志进行清洗与多维统计（整体/日/类目/省份/小时指标及商品 TopN）。支持 Windows 本地开发与 YARN 集群部署，计算结果持久化至 HDFS（CSV/Parquet）并同步至 SQL Server 实现可视化展示。
