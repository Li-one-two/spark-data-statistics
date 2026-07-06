"""
章鱼大数据平台兼容版：Spark 框架下电商用户行为数据统计。

说明：
该版本去掉了函数类型注解、f-string 等较新 Python 语法，适合 Python 3.5/3.6
平台环境使用。统计逻辑与 spark_statistics.py 保持一致。
"""

import argparse
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType


BEHAVIOR_BUY = "buy"


def create_spark(app_name="SparkStatisticsCourseDesign"):
    """创建 SparkSession。章鱼平台 Spark 可能不包含 Hive 支持，因此这里不调用 enableHiveSupport。"""
    # 集群环境下先初始化 SparkSession，作为后续所有 DataFrame 和 SQL 操作的入口。
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.warehouse.dir", "./warehouse")
        .getOrCreate()
    )


def load_behavior_data(spark, input_path):
    """读取用户行为 CSV 数据，并完成类型转换与基础清洗。"""
    # 显式定义 Schema，避免集群环境下字段类型推断不一致导致统计异常。
    schema = StructType(
        [
            StructField("event_time", StringType(), True),
            StructField("user_id", StringType(), True),
            StructField("item_id", StringType(), True),
            StructField("category", StringType(), True),
            StructField("behavior", StringType(), True),
            StructField("province", StringType(), True),
            StructField("price", DoubleType(), True),
            StructField("quantity", IntegerType(), True),
        ]
    )

    raw_df = spark.read.option("header", True).schema(schema).csv(input_path)

    # 把时间字符串转成时间戳，并派生日期/小时/成交金额字段，方便后续按天和按小时分析。
    cleaned_df = (
        raw_df.withColumn("event_ts", F.to_timestamp("event_time", "yyyy-MM-dd HH:mm:ss"))
        .withColumn("event_date", F.to_date("event_ts"))
        .withColumn("event_hour", F.hour("event_ts"))
        .withColumn(
            "pay_amount",
            F.when(F.col("behavior") == BEHAVIOR_BUY, F.col("price") * F.col("quantity")).otherwise(F.lit(0.0)),
        )
        .dropna(subset=["event_ts", "user_id", "item_id", "category", "behavior", "province"])
    )
    return cleaned_df


def overall_statistics(df):
    """统计整体 PV、UV、购买次数、销量、GMV 和购买转化率。"""
    # 这组指标是全局业务概览，反映整体流量、用户和交易规模。
    return df.agg(
        F.count(F.lit(1)).alias("total_events"),
        F.sum(F.when(F.col("behavior") == "pv", 1).otherwise(0)).alias("pv"),
        F.countDistinct("user_id").alias("uv"),
        F.sum(F.when(F.col("behavior") == BEHAVIOR_BUY, 1).otherwise(0)).alias("buy_events"),
        F.sum(F.when(F.col("behavior") == BEHAVIOR_BUY, F.col("quantity")).otherwise(0)).alias("sales_volume"),
        F.round(F.sum("pay_amount"), 2).alias("gmv"),
        F.round(
            F.sum(F.when(F.col("behavior") == BEHAVIOR_BUY, 1).otherwise(0))
            / F.sum(F.when(F.col("behavior") == "pv", 1).otherwise(0)),
            4,
        ).alias("buy_conversion_rate"),
    )


def daily_statistics(df):
    """按日期统计访问、用户和交易指标。"""
    # 按日期聚合可以观察用户活跃度和成交趋势变化。
    return (
        df.groupBy("event_date")
        .agg(
            F.count(F.lit(1)).alias("events"),
            F.sum(F.when(F.col("behavior") == "pv", 1).otherwise(0)).alias("pv"),
            F.countDistinct("user_id").alias("uv"),
            F.sum(F.when(F.col("behavior") == BEHAVIOR_BUY, 1).otherwise(0)).alias("orders"),
            F.round(F.sum("pay_amount"), 2).alias("gmv"),
        )
        .orderBy("event_date")
    )


def category_statistics(df):
    """按商品类目统计热度、销量和销售额。"""
    # 类目分析用于识别重点商品方向和高价值业务板块。
    return (
        df.groupBy("category")
        .agg(
            F.sum(F.when(F.col("behavior") == "pv", 1).otherwise(0)).alias("pv"),
            F.countDistinct("user_id").alias("uv"),
            F.sum(F.when(F.col("behavior") == BEHAVIOR_BUY, F.col("quantity")).otherwise(0)).alias("sales_volume"),
            F.round(F.sum("pay_amount"), 2).alias("gmv"),
        )
        .withColumn(
            "category_conversion_rate",
            F.round(F.col("sales_volume") / F.when(F.col("pv") == 0, None).otherwise(F.col("pv")), 4),
        )
        .orderBy(F.desc("gmv"), F.desc("pv"))
    )


def province_statistics(df):
    """按省份统计区域消费能力。"""
    # 按省份汇总可以发现不同地区的消费热度与收入贡献。
    return (
        df.groupBy("province")
        .agg(
            F.countDistinct("user_id").alias("uv"),
            F.sum(F.when(F.col("behavior") == "pv", 1).otherwise(0)).alias("pv"),
            F.sum(F.when(F.col("behavior") == BEHAVIOR_BUY, 1).otherwise(0)).alias("orders"),
            F.round(F.sum("pay_amount"), 2).alias("gmv"),
        )
        .orderBy(F.desc("gmv"))
    )


def hourly_statistics(df):
    """按小时统计访问峰值。"""
    # 每小时统计适合判断流量高峰时段，辅助运营活动安排。
    return (
        df.groupBy("event_hour")
        .agg(
            F.sum(F.when(F.col("behavior") == "pv", 1).otherwise(0)).alias("pv"),
            F.sum(F.when(F.col("behavior") == BEHAVIOR_BUY, 1).otherwise(0)).alias("orders"),
            F.round(F.sum("pay_amount"), 2).alias("gmv"),
        )
        .orderBy("event_hour")
    )


def top_items(df, top_n=10):
    """统计销售额 TopN 商品。"""
    # 只看购买行为，再按成交额和销量排序，可以筛选出重点商品。
    return (
        df.filter(F.col("behavior") == BEHAVIOR_BUY)
        .groupBy("item_id", "category")
        .agg(
            F.sum("quantity").alias("sales_volume"),
            F.round(F.sum("pay_amount"), 2).alias("gmv"),
        )
        .orderBy(F.desc("gmv"), F.desc("sales_volume"))
        .limit(top_n)
    )


def save_results(results, output_path):
    """将统计结果写出为 Parquet 与 CSV。"""
    # Parquet 用于后续数据仓库/作业消费，CSV 方便直接查看与调试。
    for name, result_df in results.items():
        result_dir = os.path.join(output_path, name)
        result_df.coalesce(1).write.mode("overwrite").parquet(os.path.join(result_dir, "parquet"))
        result_df.coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(result_dir, "csv"))


def create_hive_tables(spark, results, output_path):
    """创建 Hive 表映射 Parquet 结果。"""
    # 把结果注册成 Hive 外部表，方便后续在集群上做进一步查询与报表。
    spark.sql("CREATE DATABASE IF NOT EXISTS ecommerce_dw")
    spark.sql("USE ecommerce_dw")
    for name, result_df in results.items():
        table_name = "ads_{}".format(name)
        parquet_path = os.path.abspath(os.path.join(output_path, name, "parquet")).replace("\\", "/")
        result_df.createOrReplaceTempView("tmp_{}".format(name))
        spark.sql("DROP TABLE IF EXISTS {}".format(table_name))
        spark.sql(
            """
            CREATE TABLE {table_name}
            USING PARQUET
            LOCATION '{parquet_path}'
            AS SELECT * FROM tmp_{name}
            """.format(table_name=table_name, parquet_path=parquet_path, name=name)
        )


def write_to_sqlserver_if_enabled(results, sqlserver_url, sqlserver_user, sqlserver_password):
    """可选：通过 JDBC 将分析结果写入 SQL Server。"""
    # 如果未提供 SQL Server 地址，就跳过这一步，避免在无配置环境下报错。
    if not sqlserver_url:
        return

    for name, result_df in results.items():
        result_df.write.mode("overwrite").format("jdbc").option("url", sqlserver_url).option("dbtable", "ads_{}".format(name)).option(
            "user", sqlserver_user
        ).option("password", sqlserver_password).option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver").save()


def parse_args():
    # 命令行参数让脚本能适配本地、HDFS 和 SQL Server 等不同运行方式。
    parser = argparse.ArgumentParser(description="Spark 框架下电商用户行为数据统计课程设计")
    parser.add_argument("--input", default="data/user_behavior.csv", help="输入数据路径，本地或 hdfs:// 路径均可")
    parser.add_argument("--output", default="output/statistics", help="统计结果输出路径")
    parser.add_argument("--top-n", type=int, default=10, help="TopN 商品数量")
    parser.add_argument("--sqlserver-url", default="", help="SQL Server JDBC URL")
    parser.add_argument("--sqlserver-user", default="sa", help="SQL Server 用户名")
    parser.add_argument("--sqlserver-password", default="", help="SQL Server 密码")
    parser.add_argument("--no-save", action="store_true", help="只显示统计结果，不保存 CSV/Parquet 文件")
    parser.add_argument("--no-hive", action="store_true", help="不创建 Hive/Spark SQL 结果表")
    return parser.parse_args()


def main():
    # 主流程串联读取、统计、展示、保存和写库等步骤。
    args = parse_args()
    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    behavior_df = load_behavior_data(spark, args.input)
    behavior_df.cache()

    results = {
        "overall": overall_statistics(behavior_df),
        "daily": daily_statistics(behavior_df),
        "category": category_statistics(behavior_df),
        "province": province_statistics(behavior_df),
        "hourly": hourly_statistics(behavior_df),
        "top_items": top_items(behavior_df, args.top_n),
    }

    for name, result_df in results.items():
        print("\n========== {} ==========".format(name))
        result_df.show(truncate=False)

    if not args.no_save:
        save_results(results, args.output)
    if not args.no_hive:
        create_hive_tables(spark, results, args.output)
    write_to_sqlserver_if_enabled(results, args.sqlserver_url, args.sqlserver_user, args.sqlserver_password)

    behavior_df.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
