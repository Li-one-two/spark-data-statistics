import argparse
import os
from typing import Dict

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType


BEHAVIOR_BUY = "buy"


def create_spark(app_name: str = "SparkStatisticsCourseDesign") -> SparkSession:
    """创建 SparkSession，并开启 Hive 支持以模拟数据仓库环境。"""
    # SparkSession 是所有 DataFrame / SQL 操作的入口，也是本项目的主运行环境。
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.warehouse.dir", "./warehouse")
        .enableHiveSupport()
        .getOrCreate()
    )


def load_behavior_data(spark: SparkSession, input_path: str) -> DataFrame:
    """读取用户行为 CSV 数据，并完成类型转换与基础清洗。"""
    # 显式定义 Schema，保证字段类型稳定，避免 Spark 推断带来的异常。
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

    # 关键清洗：把时间字符串转成时间戳，补充日期/小时字段，并派生成交金额。
    # 只有购买行为才会计入 GMV，其余行为金额为 0，避免统计偏差。
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


def overall_statistics(df: DataFrame) -> DataFrame:
    """统计整体 PV、UV、购买次数、销量、GMV 和购买转化率。"""
    # 全局指标用于快速掌握业务总量和转化情况，是后续分析的基线。
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


def daily_statistics(df: DataFrame) -> DataFrame:
    """按日期统计访问、用户和交易指标。"""
    # 按天聚合可以观察流量和交易趋势，适合做时间序列分析。
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


def category_statistics(df: DataFrame) -> DataFrame:
    """按商品类目统计热度、销量和销售额，用于分析重点品类。"""
    # 这一步是品类分析的核心，用来判断哪个类目最值得重点运营。
    return (
        df.groupBy("category")
        .agg(
            F.sum(F.when(F.col("behavior") == "pv", 1).otherwise(0)).alias("pv"),
            F.countDistinct("user_id").alias("uv"),
            F.sum(F.when(F.col("behavior") == BEHAVIOR_BUY, F.col("quantity")).otherwise(0)).alias("sales_volume"),
            F.round(F.sum("pay_amount"), 2).alias("gmv"),
        )
        .withColumn("category_conversion_rate", F.round(F.col("sales_volume") / F.when(F.col("pv") == 0, None).otherwise(F.col("pv")), 4))
        .orderBy(F.desc("gmv"), F.desc("pv"))
    )


def province_statistics(df: DataFrame) -> DataFrame:
    """按省份统计区域消费能力。"""
    # 按省份聚合能反映不同地区的消费活跃度与购买贡献。
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


def hourly_statistics(df: DataFrame) -> DataFrame:
    """按小时统计访问峰值，体现 Spark 对时间维度日志统计的支持。"""
    # 按小时统计是分析用户活跃时间段的关键步骤，便于优化活动安排。
    return (
        df.groupBy("event_hour")
        .agg(
            F.sum(F.when(F.col("behavior") == "pv", 1).otherwise(0)).alias("pv"),
            F.sum(F.when(F.col("behavior") == BEHAVIOR_BUY, 1).otherwise(0)).alias("orders"),
            F.round(F.sum("pay_amount"), 2).alias("gmv"),
        )
        .orderBy("event_hour")
    )


def top_items(df: DataFrame, top_n: int = 10) -> DataFrame:
    """统计销售额 TopN 商品。"""
    # 只保留购买行为，再按成交额和销量排序，可以得到重点商品名单。
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


def save_results(results: Dict[str, DataFrame], output_path: str) -> None:
    """将统计结果写出为 Parquet 与 CSV，便于 Hive 建表和人工查看。"""
    # Parquet 适合后续 Hive / 数据仓库读取，CSV 适合人工查看与调试。
    for name, result_df in results.items():
        result_dir = os.path.join(output_path, name)
        result_df.coalesce(1).write.mode("overwrite").parquet(os.path.join(result_dir, "parquet"))
        result_df.coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(result_dir, "csv"))


def create_hive_tables(spark: SparkSession, results: Dict[str, DataFrame], output_path: str) -> None:
    """创建 Hive 外部表映射 Parquet 结果，体现数据仓库集成。"""
    # 这里把分析结果注册成 Hive 表，方便后续在数据仓库中直接查询。
    spark.sql("CREATE DATABASE IF NOT EXISTS ecommerce_dw")
    spark.sql("USE ecommerce_dw")
    for name, result_df in results.items():
        table_name = f"ads_{name}"
        parquet_path = os.path.abspath(os.path.join(output_path, name, "parquet")).replace("\\", "/")
        result_df.createOrReplaceTempView(f"tmp_{name}")
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")
        spark.sql(
            f"""
            CREATE TABLE {table_name}
            USING PARQUET
            LOCATION '{parquet_path}'
            AS SELECT * FROM tmp_{name}
            """
        )


def write_to_sqlserver_if_enabled(
    results: Dict[str, DataFrame], sqlserver_url: str, sqlserver_user: str, sqlserver_password: str
) -> None:
    """可选：通过 JDBC 将分析结果写入 SQL Server；需要提交 mssql-jdbc 驱动。"""
    # 如果未配置 SQL Server 地址，就跳过这一步，避免运行时报错。
    if not sqlserver_url:
        return

    for name, result_df in results.items():
        result_df.write.mode("overwrite").format("jdbc").option("url", sqlserver_url).option("dbtable", f"ads_{name}").option(
            "user", sqlserver_user
        ).option("password", sqlserver_password).option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver").save()


def parse_args() -> argparse.Namespace:
    # 命令行参数用于适配本地运行、HDFS 路径以及 SQL Server 输出等不同场景。
    parser = argparse.ArgumentParser(description="Spark 框架下电商用户行为数据统计课程设计")
    parser.add_argument("--input", default="data/user_behavior.csv", help="输入数据路径，本地或 hdfs:// 路径均可")
    parser.add_argument("--output", default="output/statistics", help="统计结果输出路径")
    parser.add_argument("--top-n", type=int, default=10, help="TopN 商品数量")
    parser.add_argument(
        "--sqlserver-url",
        default="",
        help="SQL Server JDBC URL，例如 jdbc:sqlserver://localhost:1433;databaseName=bigdata_result;encrypt=false;trustServerCertificate=true",
    )
    parser.add_argument("--sqlserver-user", default="sa", help="SQL Server 用户名")
    parser.add_argument("--sqlserver-password", default="", help="SQL Server 密码")
    parser.add_argument("--no-save", action="store_true", help="只显示统计结果，不保存 CSV/Parquet 文件")
    parser.add_argument("--no-hive", action="store_true", help="不创建 Hive/Spark SQL 结果表")
    return parser.parse_args()


def main() -> None:
    # 入口函数负责串联读取、统计、保存和输出等完整流程。
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
        print(f"\n========== {name} ==========")
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
