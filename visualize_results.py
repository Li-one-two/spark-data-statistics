from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# 数据源文件与输出目录
DATA_PATH = Path("data/user_behavior.csv")
FIGURE_DIR = Path("output/figures")
TABLE_DIR = Path("output/chinese_tables")


# 字段中文映射，便于后续生成中文表头
COLUMN_MAP = {
    "event_time": "行为时间",
    "user_id": "用户编号",
    "item_id": "商品编号",
    "category": "商品类目",
    "behavior": "行为类型",
    "province": "省份",
    "price": "商品单价",
    "quantity": "商品数量",
}

BEHAVIOR_MAP = {
    "pv": "浏览",
    "fav": "收藏",
    "cart": "加购",
    "buy": "购买",
}


def configure_chinese_font() -> None:
    """配置中文字体，尽量兼容 Windows 常见字体。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def load_data() -> pd.DataFrame:
    """读取原始数据并完成基础字段处理。"""
    # 读取原始行为日志，转换时间列并生成辅助分析字段
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df["event_time"] = pd.to_datetime(df["event_time"])
    # 提取日期与小时，方便按天、按小时做趋势分析
    df["event_date"] = df["event_time"].dt.date.astype(str)
    df["event_hour"] = df["event_time"].dt.hour
    # 只在购买行为下计算成交金额，其他行为金额记为 0
    df["pay_amount"] = df.apply(
        lambda row: row["price"] * row["quantity"] if row["behavior"] == "buy" else 0.0,
        axis=1,
    )
    return df


def save_chinese_table(df: pd.DataFrame, filename: str) -> None:
    """保存中文表头 CSV，避免结果文件中出现英文表头。"""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / filename, index=False, encoding="utf-8-sig")


def build_statistics(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """构造中文表头统计结果。"""
    # 先计算整体核心指标，用于生成“整体统计”表
    total_events = len(df)
    pv = int((df["behavior"] == "pv").sum())
    uv = df["user_id"].nunique()
    buy_events = int((df["behavior"] == "buy").sum())
    sales_volume = int(df.loc[df["behavior"] == "buy", "quantity"].sum())
    gmv = float(df["pay_amount"].sum())
    conversion_rate = round(buy_events / pv, 4) if pv else 0.0

    # 整体统计表：反映全局流量、用户和交易表现
    overall = pd.DataFrame(
        [
            {
                "总行为数": total_events,
                "浏览量": pv,
                "独立用户数": uv,
                "购买次数": buy_events,
                "销售件数": sales_volume,
                "成交总额": round(gmv, 2),
                "购买转化率": conversion_rate,
            }
        ]
    )

    # 按日期聚合，展示每日流量与成交趋势
    daily = (
        df.groupby("event_date")
        .agg(
            行为总数=("behavior", "size"),
            浏览量=("behavior", lambda values: int((values == "pv").sum())),
            独立用户数=("user_id", "nunique"),
            订单数=("behavior", lambda values: int((values == "buy").sum())),
            成交总额=("pay_amount", "sum"),
        )
        .reset_index()
        .rename(columns={"event_date": "日期"})
    )
    daily["成交总额"] = daily["成交总额"].round(2)

    # 按商品类目聚合，分析各类目销售表现
    category = (
        df.groupby("category")
        .agg(
            浏览量=("behavior", lambda values: int((values == "pv").sum())),
            独立用户数=("user_id", "nunique"),
            销售件数=("quantity", lambda values: int(values[df.loc[values.index, "behavior"] == "buy"].sum())),
            成交总额=("pay_amount", "sum"),
        )
        .reset_index()
        .rename(columns={"category": "商品类目"})
        .sort_values("成交总额", ascending=False)
    )
    category["成交总额"] = category["成交总额"].round(2)

    # 按省份聚合，观察不同地区的成交贡献
    province = (
        df.groupby("province")
        .agg(
            独立用户数=("user_id", "nunique"),
            浏览量=("behavior", lambda values: int((values == "pv").sum())),
            订单数=("behavior", lambda values: int((values == "buy").sum())),
            成交总额=("pay_amount", "sum"),
        )
        .reset_index()
        .rename(columns={"province": "省份"})
        .sort_values("成交总额", ascending=False)
    )
    province["成交总额"] = province["成交总额"].round(2)

    # 按小时聚合，分析访问与购买的时间段分布
    hourly = (
        df.groupby("event_hour")
        .agg(
            浏览量=("behavior", lambda values: int((values == "pv").sum())),
            订单数=("behavior", lambda values: int((values == "buy").sum())),
            成交总额=("pay_amount", "sum"),
        )
        .reset_index()
        .rename(columns={"event_hour": "小时"})
    )
    hourly["成交总额"] = hourly["成交总额"].round(2)

    # 只看购买行为，筛选出成交额前 10 的商品
    buy_df = df[df["behavior"] == "buy"]
    top_items = (
        buy_df.groupby(["item_id", "category"])
        .agg(销售件数=("quantity", "sum"), 成交总额=("pay_amount", "sum"))
        .reset_index()
        .rename(columns={"item_id": "商品编号", "category": "商品类目"})
        .sort_values(["成交总额", "销售件数"], ascending=[False, False])
        .head(10)
    )
    top_items["成交总额"] = top_items["成交总额"].round(2)

    # 统计各类用户行为的占比，便于分析用户行为分布
    behavior = (
        df.assign(行为名称=df["behavior"].map(BEHAVIOR_MAP))
        .groupby("行为名称")
        .size()
        .reset_index(name="行为次数")
        .sort_values("行为次数", ascending=False)
    )

    return {
        "整体统计.csv": overall,
        "每日统计.csv": daily,
        "类目统计.csv": category,
        "省份统计.csv": province,
        "小时统计.csv": hourly,
        "商品销售额前十.csv": top_items,
        "行为类型统计.csv": behavior,
    }


def add_bar_labels(ax) -> None:
    """给柱状图添加数值标签。"""
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=9)


def plot_daily_trend(daily: pd.DataFrame) -> None:
    # 绘制每日浏览量、订单数与成交总额的复合趋势图
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(daily["日期"], daily["浏览量"], marker="o", label="浏览量", color="#2f80ed")
    ax1.plot(daily["日期"], daily["订单数"], marker="s", label="订单数", color="#27ae60")
    ax1.set_xlabel("日期")
    ax1.set_ylabel("浏览量 / 订单数")
    ax1.tick_params(axis="x", rotation=25)

    ax2 = ax1.twinx()
    ax2.bar(daily["日期"], daily["成交总额"], alpha=0.25, label="成交总额", color="#f2994a")
    ax2.set_ylabel("成交总额")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.title("每日浏览量、订单数与成交总额趋势")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "每日趋势图.png", dpi=200)
    plt.close()


def plot_category_gmv(category: pd.DataFrame) -> None:
    # 用条形图对比各商品类目的成交总额表现
    fig, ax = plt.subplots(figsize=(10, 5))
    data = category.sort_values("成交总额", ascending=True)
    ax.barh(data["商品类目"], data["成交总额"], color="#2d9cdb")
    ax.set_xlabel("成交总额")
    ax.set_ylabel("商品类目")
    ax.set_title("各商品类目成交总额对比")
    for index, value in enumerate(data["成交总额"]):
        ax.text(value, index, f" {value:.0f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "类目成交总额对比图.png", dpi=200)
    plt.close()


def plot_province_gmv(province: pd.DataFrame) -> None:
    # 展示省份成交额排名前十的地域分布情况
    fig, ax = plt.subplots(figsize=(10, 5))
    data = province.head(10)
    ax.bar(data["省份"], data["成交总额"], color="#9b51e0")
    ax.set_xlabel("省份")
    ax.set_ylabel("成交总额")
    ax.set_title("省份成交总额前十名")
    ax.tick_params(axis="x", rotation=30)
    add_bar_labels(ax)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "省份成交总额前十图.png", dpi=200)
    plt.close()


def plot_hourly(df: pd.DataFrame) -> None:
    # 观察一天中浏览与购买行为的时间分布特征
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["小时"], df["浏览量"], marker="o", label="浏览量", color="#2f80ed")
    ax.plot(df["小时"], df["订单数"], marker="s", label="订单数", color="#eb5757")
    ax.set_xlabel("小时")
    ax.set_ylabel("数量")
    ax.set_title("每小时浏览量与订单数变化")
    ax.set_xticks(range(0, 24))
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "小时浏览订单变化图.png", dpi=200)
    plt.close()


def plot_top_items(top_items: pd.DataFrame) -> None:
    # 重点展示成交额最高的前十个商品及其类目
    fig, ax = plt.subplots(figsize=(10, 5))
    data = top_items.sort_values("成交总额", ascending=True)
    labels = data["商品编号"] + "（" + data["商品类目"] + "）"
    ax.barh(labels, data["成交总额"], color="#f2c94c")
    ax.set_xlabel("成交总额")
    ax.set_ylabel("商品")
    ax.set_title("商品销售额前十名")
    for index, value in enumerate(data["成交总额"]):
        ax.text(value, index, f" {value:.0f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "商品销售额前十图.png", dpi=200)
    plt.close()


def plot_behavior_pie(behavior: pd.DataFrame) -> None:
    # 用饼图表示不同用户行为类型的占比分布
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        behavior["行为次数"],
        labels=behavior["行为名称"],
        autopct="%.1f%%",
        startangle=90,
        counterclock=False,
    )
    ax.set_title("用户行为类型占比")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "用户行为类型占比图.png", dpi=200)
    plt.close()


def main() -> None:
    # 初始化中文字体与输出目录，保证图表和表格正常生成
    configure_chinese_font()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    statistics = build_statistics(df)

    for filename, table in statistics.items():
        save_chinese_table(table, filename)

    plot_daily_trend(statistics["每日统计.csv"])
    plot_category_gmv(statistics["类目统计.csv"])
    plot_province_gmv(statistics["省份统计.csv"])
    plot_hourly(statistics["小时统计.csv"])
    plot_top_items(statistics["商品销售额前十.csv"])
    plot_behavior_pie(statistics["行为类型统计.csv"])

    print(f"中文统计表已保存到：{TABLE_DIR.resolve()}")
    print(f"可视化图表已保存到：{FIGURE_DIR.resolve()}")


if __name__ == "__main__":
    main()
