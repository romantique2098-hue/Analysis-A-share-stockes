from pathlib import Path
import akshare as ak
import pandas as pd



PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

def download_daily_data(
        symbol: str,
        start_date: str,
        end_date: str,
) ->pd.DataFrame:
    """下载一支A股的未复权日线行情"""
    print(f"正在下载 {symbol} 的日线行情数据...")

    market_symbol = (
        f"sh{symbol}"
        if symbol.startswith("6")
        else f"sz{symbol}"
    )
    raw = ak.stock_zh_a_daily(
        symbol=market_symbol,
        start_date=start_date,
        end_date=end_date,
        adjust="",
    )

    if raw.empty:
        raise ValueError(f"未下载到 {symbol} 的日线行情数据，请检查股票代码是否正确。")

    # 将数据源中的中文字符编码为方便编码的英文字符
    data = raw.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        }
    ).copy()

    required_columns = {
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError (f"下载的数据缺少字段 {missing_columns}, 请检查数据源是否发生变化。")

    data["symbol"] = symbol
    data["date"] = pd.to_datetime(data["date"])

    data = data[
        [
        "symbol", "date", "open", "close", "high", "low", "volume", "amount"
        ]
    ]

    #按日期排序并删除多余数据
    data=(
        data.sort_values("date")
        .drop_duplicates(subset=["symbol", "date"])
        .reset_index(drop=True)
    )
    invalid_prices = (
        (data["low"] > data["high"])
        | (data["open"] < data["low"])
        | (data["open"] > data["high"])
        | (data["close"] < data["low"])
        | (data["close"] > data["high"])
     )

    if invalid_prices.any():
        raise ValueError(f"下载的数据中存在不合理的开收盘价格")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RAW_DATA_DIR / f"{symbol}.parquet"
    data.to_parquet(output_path, index=False)

    print(f"共得到 {len(data)} 行数据")
    print(f"保存位置：{output_path}")

    return data


if __name__ == "__main__":
    today = pd.Timestamp.today().strftime("%Y%m%d")

    stock_data = download_daily_data(
        symbol="000001",
        start_date="20200101",
        end_date=today,
    )

    print("\n最早的五条记录：")
    print(stock_data.head())

    print("\n最近的五条记录：")
    print(stock_data.tail())
