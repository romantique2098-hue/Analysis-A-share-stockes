import pandas as pd

from aquant.legacy import download_daily_data

if __name__ == "__main__":
    today = pd.Timestamp.today().strftime("%Y%m%d")

    stock_data = download_daily_data("000001", "20200101", today)

    print("\n最早的五条记录：")
    print(stock_data.head())

    print("\n最近的五条记录：")
    print(stock_data.tail())
