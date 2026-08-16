import argparse

from aquant.legacy import plot_backtest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制指定股票的回测净值曲线")
    parser.add_argument("symbol")
    plot_backtest(parser.parse_args().symbol)
