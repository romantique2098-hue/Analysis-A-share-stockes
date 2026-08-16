import argparse

from aquant.legacy import build_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算指定股票的量化特征")
    parser.add_argument("symbol")
    build_features(parser.parse_args().symbol)
