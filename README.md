# A 股均线策略研究

这是一个使用 Python 完成的 A 股量化研究项目。项目以股票 `000001` 为示例，建立了从行情下载、特征构建、交易成本建模到参数敏感性分析和滚动样本外验证的完整流程。

## 项目目标

- 检验简单均线择时策略是否具有稳定的投资价值。
- 避免未来函数，并将佣金、印花税和滑点纳入回测。
- 通过样本外测试识别参数过拟合，而不是只展示历史最优结果。
- 生成可复核的 CSV 结果和净值、回撤图。

## 主要结论

数据范围为 2020-01-02 至 2026-07-14。2022—2025 年滚动样本外验证结果如下：

| 指标 | 滚动均线策略 | 买入持有 |
| --- | ---: | ---: |
| 最终净值 | 0.8901 | 0.8322 |
| 年化收益率 | -2.88% | -4.50% |
| 最大回撤 | -21.99% | -45.90% |
| 夏普比率 | -0.2860 | -0.1451 |

均线策略没有实现稳定盈利，但相对买入持有减少了约 23.91 个百分点的最大回撤。研究结果说明，该策略在风险控制方面有一定效果，但仍不足以证明其具有可交易的稳定超额收益。

## 项目结构

```text
.
├── data/
│   ├── raw/                 # 原始后复权日线数据
│   └── features/            # 特征与基础回测数据
├── output/                  # 回测图和研究结果表
├── src/
│   ├── download_data.py     # 下载并校验行情数据
│   ├── build_features.py    # 构建均线、仓位、成本和绩效指标
│   ├── plot_backtask.py     # 绘制净值及回撤曲线
│   ├── cost_sensitivity.py  # 交易成本敏感性分析
│   ├── parameter_sensitivity.py # 参数和样本外稳健性分析
│   └── batch_research.py    # 批量运行跨股票研究
└── requirements.txt
```

## 安装与运行

建议使用 Python 3.10 或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

项目已经包含 `000001` 的示例数据，可以直接运行：

```bash
python src/build_features.py 000001
python src/cost_sensitivity.py 000001
python src/parameter_sensitivity.py
MPLBACKEND=Agg python src/plot_backtask.py 000001
python src/batch_research.py
```

如需更新行情数据，可运行：

```bash
python src/download_data.py
```

## 方法说明

- 当日收盘后生成信号，向后移动一个交易日后再作为实际仓位，避免未来信息泄漏。
- 成本模型包含双边佣金、卖出印花税和滑点。
- 参数网格覆盖 8 组短期/长期均线组合。
- 固定样本切分用于观察参数退化，滚动样本外验证用于模拟逐年重新选择参数。
- 结果仅用于学习和研究，不构成任何投资建议。
