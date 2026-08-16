# AQuant：A股量化分析与模拟盘

AQuant 是一个面向个人研究的日频量化软件。它可以轮换维护全 A 股本地行情、定时执行行业中性多因子选股、运行无未来函数的组合回测、生成收盘信号，并在 SQLite 中记录下一交易日的模拟订单、成交、持仓和资金曲线。

> 本项目不会连接券商，也不会发送真实委托。免费行情可能延迟、中断或修订，输出仅供研究，不构成投资建议。

## 已实现能力

- AkShare 行情适配器：东方财富为主、失败后切换新浪，仍失败时保留最后有效缓存。
- 未复权行情用于模拟成交与估值，前复权行情用于指标和研究回测。
- 七种可比较策略：均线趋势、突破趋势、横截面动量、RSI/布林带均值回归、等权组合和风险组合。
- 全 A 股代码同步、断点轮换更新、20/60/120 日动量预选，以及价值、质量、低波、流动性行业中性评分。
- 独立行业分类缓存：申万一级行业全量更新、巨潮逐股回退、错误追踪和 SQLite 手工修正。
- 等权或逆波动率配置、目标波动率、现金缓冲、单票/行业上限、最大回撤冷静期。
- 动量排名缓冲、最短持有期、最小调仓阈值和单日换手预算，减少无效往返交易。
- T+1、100股交易单位、佣金最低收费、卖出印花税和滑点。
- 费前/费后/买入持有/沪深300曲线，以及超额收益、跟踪误差、信息比率、Beta、Alpha 等相对绩效指标和年度归因。
- 幂等的每日模拟盘、SQLite 自动备份、CSV/Parquet 结果和轮转日志。
- 命令行与 Streamlit 五页本地网页。

## 安装

项目支持 Python 3.11–3.13。以下命令均在项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```

如需开发工具，可改用 `python -m pip install -e ".[dev]"`。

## 首次运行

先从公开示例复制一份本地配置。`config/settings.yaml` 已被 Git 忽略，可以安全地保存个人股票池和模拟盘参数；六位代码必须使用引号，避免 YAML 把前导零丢掉。

```bash
cp config/settings.example.yaml config/settings.yaml
```

然后执行初始化命令：

```bash
# 下载配置股票池的未复权、前复权行情和沪深300基准
aquant --config config/settings.yaml data update

# 首次同步全部 A 股代码；之后轮换更新前复权研究行情
aquant --config config/settings.yaml universe sync
aquant --config config/settings.yaml universe status
aquant --config config/settings.yaml universe update --batch-size 200

# 更新行业分类；全量源不可用时会逐股补全本批 200 只
aquant --config config/settings.yaml industry update --batch-size 200
aquant --config config/settings.yaml industry status

# 检查依赖、数据库、行情、股票宇宙和因子快照
aquant --config config/settings.yaml doctor

# 立即执行一次选股；首次建议刷新技术预选股的基本面
aquant --config config/settings.yaml screen --force --refresh-fundamentals

# 执行并保存组合回测
aquant --config config/settings.yaml backtest --strategy ma_cross

# 用连续时间分段比较均线参数的稳健性
aquant --config config/settings.yaml backtest --ma-robustness

# 在完全相同的数据、费用与成交规则下对比全部内置策略
aquant --config config/settings.yaml backtest --compare-all

# 启动网页
aquant --config config/settings.yaml app

# 在终端查看后台任务；--show 可查看单个任务及最近日志
aquant --config config/settings.yaml jobs
aquant --config config/settings.yaml jobs --show <任务编号>
```

网页默认由 Streamlit 打开。普通页面只读取本地缓存；只有明确点击同步、更新、筛选时勾选基本面更新，或运行每日流程才会访问网络。

## 网页后台任务

网页中的“更新全部行情、同步A股代码、轮换更新、行业补全、筛选、回测、策略对比、稳健性分析、每日模拟盘”都会提交给独立 Python 后台进程，按钮不会再占用 Streamlit 页面线程。提交成功后可以切换功能页、刷新、关闭浏览器标签，甚至重启网页进程，已经启动的任务仍会继续。

任务状态、参数快照、进度、进程号和结果引用保存在 SQLite 的 `background_jobs` 表，标准输出和异常堆栈保存在 `logs/jobs/<job_id>.log`，任务摘要保存在 `output/jobs/<job_id>/result.json`。相关页面每3秒刷新任务状态；离开后重新进入也会从数据库恢复，不依赖浏览器 `session_state`。

为保证行情文件、因子快照和模拟盘账户一致，所有网页耗时任务仍按全局队列串行执行，同一时间只运行一个；当前任务运行时，可以从其他页面继续提交不同类型的任务，它们会按提交顺序自动排队。相同类型的任务在运行或排队时不能重复提交。后台进程异常退出后，等待进程、再次打开页面或运行 `aquant jobs` 都会检测失联心跳，释放运行锁并让下一项任务继续。

## 内置策略

| 配置名称 | 方向 | 默认规则 |
|---|---|---|
| `ma_cross` | 趋势跟踪 | 5/20日均线价差越过0.3%缓冲带并连续确认2个交易日后切换持仓 |
| `trend_breakout` | 趋势突破 | 均线或20日新高入场，以均线转弱或3倍ATR移动止损退出 |
| `momentum` | 横截面动量 | 综合20/60/120日收益排名，持有正动量前列股票 |
| `rsi_reversion` | 均值回归 | RSI(14) 不高于30时入场，不低于55时退出 |
| `mean_reversion` | 超跌反弹 | RSI、布林带下轨或5日跌幅触发，RSI/中轨修复后退出 |
| `ensemble_equal` | 多策略组合 | 趋势、动量、均值回归等权，综合分数达到0.5时持有 |
| `ensemble_risk` | 多策略组合 | 按子策略已实现收益的20日逆波动率加权，样本不足退回等权 |

所有指标只使用信号日收盘前已经出现的数据，订单统一延迟到下一条可交易日线开盘撮合。横截面动量必须一次读取完整股票池，不能逐只股票单独计算。风险组合中的权重是信号层权重，用于降低历史波动较高的子策略影响，并不代表真实基金产品的独立策略账户。

均线策略默认使用 `strategy.ma_signal_buffer: 0.003` 和 `strategy.ma_confirmation_days: 2`。短均线高出长均线0.3%以上并连续满足2个交易日才确认入场；持仓后，只有短均线低于长均线0.3%以上并连续满足2日才确认退出；价差位于正负0.3%的缓冲区时保持原状态。这个迟滞机制用于过滤均线附近的短暂反复穿越，通常能减少换手和交易成本，但也会让趋势入场、退出稍有延迟。

`config/settings.yaml` 中的 `strategy.name` 决定每日模拟盘使用哪一种策略。网页回测页可以临时选择其他策略或对比全部策略，不会自动改变每日模拟盘配置。

## 交易频率控制

回测和每日模拟盘共用以下执行参数，网页回测页也可以临时修改：

| 配置项 | 默认值 | 含义 |
|---|---:|---|
| `strategy.momentum_rank_buffer` | 2 | 已持有的动量股票只要仍为正动量且排名未跌出 `前 N + 2`，就继续持有；只影响动量策略及组合策略中的动量组件 |
| `portfolio.min_holding_days` | 3 | 普通卖出至少等待买入后的第 3 个交易日，覆盖 T+1 并减少一天内或隔日反复切换 |
| `portfolio.minimum_rebalance_weight_change` | 0.02 | 目标权重与当前权重相差不足 2 个百分点时忽略该次普通调仓 |
| `portfolio.max_daily_turnover` | 0.20 | 普通调仓每天最多使用组合权益 20% 的成交预算，多个买入目标按缺口同比例分批完成 |

完整退出和减仓优先占用换手预算；股票池移除及组合回撤熔断可以越过普通换手和最短持有约束，避免风险仓位被频率限制困住。所有限制都在生成目标订单之前应用，100 股取整、现金和费用检查仍在实际撮合时再次执行。

## 自动选股

默认配置 `screening.universe_source: all_a`。系统先同步约五千只 A 股代码，再由 `universe update` 按 SQLite 中的游标轮换下载前复权研究行情；任务中断后下次从原游标继续。筛选只读取已经完整落盘的前复权文件，因此覆盖率会随每日批次逐渐增加，不会因一只股票下载失败破坏已有缓存。未复权行情只需为模拟盘股票池维护；确需全市场同时下载时可添加 `--include-raw`。

筛选流程如下：

1. 排除 ST/退市标记、历史不足、成交额不足和数据陈旧股票。
2. 用 20/60/120 日动量从本地可用股票中做技术预选。
3. 对预选股读取或更新估值、ROE、利润增长和负债率；行业从独立全市场分类缓存读取。
4. 对动量、价值、质量、低波、流动性因子去极值并在行业内标准化；缺失因子自动按剩余权重归一。
5. 按总分排序，并应用最终数量和单行业入选上限。

结果写入 `output/screens/<run_id>/` 和 SQLite。`screening.schedule` 支持 `daily`、`weekly`、`monthly`；每日任务到期时会自动执行筛选并刷新技术预选股基本面。网页“自动选股”页可以查看覆盖告警、结果和导出文件，也可以从本次入选结果中选择一只或多只股票加入股票池；已有代码会自动排除。新加入的股票需要再到“数据与股票池”页面更新未复权行情，之后才能安全用于每日模拟盘。

“未分类”不表示该公司没有行业，而表示本地还没有取得可用分类。系统会先尝试一次下载全部股票的申万分类；若该来源暂时不可用，再按 `screening.industry_batch_size` 通过巨潮接口逐股回填。自动选股页会分别显示“无记录、空响应、接口错误”，后台任务离开页面后仍会继续。对持续失败或需要自定义口径的股票，可在页面手工修正，或运行：

```bash
aquant --config config/settings.yaml industry set 600000 银行 --note "人工复核"
aquant --config config/settings.yaml industry remove 600000
```

筛选时的优先级为“手工修正 > 独立数据源分类 > 旧因子快照 > 未分类”。接口失败只更新错误状态，不会抹掉上一次有效行业。为避免混用“半导体、电子设备制造业”等不同层级，申万记录统一映射为一级行业；只有申万缺失时才使用带来源标记的回退口径。

## 每日模拟盘

收盘后运行：

```bash
aquant --config config/settings.yaml daily-run
```

顺序固定为：更新自选股行情、处理以前的待成交订单、检查数据新鲜度、计算当天信号、应用组合配置与回撤熔断、生成下一交易日目标订单、保存组合快照，并在筛选周期到期时执行自动选股。相同数据重复运行不会重复创建信号、订单、成交，也不会重复扣减回撤冷静期。

若当天已经单独更新过行情，可使用：

```bash
aquant --config config/settings.yaml daily-run --skip-update
```

任何股票缺少行情、数据过旧或本次更新回退到缓存时，新订单都会被阻止；已有待成交订单仍按本地后续行情尝试模拟撮合。

### 将回测策略晋级到模拟盘

研究页选择策略和参数只影响该次回测，不会自动改写模拟盘。完成并保存回测后，在“回测研究 → 模拟盘策略晋级”中选择“已保存回测”，复核指标和参数差异，勾选确认后才能提交晋级。系统会校验回测编号、引擎版本、参数指纹及全部落盘文件校验和，并将来源回测、数据截止日、完整源参数、指标、备注和启停时间保存到 SQLite。

模拟盘拥有独立部署策略；`settings.yaml` 继续作为研究默认值。晋级只复制 `StrategyConfig`，不会重置现金、持仓或风险冷静期，也不会复制回测当时的组合风控和交易成本。存在运行或排队中的后台任务时禁止提交晋级。

若旧策略还有待成交订单，新部署先进入“待生效”：每日流程只继续处理旧订单，不再生成新的旧策略订单；旧订单全部成交、跳过或拒绝后，新部署会在生成下一批信号前自动启用。信号和订单使用带部署编号的执行键，因此同一种策略更换参数后不会与旧记录的幂等键冲突。当前部署、待生效状态和完整部署历史可在“模拟盘”页面查看。

## 回测评估与稳健性分析

回测除绝对收益、波动率和回撤外，还会相对沪深300计算超额总收益与年化超额收益、跟踪误差、信息比率、Beta、年化 Alpha 和相关性。买入持有曲线用于区分“股票池本身上涨”与“择时带来的增减益”。费前曲线固定使用费后回测已经发生的成交日期与数量，只加回佣金、印花税和滑点，因此与费后曲线的差值可以逐笔核对累计成本，不会因为重新分配资金或触发不同风控路径而混入第二套交易结果。

年度归因按自然年列出策略费后、费前、股票池买入持有和沪深300收益，以及相对基准收益、平均仓位、换手率、成交笔数、成交额和交易成本。滚动分析默认使用252个交易日窗口，展示滚动策略收益、基准收益、超额收益、跟踪误差、信息比率和 Beta。

运行以下命令会在相同股票池和成交规则下，将多组均线周期、缓冲带和确认天数组合放入连续的历史时间段中比较：

```bash
aquant --config config/settings.yaml backtest --ma-robustness
```

稳健性排名优先考虑跑赢基准的分段数量、年化超额收益下四分位、最差分段年化超额、信息比率和换手率。下四分位同时反映典型区间与较差区间，避免仅凭单个最差折的微小差距决定排名。分析只提供参数敏感性证据，不会把排名第一的参数自动写回 `config/settings.yaml`。当前股票池可能包含事后才选择的股票，连续时间分段也仍复用了同一股票池、同一研究过程，因此结果不是真正的样本外检验；正式采用参数前仍应使用未参与开发的数据、滚动训练/验证和足够长的模拟盘观察。

分段边界取所有股票与沪深300的共同可用日期，并按基准交易日数量尽量等分；排名中的最差超额使用年化超额收益，以免长短不一的区间累计收益不可比。日线回测在下一交易日有效开盘价撮合；`volume == 0` 仅作为停牌或全天无成交的代理，不读取正成交量大小或当天最高、最低价来决定开盘订单是否成交。

## 定时运行（macOS）

提供两个任务：16:20 轮换更新 200 只全市场前复权行情，18:00 执行自选股模拟盘并按周期筛选。复制两个 plist，将其中占位路径替换为本机项目绝对路径和虚拟环境中的 `aquant` 路径，然后执行：

```bash
cp scripts/com.aquant.daily.plist.example ~/Library/LaunchAgents/com.aquant.daily.plist
cp scripts/com.aquant.universe.plist.example ~/Library/LaunchAgents/com.aquant.universe.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.aquant.universe.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.aquant.daily.plist
```

两个示例任务均在周一至周五运行。法定休市日仍可能唤醒，但幂等和新鲜度检查会避免重复订单。首次全市场历史回填可手动提高到 `--batch-size 500`，但免费接口可能限流；日常建议保持 200。卸载任务使用：

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.aquant.daily.plist
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.aquant.universe.plist
```

## 数据与结果

- `data/raw/`：未复权股票行情。
- `data/adjusted/`：前复权研究行情。
- `data/benchmark/`：指数基准。
- `data/universe.parquet`：全 A 股代码、名称与 ST 标记。
- `data/factors/latest.parquet`：基本面最近有效快照（旧文件中的行业列仅作兼容回退）。
- `state/aquant.sqlite3`：信号、订单、成交、持仓、行业分类/手工修正、后台任务和运行状态。
- `output/backtests/<run_id>/`：净值、信号、成交、参数和指标；其中 `annual_returns.csv` 是年度收益与交易归因，`rolling_relative.parquet` 是252日滚动相对基准指标，`manifest.json` 记录回测实现版本及每个结果文件的 SHA-256 校验值。整组文件先写临时目录再原子替换，避免中断后留下新旧混合结果。
- `output/robustness/<analysis_id>/`：均线参数连续时间分段的排名摘要 `summary.csv`、逐分段结果 `folds.csv` 和包含完整配置、分段及底层回测编号的 `manifest.json`。
- `output/comparisons/<comparison_id>/`：七策略指标表、净值对比和对应运行编号。
- `output/screens/<run_id>/`：全量评分、入选结果与筛选摘要。
- `output/paper/<YYYYMMDD>/`：当日模拟盘摘要、持仓、订单、成交、信号和权益快照。
- `output/jobs/<job_id>/`：网页后台任务的持久化摘要和质量报告。
- `output/backups/`：每日任务运行前的 SQLite 备份。
- `logs/aquant.log`：主日志，最多保留五个轮转文件。
- `logs/jobs/<job_id>.log`：每个独立后台任务的进度、标准输出和异常堆栈。

运行数据默认不纳入 Git。修改股票池、初始资金、费用、滑点和仓位上限时，只编辑 `config/settings.yaml`；示例和字段说明见 `config/settings.example.yaml`。

仓库不附带任何股票行情、因子快照、模拟盘账户或回测结果。首次克隆后需要复制示例配置并自行下载数据。

## 项目结构

```text
config/                 公开配置示例（个人 settings.yaml 不入库）
scripts/                macOS launchd 定时任务示例
src/aquant/             数据、策略、回测、模拟盘、网页和命令行模块
tests/                  不依赖网络的单元与集成测试
data/ state/ output/    本地行情、运行状态和研究结果（均不入库）
```

## 回测口径

- T 日收盘信号只允许在 T+1 的第一条可交易日线开盘成交。
- 买入向上、卖出向下施加滑点；卖出另收印花税。
- 买入数量向下取整到 100 股，先卖后买，不使用杠杆或做空。
- 普通卖出遵守最短持有期；小于最小权重差异的调仓被忽略，较大的建仓按每日换手预算分批执行。
- 一字行情、零成交量和停牌日期不撮合；缺失标的继续等待下一条可交易行情。
- 前复权数据用于长期收益研究，未复权数据用于模拟盘现金和持仓估值。免费日线无法还原盘口排队，结果不等同于真实可成交收益。
- 组合默认逆波动率配置，并按目标年化波动缩放总仓位；权益触及最大回撤阈值后，在配置的交易日冷静期内目标仓位归零。

## 测试

```bash
PYTHONPATH=src pytest
ruff check src tests
```

测试使用固定的三标的合成行情，不依赖网络，覆盖信号延迟、排名缓冲、最短持有、调仓阈值、换手预算、费用、交易单位、幂等、模拟成交、数据校验和完整结果落盘。
