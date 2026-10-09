# ENSO 缺失数据恢复代码

本项目用 Python 复现 Wu 等人的 Full-Partial Reconstruction Mapping（FPRM），并实现固定多尺度延迟嵌入与可观测标签交叉验证加权。交付内容是代码、配置、测试和运行说明；不包含实验报告。原版是 **Python 方法复现**，不是 MATLAB 数值逐点复刻。来源、参数与差异见 [REPRODUCTION.md](REPRODUCTION.md)。

2026-10-09 已使用当前交付代码完整重跑正式 20 种子实验：960 个案例、6360 条方法结果，零方法失败。保存的预测结果已逐项复算核验，运行清单中的模型源码指纹与当前代码一致。实验报告和正式结果保留在本地工作区，不纳入代码仓库；完整运行记录与缓存位于上级工作区的 `codex_proc/`。

2026-10-08 当前版本已在指定 Miniforge 环境验证：139 项测试通过；默认冒烟实验的八种方法全部成功，包含图表生成。续跑除检查方法与字段完整性外，还从预测文件复算指标并与参数文件核对融合权重。指标数值错误或权重缺失的记录均在真实续跑中被跳过，补算写入新尝试，再次续跑正确复用。无符号整数索引已验证与有符号索引等价，乱序与越界索引会明确拒绝。默认冒烟实验的八种方法预测与修复前完全一致，融合权重也一致；此前正式实验的 960 个完整案例记录全部通过新校验。旧记录格式兼容，续跑仍要求运行身份一致。

此前版本已完成 `validation.yaml` 的 48 个真实数据案例、318 条方法结果，以及正式 20 种子实验的 960 个案例、6360 条方法结果，零方法失败；逐项复算指标及输出文件校验均通过。旧结果保留，修复后代码指纹发生变化，新的实验应使用新的运行目录。常数预测的相关系数按未定义保存，核参数边界警告保留在参数文件。运行产物位于工作区的 `codex_proc/`，不纳入代码仓库。

## 快速运行

在 PowerShell 中进入本项目目录。已验证的环境为 `D:\miniforge3\envs\enso-fprm`，Python 3.11。

```powershell
# 校验数据和下载原作者源码（已有校验通过的文件会直接复用）
.\run.ps1 -Mode prepare

# 运行测试，临时目录自动放在上级工作区的 codex_proc
.\run.ps1 -Mode test

# 一个目标、50% 随机缺失、一个种子的完整流程，含图表
.\run.ps1 -Suite smoke

# 每组一个种子，验证所有缺失率、缺失模式、参考变量和评价范围
.\run.ps1 -Suite full -Config .\configs\validation.yaml

# 正式实验：每组 20 个种子
.\run.ps1 -Suite full
```

若运行脚本被本机 PowerShell 执行策略阻止，可以按下面的直接命令执行，无需修改策略。Miniforge 位置可用 `-CondaExe` 指定。

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
& 'D:\miniforge3\Scripts\conda.exe' run --no-capture-output -n enso-fprm python -B -m enso_fprm --suite smoke --config .\configs\default.yaml
```

每次运行打印 `RUN_DIR` 和 `EXPORT_DIR`。`full` 在默认配置下包含 960 个目标/掩码案例、6360 条方法结果，其中六个主比较方法和两个单尺度消融复用最终视图拟合。加权方法每个案例另有 9 次折内拟合，因此完整实验耗时明显长于冒烟运行。以本机实际运行时间为准。

## 环境恢复

现有环境可直接使用，不在 base 中安装依赖。如果在另一台机器新建环境：

```powershell
& 'D:\miniforge3\Scripts\conda.exe' env create -f .\environment.yml
```

`environment.yml` 为跨平台依赖清单；`requirements-lock.txt` 记录本次实现环境的直接依赖版本。`conda-win-64.lock.txt` 为本机 Conda 包的精确快照，适用于 Windows x64：

```powershell
& 'D:\miniforge3\Scripts\conda.exe' create -n enso-fprm-replay --file .\conda-win-64.lock.txt
```

后者环境名与运行脚本默认名不同，可将其改为 `enso-fprm` 或用 `conda run -n enso-fprm-replay` 直接运行。运行时以 `threadpoolctl` 强制数值库单线程，记录实际 Python、库版本、平台、CPU 信息和线程池。

## 实验与算法

数据固定为 Figshare v3 的 `ENSO data.xlsx`。提取 1990-01 至 2025-02 的 422 个连续月份，变量为 NINO4、NINO34、NINO3、NINO12。下载前后均核验 MD5 和 SHA256；日期、表头与原始数值异常时停止，不删行、不自动填补。该版本表头将 `Month` 拆为 `Mo` 与 `nth NINO4`，代码接受这个明确列出的表头变体，运行清单记录映射。

| 方法名 | 实现 |
|---|---|
| `mean` | 当前可观测目标均值 |
| `linear` | 原月份索引线性插值；边界取最近观测 |
| `gpr_raw` | 单个同时刻参考值输入 GPR |
| `fprm` | E=3、tau=3 的前向嵌入 GPR |
| `multiscale_equal` | tau=1、3、6 的三视图等权融合 |
| `multiscale_weighted` | 可观测标签按时间均分三折，以折外 MSE 的倒数确定权重 |
| `fprm_tau1` / `fprm_tau6` | 单尺度消融，另一个单尺度就是 `fprm` |

表中为默认延迟的命名。修改 `gp.delays` 后，会为每个非原版延迟自动生成 `fprm_tauN` 消融方法，例如 `[2, 3, 9]` 对应 `fprm_tau2`、`fprm_tau9`；原版延迟仍由 `fprm` 表示。图表的消融曲线和延迟标签随配置变化。冒烟实验选择与参考变量不同的第一个目标；默认仍为 NINO4。

三视图均只按目标标量标签的可见性训练，标准化只用对应训练折。GPR 为共享长度尺度的 `Constant * RBF + White` 核，不使用 ARD。每次拟合固定种子，以 L-BFGS-B 优化并额外重启两次；分解失败才逐次增大数值 jitter。有限预测对应的优化警告会保留，失败不会静默替换为其他算法。

主比较统一前 410 个月，参考序列保留全部 422 个月；末 12 个月不恢复、不评价。原版复现检查单独使用 416 个月。模型使用参考变量的未来观测，因此适用于**离线缺失恢复**，不能作为实时预测效果。

默认随机缺失率为 10%、30%、50%、70%、90%，种子 0–19。随机掩码在每个目标/种子下嵌套；共同目标切换参考时共享掩码。连续缺失是一段随机位置区间，可触及首尾，允许重复。模型收到的目标在所有隐藏位置均为 NaN，完整真值仅由评价层保存。

`configs/default.yaml` 为正式配置；`configs/validation.yaml` 使用相同算法参数但仅一个种子。可修改率、种子、实验组和 GP 设置；若改变研究设置，应把结果作为新实验，不与默认范围混合。`bootstrap_iterations` 必须为正整数，各种子必须为非负整数，GP 的维度、延迟、折数与重启次数必须使用对应范围内的整数；浮点数或布尔值不作为整数接受。配置加载及直接调用 `run` 都会在数据准备和训练前拒绝这些错误。

生成实验计划时还会按实际 suite、缺失率及请求方法检查有效长度、隐藏数、可见标签数，以及加权方法每折是否至少有两个训练标签。不兼容的配置在数据准备和创建运行目录前拒绝，错误信息包含实验组、目标、维度、所用延迟和有效长度。例如 `E=3、delays=[1,3,211]` 的冒烟实验会明确指出 `n=0`。仅运行原版复现时只检查原版延迟，不受未使用的多尺度延迟、配置缺失率或验证折数限制；冒烟使用其固定的 50% 缺失率。

## 输出位置与续跑

所有数据、下载源码、缓存、日志、测试临时文件和运行结果均在**本项目上级工作区**的 `codex_proc/enso_fprm/`。项目目录只有最终代码和配置。脚本不删除或移动已有文件。

```text
codex_proc/enso_fprm/
  data/                     原始数据、作者源码、校验信息
  cache/                    matplotlib 缓存
  tests/                    各次测试的独立临时目录
  runs/<独立运行目录>/
    manifest.json           数据/配置/代码/环境指纹、表头映射
    config_used.yaml        实际配置
    enso_selected.csv       422x4 数据选择
    cases/<案例>/attempt_*/
      mask.csv              原索引、日期、隐藏位置
      predictions.csv       真值、模型可见输入、各方法恢复值
      parameters.json       核参数、折内标准化、权重、警告、耗时
      completed.json        案例结果及输出文件 SHA256
    exports_*/
      metrics.csv           逐次指标与分阶段时间
      summary_by_target.csv 各目标种子均值、样本标准差、有效数量
      overall_by_seed.csv   每种子先平均目标 NRMSE
      summary_overall.csv   再在种子间汇总整体 NRMSE
      paired_statistics.csv 加权减原版的配对差、胜率、bootstrap 区间
      weights.csv           各尺度权重和折外 MSE
      checks.json           预期与实际案例/方法数量、失败数量、未定义指标数量
      figures/              精度、相关、耗时、消融和恢复曲线
```

续跑时使用运行打印的真实路径，保持同一配置和 suite：

```powershell
.\run.ps1 -Suite full -Resume 'D:\实际工作区\codex_proc\enso_fprm\runs\实际运行目录'
```

数据、配置、代码、环境、线程设定或案例上限任何一项不一致时，拒绝复用；另开新运行即可。只有案例身份符合当前计划、每个请求方法恰有一条完整结果、预测表包含全部方法列、全部方法成功且文件校验通过的案例可复用。复用前从预测文件复算隐藏位置的四项指标及未定义原因，并核对掩码、观测数、隐藏数、观测值保持和阶段耗时求和；融合权重必须完整、唯一，并与参数文件的每个延迟、权重和折外 MSE 一致。CSV 浮点数按原值读取，避免常数预测的相关系数被读表舍入改变。损坏、不完整或缺少输出校验信息的完成记录会打印 `CACHE_SKIPPED` 诊断并跳过；仍可复用较早的有效尝试，没有有效尝试时重新计算。最终汇总在创建导出目录前再次检查计划案例、方法结果及输出内容，缺失、重复或与输出文件不一致的结果会明确报错，不能按零失败判为成功。失败或未完成案例写入新的 `attempt` 目录；汇总输出也另建目录，不删除旧尝试。`-LimitCases 5` 仅供开发截取前几个案例，且被写入指纹，不能当作完整实验。

## 指标与解读边界

仅在隐藏位置计算 RMSE、MAE、Pearson rho、`NRMSE = RMSE / std(隐藏真值, ddof=1)`。常数预测的 rho 未定义，保存为空并写原因；不能用 0 代替。一个种子的样本标准差、bootstrap 置信区间也留空。配对 bootstrap 每次整组抽取种子差异，默认 10000 次；三个目标先在同一种子内平均，不作为三个独立重复。

时间分为最终训练、验证与权重、预测和总计。共享视图的成本按实际已执行步骤计入相应方法，不能用缓存命中的时间冒充重新训练成本；各方法时间求和也不等于整个 bundle 的墙钟时间。下载、读表、绘图和预热排除在算法时间之外。各阶段具体元数据保存在 `parameters.json`。

多尺度加权是本项目的改进假设，效果由实验决定，不预设所有缺失模式都会胜出。种子区间只反映同一 ENSO 数据上人工缺失位置的变化，不代表独立气候数据集的泛化不确定性。

## 文件入口

`enso_fprm/data.py` 数据获取与校验；`embedding.py` 作者嵌入函数的 Python 移植；`models.py` 各恢复算法；`experiment.py` 实验、计时与续跑；`statistics.py` 汇总与配对统计；`plots.py` 图表；`tests/test_core.py` 正确性、失败分支、确定性和隐藏真值泄漏验证。

公开模型接口：

```python
from enso_fprm.models import recover, GPSettings
# complete_reference: 完整参考；observed_target: 隐藏位置为 NaN；indices: 原始月份索引。
result = recover(complete_reference, observed_target, indices,
                 method="multiscale_weighted", seed=0, settings=GPSettings())
# result.recovered / result.metadata / result.error / result.total_seconds
```

索引支持 NumPy 有符号及无符号整数数组，必须唯一、严格递增且在有效范围内。范围检查后统一转为平台有符号索引，不改变调用者的数组。

单独调用 `mean`、`linear` 或 `gpr_raw` 可以使用完整参考范围；单尺度方法按自身延迟确定有效索引，支持显式 `method="fprm_tauN"`。同时请求多个方法时，索引必须对所有请求的方法有效。主比较仍由实验编排统一使用前 410 个月。`recover_bundle` 未指定 `methods` 时，自动使用当前配置的主方法和各尺度消融。
