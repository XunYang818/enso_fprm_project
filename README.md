# ENSO 缺失数据恢复代码

本项目用 Python 复现 Wu 等人的 Full-Partial Reconstruction Mapping（FPRM），并实现固定多尺度延迟嵌入与可观测标签交叉验证加权。交付内容是代码、配置、测试和运行说明；不包含实验报告。原版是 **Python 方法复现**，不是 MATLAB 数值逐点复刻。来源、参数与差异见 [REPRODUCTION.md](REPRODUCTION.md)。

2026-10-08 已在指定 Miniforge 环境验证：24 项测试通过；`validation.yaml` 的 48 个真实数据案例、318 条方法结果全部成功；相同配置续跑复用了全部 48 个案例。正式 20 种子实验也已完成，共 960 个案例、6360 条方法结果，零方法失败；逐项复算指标及输出文件校验均通过。常数预测的相关系数按未定义保存，核参数边界警告保留在参数文件。运行产物位于工作区的 `codex_proc/`，不纳入代码仓库。

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

三视图均只按目标标量标签的可见性训练，标准化只用对应训练折。GPR 为共享长度尺度的 `Constant * RBF + White` 核，不使用 ARD。每次拟合固定种子，以 L-BFGS-B 优化并额外重启两次；分解失败才逐次增大数值 jitter。有限预测对应的优化警告会保留，失败不会静默替换为其他算法。

主比较统一前 410 个月，参考序列保留全部 422 个月；末 12 个月不恢复、不评价。原版复现检查单独使用 416 个月。模型使用参考变量的未来观测，因此适用于**离线缺失恢复**，不能作为实时预测效果。

默认随机缺失率为 10%、30%、50%、70%、90%，种子 0–19。随机掩码在每个目标/种子下嵌套；共同目标切换参考时共享掩码。连续缺失是一段随机位置区间，可触及首尾，允许重复。模型收到的目标在所有隐藏位置均为 NaN，完整真值仅由评价层保存。

`configs/default.yaml` 为正式配置；`configs/validation.yaml` 使用相同算法参数但仅一个种子。可修改率、种子、实验组和 GP 设置；若改变研究设置，应把结果作为新实验，不与默认范围混合。

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
      checks.json           完成数量、失败数量、未定义指标数量
      figures/              精度、相关、耗时、消融和恢复曲线
```

续跑时使用运行打印的真实路径，保持同一配置和 suite：

```powershell
.\run.ps1 -Suite full -Resume 'D:\实际工作区\codex_proc\enso_fprm\runs\实际运行目录'
```

数据、配置、代码、环境、线程设定或案例上限任何一项不一致时，拒绝复用；另开新运行即可。只有全部方法成功且文件校验通过的案例可复用。失败或未完成案例写入新的 `attempt` 目录；汇总输出也另建目录，不删除旧尝试。`-LimitCases 5` 仅供开发截取前几个案例，且被写入指纹，不能当作完整实验。

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
