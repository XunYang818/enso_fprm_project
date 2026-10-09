# 基于多尺度融合的 ENSO 数据恢复

本项目用 Python 复现 Full-Partial Reconstruction Mapping（FPRM），并实现多尺度等权融合和交叉验证加权融合。实验报告位于项目上一级目录，正式实验结果位于 `results/`。

## 文件结构

```text
enso_fprm_project/
  enso_fprm/                 算法、实验、评价与绘图代码
  configs/default.yaml       正式配置，20 个种子
  configs/validation.yaml    验证配置，1 个种子
  tests/                     正确性、防泄漏与续跑测试
  data/                      原始数据、作者源码及来源记录
  results/                   报告对应的结果、图表和版本记录
  environment.yml            Conda 环境配置
  requirements-lock.txt      主要 Python 依赖版本
  pyproject.toml             Python 项目及测试配置
  run.ps1                    可选运行入口
```

## 安装与运行

在 Miniforge Prompt 中进入项目目录。已验证环境为 Python 3.11，环境名为 `enso-fprm`。已有该环境时可跳过创建步骤。

```shell
conda env create -f ./environment.yml
```

若希望安装报告使用的主要 Python 库版本，可在创建环境后执行：

```shell
conda run -n enso-fprm python -m pip install -r ./requirements-lock.txt
```

版本清单仅固定直接依赖；原实验的环境版本见 `results/manifest.json`。不同平台、库版本或硬件下的数值和耗时可能有差异。

在项目目录执行以下命令：

```shell
# 校验随包数据及作者源码；文件缺失时才下载
conda run --no-capture-output -n enso-fprm python -B -m enso_fprm --prepare

# 冒烟实验：一个目标、50% 随机缺失、一个种子，含八种方法及图表
conda run --no-capture-output -n enso-fprm python -B -m enso_fprm --suite smoke

# 验证全部实验组，每组一个种子
conda run --no-capture-output -n enso-fprm python -B -m enso_fprm --suite full --config ./configs/validation.yaml

# 正式实验：20 个种子，960 个案例、6360 条方法结果
conda run --no-capture-output -n enso-fprm python -B -m enso_fprm --suite full
```

正式实验耗时明显长于冒烟实验，建议先运行 `smoke`。原始数据已随包提供，校验通过时实验无需下载数据；环境安装需要获取依赖。

测试命令中的临时目录应使用一个尚不存在的名称：

```shell
conda run --no-capture-output -n enso-fprm python -B -m pytest --basetemp ./outputs/tests/check_01
```

`run.ps1` 也提供 `-Mode prepare`、`-Mode test`、`-Suite smoke`、`-Suite full` 入口，自动调用当前终端可用的 Conda，并为每次测试选择独立临时目录。

## 数据与输出

程序从 `data/` 读取原始文件，新运行结果写入 `outputs/runs/`，缓存和测试临时文件写入 `outputs/` 下的相应子目录。输出目录会自动创建。`results/` 保存报告对应的固定结果，新运行不会覆盖这些文件。

每次运行会打印 `RUN_DIR` 和 `EXPORT_DIR`。运行目录保存配置、环境、掩码、预测与参数；导出目录保存指标表、统计结果及 `figures/` 图表。

续跑时，将下面的 `运行目录名` 替换为实际生成的目录名：

```shell
conda run --no-capture-output -n enso-fprm python -B -m enso_fprm --suite full --resume ./outputs/runs/运行目录名
```

续跑核对数据、配置、代码和环境，并验证逐案例结果；不一致时应另开新运行。随包 `results/` 不含逐案例缓存，不能直接续跑。`--limit-cases` 仅供开发检查，不代表完整实验。

数据来自作者 Figshare v3 的 `ENSO data.xlsx`，选取 1990-01 至 2025-02 的 422 个月，变量为 NINO4、NINO34、NINO3、NINO12。程序核验 MD5、SHA256、日期、表头和数值，不自动填补原始异常。下载地址和校验值见 `data/sources.json`。作者 MATLAB 文件作为来源归档，运行本项目不需要 MATLAB。

## 实验方法

| 方法 | 实现 |
|---|---|
| `mean` | 可观测目标均值填充 |
| `linear` | 月份索引线性插值，边界取最近观测 |
| `gpr_raw` | 同时刻参考值输入 GPR |
| `fprm` | E=3、tau=3 的前向嵌入 GPR |
| `fprm_tau1` / `fprm_tau6` | 单尺度消融 |
| `multiscale_equal` | tau=1、3、6 三视图等权融合 |
| `multiscale_weighted` | 可观测标签按时间分三折，以折外 MSE 倒数加权 |

默认参考为 NINO34，另用 NINO3 检查参考切换。随机缺失率为 10%、30%、50%、70%、90%，种子为 0–19；实验还包括连续区间缺失及原版复现检查。主比较评价前 410 个月，原版复现检查使用 416 个月，参考保留全部 422 个月。

标准化和权重估计只使用可观测训练标签，隐藏真值仅用于评价。仅在隐藏点计算 RMSE、MAE、Pearson rho 和 `NRMSE = RMSE / std(隐藏真值, ddof=1)`。常数预测的 rho 保存为空并记录原因。整体 NRMSE 先在同一种子内平均目标，再在种子间汇总；配对 bootstrap 以种子为单位抽样。

前向嵌入使用参考的未来观测，适用于离线缺失恢复；统计区间反映固定 ENSO 数据上人工缺失位置的变化。

## 正式结果

`results/` 对应 2026-10-09 正式运行：960 个案例、6360 条方法结果、零失败。

| 文件 | 内容 |
|---|---|
| `metrics.csv` | 逐案例、逐方法指标及分阶段耗时 |
| `summary_by_target.csv` | 各目标跨种子的均值、样本标准差与有效数量 |
| `overall_by_seed.csv` / `summary_overall.csv` | 逐种子整体 NRMSE 及汇总 |
| `paired_statistics.csv` | 配对差、胜率及 bootstrap 区间 |
| `weights.csv` | 多尺度权重及折外 MSE |
| `checks.json` | 案例与方法数量、失败及未定义指标计数 |
| `figures/` | 精度、相关系数、耗时、消融及恢复曲线 |
| `manifest.json` / `config_used.yaml` | 正式运行配置、环境和代码指纹 |
| `run_status.json` | 正式运行完成情况 |
| `provenance.json` | 提交代码与正式运行的对应关系及文件校验值 |

提交版调整了数据、输出和缓存目录及运行入口，算法、掩码、评价、统计和实验配置保持一致。正式运行记录中的本机路径已改为相对路径，并保留原运行签名及整理说明；它们用于追溯，不作为续跑缓存。

## 复现依据与来源

Wu, T., Gao, X., Tang, Y., et al. *Dynamics-informed machine learning for recovering extensive missing systems dynamics*. Nature Communications (2026)，[DOI: 10.1038/s41467-026-77922-1](https://doi.org/10.1038/s41467-026-77922-1)。

- [论文 PDF](https://www.nature.com/articles/s41467-026-77922-1_reference.pdf)：Methods 式 (13)–(18) 对应从完整参考嵌入到目标标量的 GPR 映射。
- [补充材料](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-026-77922-1/MediaObjects/41467_2026_77922_MOESM1_ESM.pdf)：Fig. S9 对应 ENSO 范围与参考切换；Table S1 给出 SST 的 E=3、tau=3。作者通用示例为 E=5、tau=3，本项目采用 SST 参数。
- [作者数据与代码 v3](https://doi.org/10.6084/m9.figshare.30446765.v3)：随包保留原始数据、`PhaSpaRecon.m` 和 `FPRM mian code.mlx`，采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

`embedding.py` 移植作者的前向延迟坐标，按行存放样本并返回原月份索引。`models.py` 参考作者主脚本，只要求目标标量标签可见，不要求整个目标延迟窗口完整。

Python 实现按训练行的均值、样本标准差（`ddof=1`）标准化特征与目标，常数列除数设为 1，预测后还原目标单位。作者脚本未手工标准化目标。核为 `ConstantKernel(0.5, [1e-4, 1e4]) * RBF(1, [1e-2, 1e2]) + WhiteKernel(0.01, [1e-8, 1e1])`，共享一个长度尺度，`normalize_y=False`，使用 L-BFGS-B 并额外重启两次。jitter 从 `1e-8` 起，仅在矩阵分解失败时尝试 `1e-7`、`1e-6`；失败及优化警告显式记录。

作者采用 MATLAB R2024b 的 `fitrgp`，其基础均值、初始化和优化行为与 scikit-learn 不同，本项目复现方法结构，不宣称逐点数值一致。作者的随机种子改为固定种子派生；缺失率采用嵌套掩码，共同目标切换参考时共享掩码。

改进方法独立拟合 tau=1、3、6 三个视图，各验证折重新计算标准化和核参数，以全部折外预测的原单位 MSE 确定权重：

```text
epsilon = 1e-8 * max(1, var(y_observed, ddof=1))
w_j = (1 / (MSE_j + epsilon)) / sum_k(1 / (MSE_k + epsilon))
```

最终视图使用全部可观测标签拟合，其预测不参与折外误差与权重估计。等权融合和单尺度方法用于消融。三折是离线恢复的标签划分，不能解释为滚动时间预测。

前向嵌入与原版 FPRM 流程改编自 Tao Wu 等人发布的代码，相关改编按 CC BY 4.0 保留归属。本项目增加了输入校验、可重复实验、Python GPR 参数约定、多尺度融合、防泄漏测试和统计输出。再发布时请保留本节来源、许可与改编说明。
