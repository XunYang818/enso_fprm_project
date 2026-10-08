# 论文与作者代码的对应关系

## 固定来源

Wu, T., Gao, X., Tang, Y., et al. **Dynamics-informed machine learning for recovering extensive missing systems dynamics**. Nature Communications (2026). DOI: [10.1038/s41467-026-77922-1](https://doi.org/10.1038/s41467-026-77922-1)。

- [论文 PDF](https://www.nature.com/articles/s41467-026-77922-1_reference.pdf)：Methods，式 (13)–(18)，从参考嵌入到目标第一坐标的标量 GPR 映射。
- [补充材料](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-026-77922-1/MediaObjects/41467_2026_77922_MOESM1_ESM.pdf)：第 8 页 Fig. S9 的 ENSO 区间与参考切换；第 19 页 Table S1 的 SST 参数 E=3、tau=3。
- [作者数据与代码 v3](https://doi.org/10.6084/m9.figshare.30446765.v3)，许可 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。原文件和下载地址、散列在 `enso_fprm/data.py` 和运行后的 `data/sources.json` 中。

| 文件 | 固定文件 ID | SHA256 |
|---|---:|---|
| ENSO data.xlsx | 64540869 | 631f8046ee637c80a2ad28a20d8530bc56b3e5b1c5b613f21a47a3d33deb3e33 |
| PhaSpaRecon.m | 64540899 | c13dbd7dd5dd40f0436f28aaf16d414062660500d1afd14c6bbda151225cad8e |
| FPRM mian code.mlx | 64540893 | f2afaae07de0d2c73474be3eaaefb8a8d0bdaac6ea6de9e0a98b485dd8e9260b |

保留作者原始文件名中的 `mian` 拼写。原 MATLAB 文件仅下载归档，不要求 MATLAB 环境；执行代码使用 Miniforge Python。

## 直接借鉴并移植的步骤

| 作者实现 | 本项目位置与行为 |
|---|---|
| `PhaSpaRecon(s,tau,m)` 各行依次取前向延迟坐标 | `embedding.delay_embedding` 返回其转置，行是样本，并额外返回原始索引 |
| `X=YX`、`Ytrain=Y(idxTrn,1)` | `models.recover_bundle` 只使用完整参考嵌入，目标是同期单个标量标签 |
| `zscore(Xtrain)` 与保存训练均值/标准差 | `models.standardize` 采用样本标准差，每次最终拟合及每个验证折独立计算 |
| `fitrgp(Ztrain,Ytrain)` | `models.fit_predict` 用 scikit-learn 平方指数 GPR 学习映射 |
| `Ztest=(Xtest-tr_mu)./tr_sigma` | 测试输入仅用相应训练拟合得到的特征标准化参数 |
| `loss`、`std(Ytest)`、`corr`、MAE | `metrics.evaluate` 只评价隐藏点，显式返回未定义原因 |

本项目不按整个目标嵌入窗口是否完整来筛选训练行。论文的映射标签与作者脚本均取第一坐标；如果要求所有目标延迟坐标可见，会在 90% 缺失时错误地丢弃大量可用标签。作者 `PhaSpaRecon.m` 是前向嵌入，不能无说明地改为过去时刻的后向窗口。

公开主脚本是通用演示：调用 `PhaSpaRecon(...,3,5)`，即 tau=3、E=5；它不是 SST 的专用配置。本项目对 ENSO 使用补充材料 Table S1 的 E=3、tau=3，并采用 Fig. S9 指定的月份及参考变量。

## 明确的 Python 方法差异

作者使用 MATLAB R2024b 的 `fitrgp` 默认配置。默认基础均值、参数初始化及优化行为与 scikit-learn 不同，不能以本项目数值声称逐点复现 MATLAB。这里保留 FPRM 的方法结构，并明示以下可复现设置：

- 特征按可观测训练行、`ddof=1` 标准化，零范围特征的除数设为 1。
- 目标也用同一训练行的均值、样本标准差标准化，再恢复原单位；常数目标的除数设为 1。这是 Python 选择，作者脚本未手工标准化目标。
- `ConstantKernel(0.5,[1e-4,1e4]) * RBF(1,[1e-2,1e2]) + WhiteKernel(0.01,[1e-8,1e1])`。RBF 共享一个长度尺度。
- `normalize_y=False`，L-BFGS-B，额外重启两次，所有重启种子记录于参数文件。
- `alpha=1e-8` 是额外数值 jitter，与可学习白噪声不同；仅在矩阵分解失败时顺序尝试 1e-7、1e-6。全失败显式记录，无替代算法。
- 警告和核边界警告原文保存在 JSON 中；有有限预测不等于优化器成功收敛。
- 作者 `rng('shuffle')` 改为 `SeedSequence` 派生固定种子，随机 Holdout 改为共享掩码、嵌套缺失率。
- 主比较限制到 410 个月，另设 416 个月原版检查，避免不同延迟的有效索引不一致。

参考 [scikit-learn GPR 文档](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)和 [MATLAB fitrgp 文档](https://www.mathworks.com/help/stats/fitrgp.html)。论文式 (19) 的 RMSE 标注为归一化指标；本项目分别报告未归一化 RMSE 与作者代码明确使用的隐藏真值样本标准差归一化 NRMSE。

## 本项目的改进

固定 E=3、tau∈{1,3,6}，独立训练三视图 GPR。可见标签按原始时间顺序均分三折，轮流验证，各折重新拟合标准化和核参数。每个视图汇总所有折外预测的原单位 MSE：

```text
epsilon = 1e-8 * max(1, var(y_observed, ddof=1))
w_j = (1 / (MSE_j + epsilon)) / sum_k(1 / (MSE_k + epsilon))
```

全部可见标签再拟合各视图并以确定的权重融合。代码为复用最终视图，可先执行全部标签的最终拟合；这些拟合结果不参与折外误差与权重估计，折内训练仍然是独立新模型。权重没有接触隐藏真值。等权融合与三种单尺度构成消融，能够分别考察延迟、多尺度和加权作用。

三折的含义是离线恢复场景中的可观测标签划分，并非滚动时间预测。验证使用其他时间段的标签，也使用完整参考的未来坐标。其误差不能直接解释为未来预测泛化误差。连续长缺失与随机缺失的差别通过单独实验组检验。

## 许可与归属

`embedding.py` 的前向延迟索引是对作者 `PhaSpaRecon.m` 的 Python 移植，`models.py` 的原版 FPRM 流程参考作者主脚本；这些改编部分按作者发布的 CC BY 4.0 保留归属。本项目增加了输入校验、可重复实验、Python GPR 参数约定、多尺度融合、防泄漏测试、统计与输出管理。使用或再发布时保留本文件及来源链接。
