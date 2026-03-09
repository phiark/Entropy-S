# 修复报告

日期：2026-03-09

## 1. 背景

本次修复针对 `resolution_edl_experiment` 中 CIFAR-10 EDL 训练与诊断流程进行了排查、修正和复验。最初暴露出来的问题包括：

- 训练/验证集 `transform` 共享导致训练增强被错误覆盖
- macOS MPS 设备未被使用，训练和评估错误回退到 CPU
- 训练阶段每个 epoch 都重复跑 `test`
- 训练进度出现明显“忽快忽慢”
- 评估脚本因字段名不一致崩溃
- 纯 EDL from scratch 训练效果异常差，导致后续诊断结果失真

## 2. 问题与根因

### 2.1 `transform` 共享 bug

文件：[train_edl_cifar10.py](/Users/zero_lab/Documents/resolution_edl_experiment/train_edl_cifar10.py)

原实现先创建一个 `full_train`，再用 `random_split` 切出 `train_set` 和 `val_set`，然后把 `val_set.dataset.transform` 改成 `eval_transform`。  
由于两个 `Subset` 指向同一个底层 `CIFAR10` 实例，修改验证集 `transform` 会同时污染训练集。

根因：`Subset.dataset` 共享同一个 dataset 对象。

### 2.2 设备选择错误

文件：[train_edl_cifar10.py](/Users/zero_lab/Documents/resolution_edl_experiment/train_edl_cifar10.py)、[evaluate_resolution_diagnostics.py](/Users/zero_lab/Documents/resolution_edl_experiment/evaluate_resolution_diagnostics.py)

原实现只判断 `cuda`，否则直接回退到 `cpu`。  
在当前 `arm64 macOS` 环境下，`torch` 支持 `mps`，但脚本没有使用，导致训练和评估速度显著下降。

根因：设备选择逻辑不完整。

### 2.3 每轮重复跑 `test`

文件：[train_edl_cifar10.py](/Users/zero_lab/Documents/resolution_edl_experiment/train_edl_cifar10.py)

原实现每个 epoch 结束后都跑一遍 `val` 和 `test`。  
`val` 用于模型选择是合理的，但 `test` 每轮运行既增加时间成本，也引入测试集反复暴露的问题。

根因：训练循环中混入了本应只在训练结束后执行的测试评估。

### 2.4 训练时“忽快忽慢”

文件：[train_edl_cifar10.py](/Users/zero_lab/Documents/resolution_edl_experiment/train_edl_cifar10.py)

原实现每个 batch 都做以下操作：

- `loss.item()`
- `sum().item()`
- `tqdm.set_postfix(...)`

在 `mps` 后端下，这些操作会频繁触发设备同步，表现为 batch 时间抖动明显。

根因：高频 CPU/GPU(MPS) 同步。

### 2.5 评估脚本字段名错误

文件：[evaluate_resolution_diagnostics.py](/Users/zero_lab/Documents/resolution_edl_experiment/evaluate_resolution_diagnostics.py)

评估脚本中把该分数命名为 `predictive_entropy`，但 `EvidentialMetrics` 中真实字段名是 `pred_entropy`。  
在混合流风险评估阶段通过 `getattr(svhn_metrics, name)` 读取时触发：

`AttributeError: 'EvidentialMetrics' object has no attribute 'predictive_entropy'`

根因：表格展示名与真实字段名未做映射。

### 2.6 纯 EDL from scratch 训练失效

文件：[train_edl_cifar10.py](/Users/zero_lab/Documents/resolution_edl_experiment/train_edl_cifar10.py)、[utils_evidential.py](/Users/zero_lab/Documents/resolution_edl_experiment/utils_evidential.py)

初始正式训练结果：

- `best val_acc ≈ 0.496`
- `test_acc ≈ 0.493`

这说明问题已经不只是诊断脚本，而是底层分类器本身没有训练起来。

进一步做小样本 sanity check 后发现：

- 同一网络配合 CE 可以快速过拟合
- EDL 在无增强时可以过拟合小子集
- EDL 一旦配合当前增强和 from-scratch recipe，表现明显恶化

根因：当前 `EDL + from scratch + 现有增强/优化超参` 组合不稳定，不适合作为唯一训练路径。

## 3. 已完成修复

### 3.1 修复数据集切分

修改为“双 dataset + 同一组 indices”：

- 训练集使用独立的 `CIFAR10(..., transform=train_transform)`
- 验证集使用独立的 `CIFAR10(..., transform=eval_transform)`
- 用同一组随机索引切分，再分别包装成 `Subset`

效果：训练增强不会再被验证集 `transform` 覆盖。

### 3.2 补全设备选择与 DataLoader 参数

在 [utils_evidential.py](/Users/zero_lab/Documents/resolution_edl_experiment/utils_evidential.py) 中新增：

- `get_default_device()`
- `build_dataloader_kwargs()`

设备选择顺序：

- `cuda`
- `mps`
- `cpu`

同时：

- 仅在 CUDA 下启用 `pin_memory`
- `num_workers > 0` 时启用 `persistent_workers`

效果：当前机器会正确打印 `Using device: mps`，训练和评估不再错误使用 CPU。

### 3.3 调整训练阶段评估策略

训练时：

- 保留每个 epoch 的 `val`
- 移除每个 epoch 的 `test`

训练结束后：

- 读取 `best.pt`
- 只跑一次最终 `test`

效果：减少冗余开销，避免测试集在训练过程中被反复查看。

### 3.4 降低训练过程的同步抖动

训练与评估改为：

- 在设备端累计 `loss` / `correct`
- 仅在 `log_interval` 才回传 CPU
- 用 `logits.argmax(dim=1)` 代替多余的 `softmax(...).argmax(...)`

新增参数：

- `--log-interval`

效果：训练进度条更稳定，batch 级别抖动明显减轻。

### 3.5 修复评估脚本字段映射

在 [evaluate_resolution_diagnostics.py](/Users/zero_lab/Documents/resolution_edl_experiment/evaluate_resolution_diagnostics.py) 中新增显式映射函数：

- `predictive_entropy -> metrics.pred_entropy`
- `resolution_ratio_r -> 1 - metrics.r`
- `projected_entropy_beta_*`
- `pair_r_hcont_logistic_probe`

效果：评估脚本已可完整跑通，输出 CSV 与图片文件。

### 3.6 新增对照训练路径

在 [train_edl_cifar10.py](/Users/zero_lab/Documents/resolution_edl_experiment/train_edl_cifar10.py) 中新增：

- `--loss {edl,ce}`
- `--init-checkpoint`
- `--no-train-augment`

并在 [README.md](/Users/zero_lab/Documents/resolution_edl_experiment/README.md) 中新增推荐实验流程：

1. 先跑 `CE baseline`
2. 再跑 `CE -> EDL finetune`

效果：可以把“分类器底座质量”与“EDL 目标本身的影响”拆开分析。

## 4. 验证结果

### 4.1 代码级验证

已完成：

- `py_compile` 语法检查通过
- 训练脚本 CLI 参数检查通过
- 评估脚本成功产出：
  - `matched_auroc.csv`
  - `matched_scatter.png`
  - `risk_coverage_summary.csv`
  - `risk_coverage_eta_0.10.png`
  - `beta_scan_ablation.csv`
  - `summary.json`

### 4.2 设备与速度验证

在当前 `mps` 环境下，简化 benchmark 显示：

- CPU 训练步约 `1.8s/step`
- MPS 训练步约 `0.19s/step`

说明设备修复前后存在接近一个数量级的差异。

### 4.3 训练结果对照

#### 纯 EDL from scratch

文件：[runs/cifar10_edl/history.json](/Users/zero_lab/Documents/resolution_edl_experiment/runs/cifar10_edl/history.json)

- 最佳 `val_acc ≈ 0.496`
- 最终 `test_acc ≈ 0.493`

结论：训练失败，底座分类器过弱。

#### CE baseline

文件：[runs/cifar10_ce/history.json](/Users/zero_lab/Documents/resolution_edl_experiment/runs/cifar10_ce/history.json)

- 最佳 `val_acc ≈ 0.9312`

结论：网络与数据管线本身正常。

#### EDL from CE

文件：[runs/cifar10_edl_from_ce/history.json](/Users/zero_lab/Documents/resolution_edl_experiment/runs/cifar10_edl_from_ce/history.json)

- 最佳 `val_acc ≈ 0.9100`

结论：EDL 作为 CE 预训练后的微调目标是可行的，但从零开始训练不稳定。

## 5. 诊断实验结果解读

### 5.1 纯 EDL 的结论不可信

文件：[runs/cifar10_eval](/Users/zero_lab/Documents/resolution_edl_experiment/runs/cifar10_eval)

由于底座分类器只有约 `49%` 精度，绝大多数单维诊断指标都接近随机。  
因此，纯 EDL 跑出的诊断图和 AUROC 不应作为主要结论依据。

### 5.2 CE 与 EDL-from-CE 对照后可得出的结论

文件：

- [runs/cifar10_eval_ce](/Users/zero_lab/Documents/resolution_edl_experiment/runs/cifar10_eval_ce)
- [runs/cifar10_eval_from_ce](/Users/zero_lab/Documents/resolution_edl_experiment/runs/cifar10_eval_from_ce)

结果表明：

- 强分类器底座下，诊断实验整体恢复到合理区间
- `EDL from CE` 相比 `CE` 在部分 mixed-stream selective risk 指标上略有改善
- 但目前还没有形成强而稳定的“EDL 显著优于 CE”的证据

更稳妥的结论是：

- 原始异常主要来自纯 EDL 训练失败
- 在强底座上，EDL 可能带来有限增益，但需要多 seed 复验

## 6. 剩余问题与建议

### 6.1 尚未修复但已确认的问题

文件：[evaluate_resolution_diagnostics.py](/Users/zero_lab/Documents/resolution_edl_experiment/evaluate_resolution_diagnostics.py)、[utils_evidential.py](/Users/zero_lab/Documents/resolution_edl_experiment/utils_evidential.py)

当前：

- `vacuity = p_u`
- `resolution_ratio_r` 在评估里实际使用的是 `1 - r = p_u`

因此这两个指标本质上重复，不应被视为独立证据。

### 6.2 建议的下一步

1. 对 `CE` 与 `EDL from CE` 各自运行多个随机 seed
2. 清理评估中的重复指标 `resolution_ratio_r`
3. 再比较：
   - `matched_auroc.csv`
   - `risk_coverage_summary.csv`
   - `beta_scan_ablation.csv`

如果多 seed 下 `EDL from CE` 仍然稳定优于 `CE`，才适合把该结论写入正式实验结论。

## 7. 当前状态

当前项目已经完成以下关键修复：

- 数据集 `transform` 共享 bug 已修复
- 设备使用错误已修复
- 训练冗余测试已移除
- 训练同步抖动已缓解
- 评估脚本崩溃已修复
- 已建立 `CE baseline` 与 `EDL from CE` 的可复现实验路径

当前系统已从“结果明显失真且流程不稳定”恢复到“可以进行有效对照实验”的状态。
