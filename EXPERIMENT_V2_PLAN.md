# 下一版实验设计

本计划直接基于 [report/entropy_s_structural_evaluation_report.tex](/Users/zero_lab/Documents/resolution_edl_experiment/report/entropy_s_structural_evaluation_report.tex) 的审计结论来改写实验，不再把重点放在“继续调现有 from-scratch EDL recipe”，而是先把 benchmark 和建模假设拆干净。

## 1. 目标重写

下一版实验只回答两个问题，不混在一起：

### Q1. 结构诊断问题

在 ordinary predictive entropy 被匹配之后，`ID-hard` 与 `OOD` 是否仍能在 resolution-aware 坐标里分离？

### Q2. 拒识决策问题

如果把 uncertainty score 用于 mixed-stream selective rejection，哪个分数在给定 coverage 下风险最低？

这两个问题分别产出，不再共享一个“万能最优分数”的叙事。

## 2. V2 的实验原则

### 原则 A: 固定 cohort，不让 benchmark 随模型漂移

当前 `ID-hard` 是由被评估模型自身的 margin 排序得到的，这会污染比较。

V2 方案：

- 先训练一个强 reference model，默认使用 `CE baseline`
- 用 reference model 在 CIFAR-10 test 上定义固定 `ID-hard` 集合
- 对 SVHN 也固定随机采样集合和匹配索引
- 所有待比较模型都在同一组 index 上评估

交付物：

- `runs/reference_cohort/id_hard_indices.npy`
- `runs/reference_cohort/ood_indices.npy`
- `runs/reference_cohort/matched_pairs.json`

## 3. V2 的模型矩阵

V2 不再只比较 “pure EDL” 和 “CE -> EDL” 两条线，而是拆成四类。

### A. Reference CE

用途：

- 提供稳定分类底座
- 提供固定 cohort
- 作为 pseudo-evidential baseline

训练方式：

- `--loss ce`

### B. Warm-start EDL

用途：

- 作为当前项目最可用的 EDL 基线

训练方式：

- 先 `CE`
- 再 `EDL finetune`

### C. Evidence function ablation

用途：

- 直接验证审计报告所说的 neutral-point vacuity floor 是否是瓶颈

要比较的 evidence map：

- `softplus`
- `shifted_softplus`
- `relu`

建议实现：

- 在 `utils_evidential.py` 的 `evidential_classification_loss` 里把 evidence function 参数化
- 评估时同步记录中性 logit `z=0` 下的理论 vacuity floor

### D. Explicit unknown training signal

用途：

- 验证“没有 unknown 监督信号，模型不会自然学出低 evidence OOD”这一审计结论

优先顺序：

1. OE-light: 用简单噪声/texture 数据做 outlier exposure
2. OE-SVHN-train: 若要避免评估泄漏，则不要直接用 SVHN test；可用独立 OOD 源或 SVHN train 只作训练辅助
3. Abstention regularizer: 直接对 OOD 辅助样本压低总 evidence

最小可行版本：

- 先做 `CE -> EDL + OE-light`
- 不先做复杂的新网络结构

## 4. V2 的评估协议

### 4.1 Matched discrimination

保持论文最核心的问题，但修正三个细节。

协议：

- 用固定 cohort
- 用 predictive entropy 做 bin matching
- 主图仍然画 `(r, H_cont)`，但表格主指标改为两组：
  - raw `(r, H_cont)`
  - geometry-aware `(r, r * H_cont)` 或 `r > r_min` 后的 `(r, H_cont)`

主指标：

- matched AUROC
- matched PR-AUC
- 每个 bin 内的 residual separation summary

预期：

- 如果理论层仍有效，那么二维诊断应继续优于单标量 entropy baseline
- 如果低 `r` 处理改对后 AUROC 上升，说明之前确实存在几何失真

### 4.2 Selective rejection

把它明确作为单独任务。

协议：

- 混合流设置维持 `eta in {0.1, 0.3, 0.5}`
- uncertainty score 不再默认复用 matched discrimination 上训练出的同一个 pair probe
- selective rejection 的二元分数需要单独训练或直接用 hand-crafted score

建议比较：

- `pred_entropy`
- `vacuity`
- `dissonance`
- `1 - r`
- `H_beta`
- `r * H_cont`
- `pair_probe_matched`
- `pair_probe_selective`

主指标：

- risk@80% coverage
- AURC
- 不同 `eta` 下的稳定性

预期：

- `pair_probe_matched` 不一定适合 selective rejection
- `pair_probe_selective` 如果仍无改善，说明二维图像信息和决策效用之间确实存在 gap

## 5. V2 的最小实现顺序

先做 benchmark 去噪，再做模型扩展。顺序不要反。

### Phase 1: 协议修正

1. 固定 reference cohort
2. 在评估脚本中支持读取固定 index
3. 新增 `r_hcont_weighted = r * h_cont`
4. 把 matched discrimination 和 selective rejection 分成两套输出文件

完成标准：

- 同一份 cohort 可被 CE / EDL / EDL+OE 重复使用
- 输出表中不再把两类任务混成同一种“优劣”

### Phase 2: evidence function ablation

1. 支持 `softplus` / `shifted_softplus` / `relu`
2. 报告 neutral-point vacuity floor
3. 比较：
   - 分类精度
   - matched AUROC
   - mixed-stream risk

完成标准：

- 能明确回答“softplus floor 是否显著限制 OOD vacuity”

### Phase 3: explicit unknown signal

1. 加入 OE-light 数据流
2. 对 OOD 辅助样本施加 low-evidence 目标
3. 先只在 warm-start EDL 上验证

完成标准：

- OOD 是否更稳定地落向低 `r`
- matched discrimination 和 selective rejection 是否同时改善，或只改善其中一项

### Phase 4: 结构化模型

如果前三阶段仍不能稳定实现论文语义，再进入结构改模。

方向：

- resolution gate head 预测 `r`
- conditional content head 预测 `tau` 或 resolved-class distribution
- 不再完全依赖单个 Dirichlet evidence 向量同时承担 resolution 与 content 两种职责

这一步是 V2 后半段，不应该在 benchmark 还没修干净时提前做。

## 6. 建议的输出目录

建议把 V2 结果按任务和模型拆开：

- `runs/v2/reference_cohort/`
- `runs/v2/ce_reference/`
- `runs/v2/edl_warmstart_softplus/`
- `runs/v2/edl_warmstart_shifted_softplus/`
- `runs/v2/edl_warmstart_relu/`
- `runs/v2/edl_warmstart_oe_light/`

每个实验目录下至少包含：

- `summary.json`
- `matched_auroc.csv`
- `matched_pr_auc.csv`
- `risk_coverage_summary.csv`
- `aurc_summary.csv`
- `cohort_manifest.json`

## 7. V2 的成败判据

### 成功

- 固定 cohort 后，二维诊断结论跨模型稳定
- geometry-aware 版本优于 raw `H_cont`
- 加入 explicit unknown signal 后，OOD 在低 `r` 方向更稳定聚集

### 部分成功

- 诊断图有信息，但 selective rejection 仍不提升
- 这时项目应转成“诊断语言有效，但决策分数未必直接受益”的结论

### 失败但仍有论文价值

- 即使 benchmark 去噪、evidence function 替换、OE-light 加入后，标准 EDL 仍不能稳定产生期望几何
- 那么负结论本身就是结果：
  - `V3` 的理论图景可以保留
  - 但“标准 EDL 是其自然实现”这个实验叙事需要收缩

## 8. 当前建议

最应该先做的不是继续补调参，而是做下面四件事：

1. 固定 `ID-hard` / `OOD` / matching cohort
2. 把评估中的 `H_cont` 改成同时报告 raw 和 `r * H_cont`
3. 给 loss / infer / eval 全链路加可切换 evidence function
4. 做一个最小的 `warm-start EDL + OE-light` 对照

只要这四步完成，下一轮结果就足够判断问题到底主要卡在：

- benchmark
- evidence parameterization
- 还是“缺少显式 unknown 监督信号”
