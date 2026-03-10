# 论文 / 审计 / 项目映射

本文件把 `report/` 中的原始论文与审计报告，和当前仓库实现直接对齐，避免结论只停留在 PDF/TEX 层面。

## 1. 材料对应关系

### 原始论文

- [report/V3.tex](/Users/zero_lab/Documents/resolution_edl_experiment/report/V3.tex)
- 论文核心目标不是提出新的通用不确定性理论，而是把显式 unknown 状态拆成两部分：
  - `r`: resolution ratio，表示已解析质量
  - `tau`: resolved subset 上的 truth ratio
- 论文的实验目标也比较具体：
  - 在 CIFAR-10 / SVHN 上，把普通 predictive entropy 匹配之后
  - 检查 `ID-hard` 和 `OOD` 是否还会在 `(r, H_cont)` 平面上残留可分结构

### 修复报告

- [REPAIR_REPORT.md](/Users/zero_lab/Documents/resolution_edl_experiment/REPAIR_REPORT.md)
- 修复报告解决的是实现层问题：
  - train/val transform 污染
  - 未使用 `mps`
  - 训练中重复跑 test
  - MPS 同步抖动
  - 评估字段名错误
  - 纯 EDL from scratch 训练失败
- 这份报告把项目从“代码有明显 bug”修到“实验可以跑通”。

### 审计报告

- [report/entropy_s_structural_evaluation_report.tex](/Users/zero_lab/Documents/resolution_edl_experiment/report/entropy_s_structural_evaluation_report.tex)
- 审计报告给出的核心判断比修复报告更深一层：
  - 当前失败主要不是训练 recipe 脏，而是当前 EDL 实例化并不自然支持论文想要的 explicit-unknown 语义
  - `softplus` evidence 在中性 logit 下就有正 evidence，导致 vacuity floor 偏高，模型默认离 unresolved vertex 很远
  - 当前训练只看 one-hot CIFAR-10 标签，没有显式机制教模型对 novel input 保持低 evidence
  - 当前 benchmark 还叠加了 cohort 漂移、低 `r` 区域误用 `H_cont`、以及“匹配判别任务”和“选择性拒识任务”混用的问题

## 2. 论文主张在代码中的落点

### 训练目标

- [train_edl_cifar10.py](/Users/zero_lab/Documents/resolution_edl_experiment/train_edl_cifar10.py)
- [utils_evidential.py](/Users/zero_lab/Documents/resolution_edl_experiment/utils_evidential.py)
- 当前 EDL loss 仍然是标准 Sensoy 风格：
  - `alpha = softplus(logits) + 1`
  - data-fit + annealed KL
- 这对应论文里的“用 EDL 输出显式 unknown 质量，再投影到 `(r, tau)` 诊断平面”。
- 审计报告指出的问题也正落在这里：
  - target evidence scale 实际上没有被良好约束
  - non-target evidence 被压制，但 novel input 为什么要低 evidence，没有监督信号
  - 因此“理论上需要 unknown 轴”，不等于“标准 EDL 会自动学出这个轴”

### 诊断指标

- [utils_evidential.py](/Users/zero_lab/Documents/resolution_edl_experiment/utils_evidential.py)
- 当前仓库已经把论文中的主要诊断量落地成可计算指标：
  - `pred_entropy`
  - `vacuity`
  - `dissonance`
  - `r`
  - `tau`
  - `h_cont`
  - `completion_score(beta)`
- 其中一个重要事实已经在代码和 README 中体现：
  - 当前实现下 `vacuity == 1 - r`
  - 所以这两个 baseline 不是独立证据

### 实验协议

- [evaluate_resolution_diagnostics.py](/Users/zero_lab/Documents/resolution_edl_experiment/evaluate_resolution_diagnostics.py)
- 当前协议和论文目标基本一致，但存在三个关键偏差：
  - `ID-hard` 由当前被评估模型自己的最小 margin 样本定义，导致 cohort 会随模型漂移
  - pair probe 在 entropy-matched 判别任务上训练，却直接复用到 mixed-stream selective risk 任务
  - `H_cont` 在低 `r` 区域被直接拿来用，和论文里“content fiber 在低解析度处退化”的几何直觉不一致

### 合成验证

- [synthetic_resolution_demo.py](/Users/zero_lab/Documents/resolution_edl_experiment/synthetic_resolution_demo.py)
- 这个脚本是“诊断图可视化 sanity check”，不是“真实训练能学出该结构”的证据。
- 审计报告的判断是合理的：它验证的是 chart，不是 learner。

## 3. 当前结果说明了什么

### 已成立的结论

- [report/matched_auroc.csv](/Users/zero_lab/Documents/resolution_edl_experiment/report/matched_auroc.csv)
- [report/risk_coverage_summary.csv](/Users/zero_lab/Documents/resolution_edl_experiment/report/risk_coverage_summary.csv)
- 在 entropy-matched AUROC 上，`pair_r_hcont_logistic_probe = 0.603`，高于各类单标量基线的 `0.505 - 0.524`。
- 这说明论文最核心的“二维诊断平面可能留有 entropy 看不到的残差信息”并没有完全失败。

### 尚未成立的结论

- 同一批结果里，mixed-stream selective risk 上 pair probe 反而差于 predictive entropy：
  - `eta=0.1`: `0.105 > 0.055`
  - `eta=0.3`: `0.257 > 0.214`
  - `eta=0.5`: `0.449 > 0.432`
- 这说明：
  - `(r, H_cont)` 作为“匹配后的结构诊断”有价值
  - 但它目前不是一个稳定的 abstention score
- 因此，项目当前最稳妥的定位不是“EDL 已验证论文实验故事”，而是：
  - 论文的诊断语言仍然有用
  - 但标准 EDL 并不是这个诊断语言的理想生成器

## 4. 项目层面的统一结论

把论文、修复报告、审计报告和代码放在一起看，当前项目处于下面这个状态：

1. 理论层：
   - `V3` 的核心数学对象和实验问题仍然成立。
2. 实现层：
   - 代码级 bug 已经基本修完，训练和评估链路可以正常工作。
3. 模型层：
   - 主要瓶颈不再是脚本 bug，而是标准 EDL 的结构偏差。
4. 基准层：
   - 当前评估协议还能进一步去噪，否则很难干净地区分“模型失败”和“协议设计使结果变脏”。

因此，下一版实验不应继续围绕“把现有脚本再微调一点”展开，而应转向：

- 固定 cohort
- 改写低 `r` 区域诊断
- 单独验证 evidence function
- 引入显式 unknown 训练信号
- 在报告里严格区分 matched discrimination 和 selective rejection 两个任务

对应的具体实验设计见 [EXPERIMENT_V2_PLAN.md](/Users/zero_lab/Documents/resolution_edl_experiment/EXPERIMENT_V2_PLAN.md)。
