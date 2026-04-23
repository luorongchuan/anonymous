<div align="center">

# DWCAL-GRPO: Integrating Dual Preference Mechanisms into GRPO [cite: 1]

</div>

---

## 💡 what is DWCAL-GRPO?

DWCAL-GRPO (Dynamically Weighted Contrastive Advantage Learning) is an enhanced reinforcement learning framework optimized for the mathematical reasoning capabilities of large language models. Building upon the standard GRPO framework , it integrates a dual preference mechanism designed to more effectively leverage intra-group ranking information. This approach aims to refine the differentiation of reasoning paths, providing the granularity necessary to distinguish between nearly correct derivations and fundamentally flawed trajectories.

---

## ✨ 核心特性

* [cite_start]**双重偏好分支 (Dual Preference Branches)** [cite: 137, 139, 582]：
    * [cite_start]**强偏好 (Strong Preference)**：采用动态加权策略，加强对低质量和冗长轨迹的惩罚 [cite: 57, 139, 581, 589]。
    * [cite_start]**弱偏好 (Weak Preference)**：利用精细的对比建模，捕捉奖励相近的边界轨迹之间的相对差异 [cite: 58, 139, 581, 589]。
* [cite_start]**自适应损失平衡 (Adaptive Loss Balancing)**：采用基于损失比例的自适应正则化机制，动态平衡 GRPO 目标与辅助偏好分支 [cite: 229, 230]。
* [cite_start]**无评论家模型的高效性 (Critic-Free Efficiency)**：保留了 GRPO 的架构优势，无需额外的价值模型 [cite: 63, 553, 580]。
* [cite_start]**极低的显存开销 (Negligible Memory Overhead)**：相比基准方法，其峰值显存占用增加极小（例如约 2.8–3.9 GiB）[cite: 748, 749, 751]。

---

## 🏗️ 算法原理

[cite_start]全量目标函数定义为 [cite: 226]：

$$\mathcal{J}_{DWCAL}(\theta) = \mathcal{J}_{GRPO}(\theta) + \lambda_{s}\mathcal{J}_{strong}(\theta) + \lambda_{w}\mathcal{J}_{weak}(\theta)$$

[cite_start]动态强偏好对的权重定义为 [cite: 185]：

$$w_{q,j}^{s} = 1 + u_{q,j} + \mathbb{I}[A_{j} < 0]L_{q,j} + g_{q}$$

[cite_start]其中 $L_{q,j}$ 量化了超过组平均值的相对超长长度，用于惩罚冗余输出 [cite: 188, 194]。

---

## 📊 实验结果 (Pass@k)

[cite_start]在 Qwen2.5-7B-Instruct 上的实验表明，DWCAL 一致提升了多项数学基准测试的分布外 (OOD) 泛化能力 [cite: 316, 319, 590]。

| 数据集 | k | GRPO | AMIR | **DWCAL** |
| :--- | :---: | :---: | :---: | :---: |
| **AIME25** | 1 | 4.2 | 5.8 | [cite_start]**6.3 (+0.5)** [cite: 316] |
| **AMC23** | 1 | 45.0 | 47.8 | [cite_start]**50.3 (+2.5)** [cite: 316] |
| **LiveMathBench** | 1 | 27.4 | 30.2 | [cite_start]**33.4 (+3.2)** [cite: 316] |

---

## 📂 项目结构

```text
├── Case_analysize/    # 案例分析与结果可视化
├── Data/              # 数据集处理与加载
├── Eval/              # 评估流水线
├── Evalcoverage/      # 覆盖度指标评估
├── Evalmargin/        # 边际奖励分析
├── Train/             # 核心训练逻辑
│   └── trainer.py     # GRPO + DWCAL 训练器
├── config.py          # 配置定义
├── requirements.txt   # 依赖项
├── run.sh             # 一键运行脚本
└── README.md          # 当前文件
