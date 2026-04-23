<div align="center">DWCAL-GRPO: Integrating Dual Preference Mechanisms into GRPO</div>💡 What is DWCAL-GRPO?DWCAL-GRPO (Dynamically Weighted Contrastive Advantage Learning) is an enhanced reinforcement learning framework optimized for the mathematical reasoning capabilities of large language models.Building upon the standard GRPO framework, it integrates a dual preference mechanism designed to more effectively leverage intra-group ranking information. This approach aims to refine the differentiation of reasoning paths, providing the granularity necessary to distinguish between nearly correct derivations and fundamentally flawed trajectories.📂 Project Structure├── Case_analysize/    # Case analysis and result visualization
├── Data/              # Dataset processing and loading
├── Eval/              # Evaluation pipeline
├── Evalcoverage/      # Coverage metric evaluation
├── Evalmargin/        # Margin reward analysis
├── Train/             # Core training logic
│   └── trainer.py     # GRPO + DWCAL trainer
├── config.py          # Configuration definitions
├── requirements.txt   # Dependencies
├── run.sh             # One-click run script
└── README.md          # This file
🏋️ TrainingThe training process is automated via the provided shell script. To start training in the background, use:nohup bash run.sh &
🔬 EvaluationYou can evaluate the performance of your trained model using the evaluation script. For example, to evaluate on the OlympiadBench dataset:python -m Eval.eval \
  --dataset_name olympiadbench \
  --model_dir trained_model_directory
📝 CitationIf you find this work useful in your research, please cite our paper:@inproceedings{dwcal2026,
  title={DWCAL-GRPO: Integrating Dual Preference Mechanisms into GRPO},
  author={Anonymous Authors},
  booktitle={The 40th Conference on Neural Information Processing Systems (NeurIPS 2026)},
  year={2026}
}
<p align="center">Built for Advanced Mathematical Reasoning in Large Language Models.</p>
