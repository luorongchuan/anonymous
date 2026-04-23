<div align="center">

# DWCAL-GRPO: Integrating Dual Preference Mechanisms into GRPO

</div>

---

##  What is DWCAL-GRPO?

DWCAL-GRPO (Dynamically Weighted Contrastive Advantage Learning) is an enhanced reinforcement learning framework optimized for the mathematical reasoning capabilities of large language models. Building upon the standard GRPO framework , it integrates a dual preference mechanism designed to more effectively leverage intra-group ranking information. This approach aims to refine the differentiation of reasoning paths, providing the granularity necessary to distinguish between nearly correct derivations and fundamentally flawed trajectories.

---



Project Structure

├── Case_analysize/    # Case analysis and result visualization
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


## 🏋️ Training

Training is launched via `nohup bash run.sh` 


## 🔬 Evaluation

Evaluate the performance of the trained model using the Eval.eval script.

```bash
python -m Eval.eval \
  --dataset_name olympiadbench \
  --model_dir trained_model_directory
```

---


## 📝 Citation

```bibtex
@misc{yari2026amirgrpoinducingimplicitpreference,
      title={AMIR-GRPO: Inducing Implicit Preference Signals into GRPO}, 
      author={Amir Hossein Yari and Fajri Koto},
      year={2026},
      eprint={2601.03661},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.03661}, 
}
