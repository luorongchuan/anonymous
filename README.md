<div align="center">

# DWCAL-GRPO: Integrating Dual Preference Mechanisms into GRPO

</div>

---

##  What is DWCAL-GRPO?

DWCAL-GRPO (Dynamically Weighted Contrastive Advantage Learning) is an enhanced reinforcement learning framework optimized for the mathematical reasoning capabilities of large language models. Building upon the standard GRPO framework , it integrates a dual preference mechanism designed to more effectively leverage intra-group ranking information. This approach aims to refine the differentiation of reasoning paths, providing the granularity necessary to distinguish between nearly correct derivations and fundamentally flawed trajectories.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/AmirHosseinYari2002/AMIR-GRPO.git
cd AMIR-GRPO

# Install necessary libraries
pip install -U pip
pip install -r requirements.txt
````

---

## 🏋️ Training

Training is launched via `python -m Train.train` and the CLI mirrors the full configuration.

```bash
python -m Train.train \
  --model_name google/gemma-3-4b-it \
  --lora_rank 16 \
  --max_seq_length 2048 \
  --load_in_4bit 0 \
  --model_dir trained_model_directory \
  --dataset_name gsm8k \
  --dataset_split train \
  --trainer_type amir_grpo \
  --calibration \
  --learning_rate 5e-6 \
  --weight_decay 0.1 \
  --max_grad_norm 0.1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_steps 1000 \
  --seed 0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --optim adamw_8bit \
  --adam_beta1 0.9 \
  --adam_beta2 0.99 \
  --num_generations 8 \
  --max_prompt_length 1024 \
  --max_completion_length 1024 \
  --loss_type grpo \
  --epsilon 0.20 \
  --epsilon_high 0.20 \
  --mask_truncated_completions 0 \
  --scale_rewards group \
  --importance_sampling_level token \
  --lambda_reg 0.01 \
  --reward_margin 2.0 \
  --beta_dpo 0.2 \
  --pair_mining all \
  --max_pairs_per_group None \
  --ref_free true \
  --logging_steps 1 \
  --save_steps 50 \
  --report_to wandb \
  --wandb_api_key your_wandb_api_key

```

---

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
