export CUDA_VISIBLE_DEVICES=4

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=600


python -m Train.train \
  --model_name ./Qwen2.5-3B-Instruct \
  --lora_rank 16 \
  --max_seq_length 2048 \
  --load_in_4bit 0 \
  --model_dir ./Qwen2.5-3B-Instruct-directory \
  --dataset_name gsm8k \
  --dataset_split train \
  --trainer_type dwcal_grpo \
  --calibration \
  --learning_rate 5e-6 \
  --weight_decay 0.1 \
  --max_grad_norm 0.1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
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
  \
  --lambda_strong 0.01 \
  --reward_margin 2.0 \
  --beta_strong 0.1 \
  --pair_mode all \
  --ref_free false \
  --dpo_chunk_size 8 \
  \
  --alpha_rank 0.5 \
  --alpha_len 0.5 \
  --alpha_group 0.5 \
  --w_max 2 \
  --group_quality_threshold 0.5 \
  \
  --weak_margin 0.15 \
  --lambda_weak 0.01 \
  --weak_warmup_steps 100 \
  --beta_weak 0.05 \
  \
  --logging_steps 1 \
  --save_steps  200\
  --report_to wandb \

echo "Training started in background. Check 'train.log' for details."
