set -x
export WANDB_API_KEY="004ba186f7e1f9bd08fe620ddeaaf98ef356c95f"
export RAY_BACKEND_LOG_LEVEL=error
export NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

## model, file and save path 
project_name='Folding-Thoughts'
experiment_name='D6-FormatPunish-Qwen2.5-Math-7B-16k-think-Prompt2StepFold-Openr1Math46k-StepFold-r4k'
model_name_or_path=/mnt/weka/home/yongxin.wang/workspace/lark/swift-pipeline/ckpt/think-step/Qwen2.5-Math-7B-16k-think-Prompt2-Step-Fold/v0-20251228-174712/checkpoint-5493
train_path=data/think-fold/openr1-math-46k.parquet  # training data path
test_path=data/think-fold/amc23_aime2425_math500_minerva.parquet
save_path=checkpoints/${project_name}/${experiment_name} # define the path for saving RL intermediate checkpoints

## system parameters
use_chat_template=True
val_before_train=False # set to 1 to launch validation before inference
use_dynamic_bsz=True
tensor_model_parallel_size=1 # rollout and training batch size
use_tqdm=True # whether using tqdm in vLLM generation
save_freq=100
test_freq=100
total_training_steps=500

## training parameters
max_generation_steps=6
val_max_generation_steps=8
apply_format_punish=True
norm_adv_by_std_in_grpo=False
total_epochs=30
train_batch_size=128 #128
val_batch_size=512 #128
ppo_mini_batch_size=128 # 64·
ppo_micro_batch_size_per_gpu=16 #16
log_prob_micro_batch_size_per_gpu=1 #1
kl_coef=0.0
kl_loss_coef=0.0
n_samples=8 # 8
val_samples=8 # 8
temperature=1.0
max_prompt_length=12288 # 12288
max_response_length=4096
ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length))
estimator=grpo
use_kl_loss=$( [ "$(echo "$kl_loss_coef > 0.0" | bc)" -eq 1 ] && echo true || echo false )
use_kl_in_reward=$( [ "$(echo "$kl_coef > 0.0" | bc)" -eq 1 ] && echo true || echo false )


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${estimator} \
    data.train_files=${train_path} \
    data.val_files=${test_path} \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=${val_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.use_chat_template=${use_chat_template} \
    data.reward_fn_key='reward_key' \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.model.path=${model_name_or_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$((log_prob_micro_batch_size_per_gpu * 2)) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.n=${n_samples} \
    +actor_rollout_ref.rollout.max_generation_steps=${max_generation_steps} \
    +actor_rollout_ref.rollout.val_max_generation_steps=${val_max_generation_steps} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.n=${val_samples} \
    actor_rollout_ref.rollout.use_tqdm=${use_tqdm} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$((log_prob_micro_batch_size_per_gpu * 2)) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    +algorithm.apply_format_punish=${apply_format_punish} \
    trainer.critic_warmup=0 \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq}  \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.val_before_train=${val_before_train} \
    trainer.default_local_dir=${save_path} \
    trainer.task='fold-thought' $@ # set the task to 'rl' or 'svs'