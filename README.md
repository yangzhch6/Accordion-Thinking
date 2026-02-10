<div align="center">


<h1 style="display: flex; justify-content: center; align-items: center; gap: 10px; margin: 0;">
  Accordion-Thinking: Self-Regulated Step Summaries for Efficient and Readable LLM Reasoning
</h1>
<p align="center">
  <em></em>
  
  <!-- <em>**We treat DARS as the focal loss in RLVR.**</em> -->
</p>
<div align="center">
  <img src="./figs/example.png" alt="overview" style="width: 80%; height: auto;">
</div>


<div align="center">
<a href="https://arxiv.org/abs/2602.03249"><img src="./figs/arxiv.png" width="15%"></a>
<a href="https://www.alphaxiv.org/abs/2602.03249"><img src="./figs/alphaxiv.png" width="15%"></a>
<a href="https://github.com/yangzhch6/Accordion-Thinking"><img src="./figs/github.png" width="15%"></a>
<a href="https://huggingface.co/datasets/yangzhch6/Accordion-Thinking-Synthetic-Data"><img src="./figs/hf.png" width="15%"></a>
</div>

</div>




## ⚙️ Setup

We recommend using [Conda](https://docs.conda.io/projects/miniconda) to manage your environment. We use [vLLM](https://github.com/vllm-project/vllm) (0.10.0) to accelerate inference. Run the following commands to setup your environment:

```sh
git git@github.com:MasterVito/SvS.git && cd SvS
conda create -n svs python=3.10.16
conda activate svs
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126 # CUDA 12.6 for example
pip install -r requirements.txt
```


## ⚡️ Training
To train the model, run the following command:

### Cold Start SFT
You may sft your base model using: https://github.com/yangzhch6/Accordion-Thinking


### Prepare RL Training Data
```python
python Folding-Thoughts/data/preprocess/think-fold/think_openr1.py
python Folding-Thoughts/data/preprocess/think-fold/think_test.py
```

### RL Training
**Mix Training:**
```sh
set -x
export WANDB_API_KEY="..."
export RAY_BACKEND_LOG_LEVEL=error
export NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

## model, file and save path 
project_name='Accordion-Thinking'
experiment_name='<your model name>-mix-d6r6k-Unfold16k'
model_name_or_path=<your cold start model here>
train_path=data/think-fold/openr1-math-46k.parquet  # training data path
test_path=data/think-fold/amc23_aime2425_math500_minerva.parquet
save_path=checkpoints/${project_name}/${experiment_name} # define the path for saving RL intermediate checkpoints

## system parameters
use_chat_template=True
val_before_train=True # set to 1 to launch validation before inference
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
max_prompt_length=14336 # 12288
max_response_length=6144
max_once_prompt_length=4096
max_once_response_length=16384

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
    +data.max_once_prompt_length=${max_once_prompt_length} \
    +data.max_once_response_length=${max_once_response_length} \
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
    trainer.task='mix-fold' $@ 
```



## ☕️ Citation

If you find this repository helpful, please consider citing our paper:

```
@misc{yang2026accordionthinkingselfregulatedstepsummaries,
      title={Accordion-Thinking: Self-Regulated Step Summaries for Efficient and Readable LLM Reasoning}, 
      author={Zhicheng Yang and Zhijiang Guo and Yinya Huang and Yongxin Wang and Wenlei Shi and Yiwei Wang and Xiaodan Liang and Jing Tang},
      year={2026},
      eprint={2602.03249},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.03249}, 
}
```
<br>

## 🙏 Acknowledgement
We sincerely appreciate the outstanding work of [veRL](https://github.com/volcengine/verl) and [SvS](https://github.com/MasterVito/SvS). The the training code is adapted from the veRL and SvS repository.
