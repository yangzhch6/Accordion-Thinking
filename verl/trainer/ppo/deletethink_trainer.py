from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayMixFoldThoughtTrainer,
    apply_kl_penalty,
    compute_response_mask,
)
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager.prime import run_reward_scoring


@dataclass
class DeletethinkChunk:
    prompt_ids: list[int]
    response_ids: list[int]


@dataclass
class DeletethinkTrace:
    uid: str
    sid: str
    metadata: dict[str, Any]
    original_prompt_ids: list[int]
    folded_query_ids: list[int]
    chunks: list[DeletethinkChunk] = field(default_factory=list)
    full_response_ids: list[int] = field(default_factory=list)
    cumulative_generated_length: int = 0
    peak_token_length: int = 0
    done: bool = False
    stopped_by_eos: bool = False
    stopped_by_budget: bool = False
    reward: float = 0.0
    advantage: float = 0.0
    full_output_text: str = ""
    answer_text: str = ""
    has_think_end: bool = False


class RayDeletethinkTrainer(RayMixFoldThoughtTrainer):
    """Independent Delethink trainer that leaves existing rk/fold/mix logic untouched."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deletethink_carryover_length = int(self.config.data.get("deletethink_carryover_length", 3072))
        self.deletethink_initial_fold_tokens = int(self.config.data.get("deletethink_initial_fold_tokens", 100))
        self.deletethink_max_cumulative_length = int(self.config.data.get("max_cumulative_length", self.config.data.max_response_length))
        self.deletethink_prompt_length = int(self.config.data.max_prompt_length)
        self.deletethink_response_length = int(self.config.data.max_response_length)
        if self.deletethink_carryover_length <= 0:
            raise ValueError("data.deletethink_carryover_length must be positive.")
        if self.deletethink_carryover_length >= self.deletethink_response_length:
            raise ValueError("data.deletethink_carryover_length must be smaller than data.max_response_length.")
        if self.use_rm:
            raise NotImplementedError("trainer.task='deletethink' currently supports rule-based reward managers only.")

    def _get_max_generation_steps(self, is_validation: bool = False) -> int:
        if is_validation:
            return int(self.config.actor_rollout_ref.rollout.get("val_max_generation_steps", self.config.actor_rollout_ref.rollout.max_generation_steps))
        return int(self.config.actor_rollout_ref.rollout.max_generation_steps)

    def _get_group_size(self) -> int:
        return int(self.config.actor_rollout_ref.rollout.val_kwargs.n) if self.config.actor_rollout_ref.rollout.get("val_kwargs") else int(self.config.actor_rollout_ref.rollout.n)

    def _get_rollout_size_divisor(self) -> int:
        if self.async_rollout_mode:
            return self.config.actor_rollout_ref.rollout.agent.num_workers
        return self.actor_rollout_wg.world_size

    def _get_query_budget(self) -> int:
        return max(1, self.deletethink_prompt_length - self.deletethink_carryover_length)

    def _truncate_prompt_ids(self, prompt_ids: list[int]) -> list[int]:
        if len(prompt_ids) <= self.deletethink_prompt_length:
            return list(prompt_ids)
        return list(prompt_ids[-self.deletethink_prompt_length :])

    def _truncate_folded_query_ids(self, prompt_ids: list[int]) -> list[int]:
        query_budget = self._get_query_budget()
        if len(prompt_ids) <= query_budget:
            return list(prompt_ids)
        # Keep the tail so the latest user query and the appended prefix survive.
        return list(prompt_ids[-query_budget:])

    def _build_prompt_batch_item(self, trace: DeletethinkTrace, prompt_ids: list[int]) -> DataProto:
        prompt_ids = self._truncate_prompt_ids(prompt_ids)
        input_ids = torch.full((1, self.deletethink_prompt_length), self.tokenizer.pad_token_id, dtype=torch.int64)
        attention_mask = torch.zeros((1, self.deletethink_prompt_length), dtype=torch.int64)
        if len(prompt_ids) > 0:
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.int64)
            input_ids[0, -len(prompt_ids) :] = prompt_tensor
            attention_mask[0, -len(prompt_ids) :] = 1
        position_ids = compute_position_id_with_mask(attention_mask)

        sample_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": np.array([prompt_ids], dtype=object),
            "uid": np.array([trace.uid], dtype=object),
            "sid": np.array([trace.sid], dtype=object),
        }
        for key, value in trace.metadata.items():
            sample_dict[key] = np.array([value], dtype=object)
        return DataProto.from_single_dict(sample_dict)

    def _build_generation_prompt_ids(self, trace: DeletethinkTrace) -> list[int]:
        if not trace.chunks:
            return list(trace.original_prompt_ids)
        carryover_ids = trace.chunks[-1].response_ids[-self.deletethink_carryover_length :]
        return trace.folded_query_ids + carryover_ids

    def _init_traces(self, gen_batch: DataProto) -> list[DeletethinkTrace]:
        traces: list[DeletethinkTrace] = []
        metadata_keys = [key for key in gen_batch.non_tensor_batch.keys() if key not in {"raw_prompt_ids", "uid", "sid"}]
        for item in gen_batch:
            metadata = {key: item.non_tensor_batch[key] for key in metadata_keys}
            prompt_ids = list(item.non_tensor_batch["raw_prompt_ids"])
            traces.append(
                DeletethinkTrace(
                    uid=item.non_tensor_batch["uid"],
                    sid=item.non_tensor_batch["sid"],
                    metadata=metadata,
                    original_prompt_ids=prompt_ids,
                    folded_query_ids=self._truncate_folded_query_ids(prompt_ids),
                )
            )
        return traces

    def _run_generation(self, gen_batch: DataProto) -> DataProto:
        if not self.async_rollout_mode:
            return self.actor_rollout_wg.generate_sequences(gen_batch)
        return self.async_rollout_manager.generate_sequences(gen_batch)

    def _extract_valid_response_ids(self, output_batch: DataProto, row_idx: int) -> list[int]:
        response_length = output_batch.batch["responses"].size(1)
        valid_len = int(output_batch.batch["attention_mask"][row_idx, -response_length:].sum().item())
        if valid_len <= 0:
            return []
        return output_batch.batch["responses"][row_idx, :valid_len].tolist()

    def _eos_token_ids(self) -> set[int]:
        eos_token_id = self.tokenizer.eos_token_id
        if isinstance(eos_token_id, list):
            return set(eos_token_id)
        return {int(eos_token_id)}

    def _trace_cumulative_lengths(self, traces: list[DeletethinkTrace]) -> list[int]:
        return [trace.cumulative_generated_length for trace in traces]

    def _trace_peak_token_lengths(self, traces: list[DeletethinkTrace]) -> list[int]:
        return [trace.peak_token_length for trace in traces]

    def _rollout_traces(self, gen_batch: DataProto, is_validation: bool = False) -> list[DeletethinkTrace]:
        traces = self._init_traces(gen_batch)
        max_generation_steps = self._get_max_generation_steps(is_validation=is_validation)
        first_chunk_length = self.deletethink_response_length
        later_chunk_length = self.deletethink_response_length - self.deletethink_carryover_length
        eos_token_ids = self._eos_token_ids()
        size_divisor = self._get_rollout_size_divisor()

        for generation_step in range(max_generation_steps):
            active_indices = [idx for idx, trace in enumerate(traces) if not trace.done]
            if not active_indices:
                break

            budget2indices: dict[int, list[int]] = defaultdict(list)
            for idx in active_indices:
                trace = traces[idx]
                remaining_budget = self.deletethink_max_cumulative_length - trace.cumulative_generated_length
                if remaining_budget <= 0:
                    trace.done = True
                    trace.stopped_by_budget = True
                    continue
                step_max_tokens = first_chunk_length if generation_step == 0 else later_chunk_length
                max_new_tokens = min(step_max_tokens, remaining_budget)
                if max_new_tokens <= 0:
                    trace.done = True
                    trace.stopped_by_budget = True
                    continue
                budget2indices[int(max_new_tokens)].append(idx)

            for max_new_tokens, trace_indices in budget2indices.items():
                prompt_items = []
                for trace_idx in trace_indices:
                    prompt_ids = self._build_generation_prompt_ids(traces[trace_idx])
                    prompt_items.append(self._build_prompt_batch_item(traces[trace_idx], prompt_ids))
                current_gen_batch = DataProto.concat(prompt_items)
                current_gen_batch, pad_size = pad_dataproto_to_divisor(current_gen_batch, size_divisor)
                current_gen_batch.meta_info = {
                    "once_thought": True,
                    "max_tokens": max_new_tokens,
                    "n": 1,
                }

                current_output = self._run_generation(current_gen_batch)
                current_output = unpad_dataproto(current_output, pad_size=pad_size)

                for row_idx, trace_idx in enumerate(trace_indices):
                    trace = traces[trace_idx]
                    prompt_ids = self._build_generation_prompt_ids(trace)
                    valid_response_ids = self._extract_valid_response_ids(current_output, row_idx)
                    trace.chunks.append(DeletethinkChunk(prompt_ids=prompt_ids, response_ids=valid_response_ids))
                    trace.full_response_ids.extend(valid_response_ids)
                    trace.cumulative_generated_length += len(valid_response_ids)
                    truncated_prompt_len = len(self._truncate_prompt_ids(prompt_ids))
                    trace.peak_token_length = max(trace.peak_token_length, truncated_prompt_len + len(valid_response_ids))
                    if trace.cumulative_generated_length >= self.deletethink_max_cumulative_length:
                        trace.stopped_by_budget = True
                        trace.done = True
                    if generation_step == 0:
                        folded_prefix = valid_response_ids[: self.deletethink_initial_fold_tokens]
                        trace.folded_query_ids = self._truncate_folded_query_ids(trace.original_prompt_ids + folded_prefix)
                    if valid_response_ids and valid_response_ids[-1] in eos_token_ids:
                        trace.stopped_by_eos = True
                        trace.done = True

        return traces

    def _score_traces(self, traces: list[DeletethinkTrace], reward_manager) -> tuple[list[float], dict[str, list[float]]]:
        compute_score = getattr(reward_manager, "compute_score", default_compute_score)
        reward_key = getattr(reward_manager, "reward_fn_key", self.config.data.reward_fn_key)
        think_end_token = "</think>"

        tasks = []
        ground_truths = []
        extra_infos = []
        valid_answers = []
        valid_indices = []
        scores = [0.0 for _ in traces]
        format_hits = []

        for idx, trace in enumerate(traces):
            full_output_text = self.tokenizer.decode(trace.full_response_ids, skip_special_tokens=True)
            trace.full_output_text = full_output_text
            split_pos = full_output_text.rfind(think_end_token)
            trace.has_think_end = split_pos != -1
            format_hits.append(float(trace.has_think_end))
            if trace.stopped_by_budget:
                trace.answer_text = ""
                continue
            if not trace.has_think_end:
                trace.answer_text = ""
                continue

            answer_text = full_output_text[split_pos + len(think_end_token) :].strip()
            trace.answer_text = answer_text
            valid_answers.append(answer_text)
            valid_indices.append(idx)
            tasks.append(trace.metadata.get(reward_key, trace.metadata.get("data_source", "unknown")))
            ground_truths.append(trace.metadata["reward_model"]["ground_truth"])
            extra_infos.append(trace.metadata.get("extra_info"))

        if valid_answers:
            try:
                compute_score(tasks[0], valid_answers[0], ground_truths[0], extra_infos[0])
                valid_scores = run_reward_scoring(
                    compute_score,
                    completions=valid_answers,
                    references=ground_truths,
                    tasks=tasks,
                    extra_info=extra_infos,
                    num_processes=64,
                )
            except Exception as exc:
                print(f"[deletethink] Reward scoring failed, setting valid traces to 0. {exc}")
                valid_scores = [0.0 for _ in valid_answers]
            for trace_idx, score in zip(valid_indices, valid_scores, strict=True):
                scores[trace_idx] = float(score)

        for trace, score in zip(traces, scores, strict=True):
            trace.reward = float(score)

        reward_extra_infos_dict = {
            "reward": scores,
            "has_think_end": format_hits,
            "stopped_by_budget": [float(trace.stopped_by_budget) for trace in traces],
        }
        return scores, reward_extra_infos_dict

    def _assign_trace_advantages(self, traces: list[DeletethinkTrace]) -> None:
        uid2scores: dict[str, list[float]] = defaultdict(list)
        for trace in traces:
            uid2scores[trace.uid].append(trace.reward)

        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
        epsilon = 1e-6
        uid2mean: dict[str, float] = {}
        uid2std: dict[str, float] = {}
        for uid, scores in uid2scores.items():
            if len(scores) == 1:
                uid2mean[uid] = 0.0
                uid2std[uid] = 1.0
            else:
                scores_tensor = torch.tensor(scores, dtype=torch.float32)
                uid2mean[uid] = float(torch.mean(scores_tensor).item())
                uid2std[uid] = float(torch.std(scores_tensor).item())

        for trace in traces:
            centered_reward = trace.reward - uid2mean[trace.uid]
            if norm_adv_by_std_in_grpo:
                trace.advantage = centered_reward / (uid2std[trace.uid] + epsilon)
            else:
                trace.advantage = centered_reward

    def _build_chunk_batch(self, trace: DeletethinkTrace, chunk: DeletethinkChunk, include_training_tensors: bool) -> DataProto:
        prompt_ids = self._truncate_prompt_ids(chunk.prompt_ids)
        response_ids = list(chunk.response_ids[: self.deletethink_response_length])
        prompt_len = len(prompt_ids)
        response_len = len(response_ids)

        prompts = torch.full((1, self.deletethink_prompt_length), self.tokenizer.pad_token_id, dtype=torch.int64)
        prompt_attention_mask = torch.zeros((1, self.deletethink_prompt_length), dtype=torch.int64)
        if prompt_len > 0:
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.int64)
            prompts[0, -prompt_len:] = prompt_tensor
            prompt_attention_mask[0, -prompt_len:] = 1
        prompt_position_ids = compute_position_id_with_mask(prompt_attention_mask)

        responses = torch.full((1, self.deletethink_response_length), self.tokenizer.pad_token_id, dtype=torch.int64)
        response_attention_mask = torch.zeros((1, self.deletethink_response_length), dtype=torch.int64)
        if response_len > 0:
            response_tensor = torch.tensor(response_ids, dtype=torch.int64)
            responses[0, :response_len] = response_tensor
            response_attention_mask[0, :response_len] = 1

        input_ids = torch.cat([prompts, responses], dim=-1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)
        response_delta = torch.arange(1, self.deletethink_response_length + 1, dtype=torch.int64).unsqueeze(0)
        response_position_ids = prompt_position_ids[:, -1:] + response_delta
        position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)

        sample_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_attention_mask,
            "position_ids": position_ids,
            "prompts": prompts,
            "responses": responses,
            "uid": np.array([trace.uid], dtype=object),
            "sid": np.array([trace.sid], dtype=object),
        }
        for key, value in trace.metadata.items():
            sample_dict[key] = np.array([value], dtype=object)

        batch = DataProto.from_single_dict(sample_dict)
        if include_training_tensors:
            acc = torch.tensor([trace.reward], dtype=torch.float32)
            token_level_scores = torch.zeros_like(responses, dtype=torch.float32)
            if response_len > 0:
                token_level_scores[0, response_len - 1] = trace.reward
            advantages = torch.full_like(responses, fill_value=trace.advantage, dtype=torch.float32)
            advantages = advantages * response_attention_mask

            batch.batch["acc"] = acc
            batch.batch["token_level_scores"] = token_level_scores
            batch.batch["token_level_rewards"] = token_level_scores.clone()
            batch.batch["advantages"] = advantages
            batch.batch["returns"] = advantages.clone()
        return batch

    def _make_padding_batch(self, template_batch: DataProto) -> DataProto:
        pad_batch = template_batch.select(deepcopy=True)
        pad_batch.non_tensor_batch["uid"] = np.array(["pad"], dtype=object)
        pad_batch.non_tensor_batch["sid"] = np.array(["pad"], dtype=object)
        pad_batch.batch["acc"] = torch.zeros_like(pad_batch.batch["acc"])
        pad_batch.batch["token_level_scores"] = torch.zeros_like(pad_batch.batch["token_level_scores"])
        pad_batch.batch["token_level_rewards"] = torch.zeros_like(pad_batch.batch["token_level_rewards"])
        pad_batch.batch["advantages"] = torch.zeros_like(pad_batch.batch["advantages"])
        pad_batch.batch["returns"] = torch.zeros_like(pad_batch.batch["returns"])
        pad_batch.batch["response_mask"] = torch.zeros_like(pad_batch.batch["response_mask"])
        pad_batch.batch["attention_mask"][:, -self.deletethink_response_length :] = 0
        return pad_batch

    def _build_train_batch(self, traces: list[DeletethinkTrace]) -> DataProto:
        chunk_batches = []
        for trace in traces:
            for chunk in trace.chunks:
                chunk_batches.append(self._build_chunk_batch(trace, chunk, include_training_tensors=True))

        train_batch = DataProto.concat(chunk_batches)
        pad_size = (-len(chunk_batches)) % self.actor_rollout_wg.world_size
        if pad_size:
            template_batch = chunk_batches[0]
            padding_batches = [self._make_padding_batch(template_batch) for _ in range(pad_size)]
            train_batch = DataProto.concat([train_batch] + padding_batches)
        return train_batch

    def _validate_deletethink(self):
        print("## Begin validation")
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        trace_cumulative_lengths = []
        trace_peak_lengths = []

        for test_data in tqdm(self.val_dataloader):
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)
            test_batch.non_tensor_batch["sid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in test_batch.batch["input_ids"]]
            sample_inputs.extend(input_texts)
            test_gen_batch = self.format_gen_batch(test_batch)

            traces = self._rollout_traces(test_gen_batch, is_validation=True)
            trace_scores, trace_extra_infos = self._score_traces(traces, self.val_reward_fn)

            output_texts = [trace.full_output_text for trace in traces]
            sample_outputs.extend(output_texts)
            sample_scores.extend(trace_scores)
            current_trace_cumulative_lengths = self._trace_cumulative_lengths(traces)
            current_trace_peak_lengths = self._trace_peak_token_lengths(traces)
            trace_cumulative_lengths.extend(current_trace_cumulative_lengths)
            trace_peak_lengths.extend(current_trace_peak_lengths)

            reward_extra_infos_dict["reward"].extend(trace_scores)
            reward_extra_infos_dict["acc"].extend(trace_scores)
            reward_extra_infos_dict["has_think_end"].extend(trace_extra_infos["has_think_end"])
            reward_extra_infos_dict["stopped_by_budget"].extend(trace_extra_infos["stopped_by_budget"])
            reward_extra_infos_dict["peak_token"].extend(current_trace_peak_lengths)
            reward_extra_infos_dict["cumulative_tokens"].extend(current_trace_cumulative_lengths)
            data_source_lst.append(np.array([trace.metadata.get("data_source", "unknown") for trace in traces], dtype=object))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        data_sources = np.concatenate(data_source_lst, axis=0)
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            for var_name, metric2val in var2metric2val.items():
                if var_name not in {"reward", "acc", "peak_token", "cumulative_tokens"}:
                    continue
                mean_metric_names = [name for name in metric2val.keys() if name.startswith("mean@")]
                if not mean_metric_names:
                    continue
                target_mean_metric = max(mean_metric_names, key=lambda name: int(name.split("@")[-1]))
                for metric_name, metric_val in metric2val.items():
                    if metric_name == target_mean_metric:
                        metric_dict[f"val-core/{data_source}/{var_name}/{metric_name}"] = metric_val

        metric_dict["trace_scores/val_deletethink"] = sum(sample_scores) / len(sample_scores)
        metric_dict["trace_scores/val_has_think_end"] = sum(reward_extra_infos_dict["has_think_end"]) / len(reward_extra_infos_dict["has_think_end"])
        metric_dict["trace_scores/val_stopped_by_budget"] = (
            sum(reward_extra_infos_dict["stopped_by_budget"]) / len(reward_extra_infos_dict["stopped_by_budget"])
            if len(reward_extra_infos_dict["stopped_by_budget"]) > 0
            else 0.0
        )
        metric_dict["trace_length/val_cumulative_tokens_avg"] = sum(trace_cumulative_lengths) / len(trace_cumulative_lengths)
        metric_dict["trace_length/val_peak_tokens_avg"] = sum(trace_peak_lengths) / len(trace_peak_lengths)
        return metric_dict

    def fit(self):
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate_deletethink()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = self.global_steps in self.config.trainer.profile_steps if self.config.trainer.profile_steps is not None else False
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()

                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                batch.non_tensor_batch["sid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                rollout_inputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.batch["input_ids"]]
                gen_batch = self.format_gen_batch(batch)
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        traces = self._rollout_traces(gen_batch, is_validation=False)

                    with marked_timer("reward", timing_raw, color="yellow"):
                        trace_scores, reward_extra_infos_dict = self._score_traces(traces, self.reward_fn)
                        self._assign_trace_advantages(traces)
                        train_batch = self._build_train_batch(traces)

                    train_batch.batch["response_mask"] = compute_response_mask(train_batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(train_batch, metrics=metrics, logging_prefix="deletethink_seqlen")
                    train_batch.meta_info["global_token_num"] = torch.sum(train_batch.batch["attention_mask"], dim=-1).tolist()

                    metrics.update(
                        {
                            "trace_scores/train_deletethink": sum(trace_scores) / len(trace_scores),
                            "trace_scores/has_think_end": sum(reward_extra_infos_dict["has_think_end"]) / len(reward_extra_infos_dict["has_think_end"]),
                            "trace_scores/stopped_by_budget": sum(reward_extra_infos_dict["stopped_by_budget"]) / len(reward_extra_infos_dict["stopped_by_budget"]),
                            "trace_scores/avg_chunks": sum(len(trace.chunks) for trace in traces) / len(traces),
                            "trace_length/train_cumulative_tokens_avg": sum(self._trace_cumulative_lengths(traces)) / len(traces),
                            "trace_length/train_peak_tokens_avg": sum(self._trace_peak_token_lengths(traces)) / len(traces),
                        }
                    )

                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(train_batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = train_batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        metrics.update({"actor/deletethink_entropy": entropy_agg.detach().item()})
                        old_log_prob.batch.pop("entropys")
                        train_batch = train_batch.union(old_log_prob)

                        if "rollout_log_probs" in train_batch.batch.keys():
                            rollout_old_log_probs = train_batch.batch["rollout_log_probs"]
                            actor_old_log_probs = train_batch.batch["old_log_probs"]
                            response_mask = train_batch.batch["attention_mask"][:, -train_batch.batch["responses"].size(1) :]
                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            metrics.update(
                                {
                                    "training/deletethink_rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
                                    "training/deletethink_rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
                                    "training/deletethink_rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(train_batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(train_batch)
                            train_batch = train_batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(train_batch)
                            train_batch = train_batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        if self.config.algorithm.use_kl_in_reward:
                            train_batch, kl_metrics = apply_kl_penalty(train_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(train_batch)
                        metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            train_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(train_batch)
                        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            self._dump_generations(
                                inputs=rollout_inputs,
                                outputs=[trace.full_output_text for trace in traces],
                                scores=trace_scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics = self._validate_deletethink()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    esi_close_to_expiration = should_save_ckpt_esi(max_steps_duration=self.max_steps_duration, redundant_time=self.config.trainer.esi_redundant_time)
                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(compute_data_metrics(batch=train_batch, use_critic=self.use_critic, prefix="deletethink"))
                metrics.update(compute_timing_metrics(batch=train_batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=train_batch, timing_raw=timing_raw, n_gpus=self.resource_pool_manager.get_n_gpus(), prefix="deletethink"))
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
