# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from datetime import datetime
import warnings
import torch.nn as nn
from einops import rearrange
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union
from scipy.stats import wasserstein_distance

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_equal_to_4_46
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.f_divergence_type = "reverse_kl"
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma

        # ADDED
        self.embeds_state_dict = kwargs.pop("embeds_state_dict", None)
        self.model_config = model.config
        self.retriever = kwargs.pop("retriever", None)
        self.act_tokenizer = kwargs.pop("act_tokenizer", None)

        self.keys = kwargs.pop("retriever_keys", None)
        self.weight_offset = kwargs.pop("retriever_weight_offset", None)

        # ADDED 1.10
        self.pool_size = finetuning_args.pool_size
        self.group_nums = finetuning_args.group_nums
        self.top_k_JL = finetuning_args.top_k_JL

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.callback_handler.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def get_batch_samples(self, epoch_iterator, num_batches):
        r"""
        Replaces the method of KTO Trainer with the one of the standard Trainer.
        """
        return Trainer.get_batch_samples(self, epoch_iterator, num_batches)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes ORPO's odds ratio (OR) loss for batched log probabilities of the policy model.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes SimPO loss for batched log probabilities of the policy model.
        """
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes loss for preference learning.
        """
        if not self.finetuning_args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
            else:
                raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")

            chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        else: # 用这里的loss, sigmoid
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )

        return losses, chosen_rewards, rejected_rewards

    # @override
    # def concatenated_forward(
    #     self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    # ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    #     r"""
    #     Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

    #     Otherwise the average log probabilities.
    #     """
    #     if self.finetuning_args.use_ref_model:
    #         batch = {k: v.detach().clone() for k, v in batch.items()}  # avoid error

    #     all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits #.to(torch.float32)
    #     all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
    #     if self.loss_type in ["ipo", "orpo", "simpo"]:
    #         all_logps = all_logps / valid_length

    #     # 这里增加一次 求生成结果的
    #     print ('[concatenated_forward batch.keys()]', batch.keys()) # dict_keys(['input_ids', 'attention_mask', 'labels'])
    #     # AttributeError: 'DistributedDataParallel' object has no attribute 'get_input_embeddings'
    #     embedding_layer = self.ref_model.get_input_embeddings()
    #     inputs_embeds = embedding_layer(batch["input_ids"]) #  inputs_embeds: shape torch.Size([1, 344, 5120])
    #     logtis_w_embeds = model(inputs_embeds=inputs_embeds, labels=batch['labels'], return_dict=True, use_cache=False).logits #.to(torch.float32)  # 返回 loss logits
    #     #  attention_mask=prompt_attention_mask,
    #     # print ('[concatenated_forward logits compare]', all_logits.shape, logtis_w_embeds.shape) #torch.Size([1, 344, 32000]) torch.Size([1, 344, 32000])
    #     all_logps_ref, valid_length_ref = get_batch_logps(logits=logtis_w_embeds, labels=batch["labels"])

    #     chosen_logps, rejected_logps = all_logps, all_logps_ref
    #     chosen_logits, rejected_logits = all_logits, logtis_w_embeds
    #     chosen_length = valid_length

    #     return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_length

    def get_topk_weight_offset(self, queries, group_idx, weight_topk=2, groups=4, pool_size=12, num_layers=1,low_rank=8, use_distance_weight=True):
        # self.pool_size = finetuning_args.pool_size
        # self.group_nums = finetuning_args.group_nums
        # self.top_k_JL = finetuning_args.top_k_JL
        # ADDED 1.10
        weight_topk = self.top_k_JL
        pool_size = self.pool_size
        
        bsz = queries.shape[0]
        group_idx = group_idx.cpu()
        queries = rearrange(queries, "b (g c) -> b g c", g=groups).cpu()
        keys = self.keys.repeat(bsz, 1, 1, 1) #.to(self.accelerator.device) # bsz, groups, pool_size, key_hidden_size=768/6
        outputs = dict()
        #print ('[self.keys, keys shape]', self.keys.shape, keys.shape) # [batch_size, groups, pool_size, key_hidden_size]

        queries = queries.unsqueeze(2).repeat(1, 1, pool_size, 1)
        sim = F.cosine_similarity(queries, keys, dim=-1)  # [bsz, groups, pool_size]


        idx_sim = sim.clone().detach()
        if not use_distance_weight:
            _, idx = idx_sim.topk(weight_topk, dim=-1)  # [bsz, group, topk]
            idx_vote = rearrange(idx, "b g k -> g (b k)")
            base = (torch.arange(0, groups, device=idx_vote.device) * pool_size).view(-1, 1)
            idx_vote = (base + idx_vote).flatten()
            bin_count = torch.bincount(idx_vote, minlength=pool_size*groups).view(groups, pool_size)
            idx_vote = torch.topk(bin_count, k=weight_topk)[1]  # [groups, topk]
        else:
            idx_sim = torch.mean(idx_sim, dim=[0,1]) # original code, batch_size=1 
            #idx_sim = torch.mean(idx_sim, dim=[1]) #，[batch_size, pool_size]
            dis_weihgt, idx_vote = idx_sim.topk(weight_topk, dim=-1) # [topk]
            dis_weihgt = dis_weihgt / (dis_weihgt.sum() + 1e-9)


        #weight_offset = torch.take_along_dim(self.weight_offset, idx_vote[:, None,None], dim=1)
        weight_offset = self.weight_offset[group_idx][:, idx_vote].squeeze(0) #.to(self.accelerator.device)
         #self.model_config.num_hidden_layers
 
        low_rank_a = weight_offset[..., 0,:].view(weight_topk, num_layers, low_rank, self.model_config.hidden_size)
        low_rank_b = weight_offset[..., 1,:].view(weight_topk, num_layers, low_rank, self.model_config.hidden_size)
        
        weight_offset = torch.einsum("n l r x, n l r y -> n l x y", low_rank_a, low_rank_b) # [2, num_hidden_layers, hidden_size, hidden_size]


        if not use_distance_weight:
            weight_offset = torch.mean(weight_offset, dim=0)
        else:
            #weight_offset = torch.mean(weight_offset, dim=0)
            weight_offset = (dis_weihgt[:, None, None, None] * weight_offset).sum(0)  # 

        outputs['weight_offset'] = weight_offset # [num_hidden_layers, hidden_size, hidden_size]
        
        return outputs
    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Computes log probabilities of the reference model.
        """
        if not self.finetuning_args.use_ref_model:
            return None, None

        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            reference_chosen_logps, reference_rejected_logps, *_ = self.concatenated_forward(ref_model, batch)

        return reference_chosen_logps, reference_rejected_logps


    def forward_with_inputs_embeds(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"], from_ref_model=False, inputs_embeds=None
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.
        batch: dict_keys(['input_ids', 'attention_mask', 'labels'])
        Otherwise the average log probabilities.
        """

        prompt_len, label_len = batch["input_ids"].size(-1), batch["labels"].size(-1)
        if self.finetuning_args.use_ref_model:
            batch = {k: v.detach().clone() for k, v in batch.items()}  # avoid error

        
        # augment
        retriever_outputs = self.get_topk_weight_offset(batch["input_embeds"], batch["group_id"] )
        weight_offset = retriever_outputs['weight_offset']
        all_logits = model(**batch, return_dict=True, use_cache=False, weight_offset=weight_offset).logits
        # no augment, 1.8 codes
        #all_logits = model(**batch, return_dict=True, use_cache=False).logits
        if prompt_len - label_len > 0:
            clean_all_logits = all_logits[:, prompt_len - label_len :, :]
        else:
            clean_all_logits = all_logits
        
        clean_all_logits_log = clean_all_logits.log_softmax(-1)

        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        return all_logps, clean_all_logits_log, valid_length

        # if self.loss_type in ["ipo", "orpo", "simpo"]:
        #     all_logps = all_logps / valid_length

        #print (f"[debug {from_ref_model} batch_has_label={batch_has_label} \n prompt_len={prompt_len}, valid_label_len={valid_label_len}, valid_label_len_min={valid_label_len_min},\n weight_offset={weight_offset}, all_logits.shape={all_logits.shape}, value={all_logits} \n clean_all_logits.shape={clean_all_logits.shape}, value={clean_all_logits} \n clean_all_logits_log.shape={clean_all_logits_log.shape}, value={clean_all_logits_log} ]")



    def compute_reference_logits(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Computes log probabilities of the reference model.
        """
        if not self.finetuning_args.use_ref_model:
            return None, None

        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            all_logps, all_logits, valid_length = self.forward_with_inputs_embeds(ref_model, batch, True)

        return all_logps, all_logits, valid_length
        

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """
        # 这里不用ref_model
        metrics = {}

        ref_loss_mask = batch["has_label"] == 0
        sft_loss_mask = batch["has_label"] == 1
        # sft_loss = torch.tensor(0.0).to(self.accelerator.device)
        # ref_loss = torch.tensor(0.0).to(self.accelerator.device)

        # speed_has_label = torch.tensor([1]).to(self.accelerator.device)

        policy_chosen_logps, policy_chosen_logits, valid_length  = self.forward_with_inputs_embeds(model, batch)
        ref_chosen_logps, ref_chosen_logits, ref_valid_length = self.compute_reference_logits(model, batch)
        
        sft_loss = -policy_chosen_logps/valid_length
        KL_pq = torch.sum(torch.exp(ref_chosen_logits) * (ref_chosen_logits - policy_chosen_logits)) 
        KL_qp = torch.sum(torch.exp(policy_chosen_logits) * (policy_chosen_logits - ref_chosen_logits))
        ref_loss = (KL_pq + KL_pq) / 1.0

        #     if False: # wasserstein
        #         wasserstein_matrix = torch.zeros(bsz, device=self.accelerator.device)
        #         policy_chosen_logits_detach = policy_chosen_logits.detach().clone().cpu()
        #         ref_chosen_logits_detach = ref_chosen_logits.detach().clone().cpu()
        #         for b in range(bsz): 
        #             wasserstein_matrix[b] = wasserstein_distance(policy_chosen_logits_detach[b, :, :].numpy().flatten(), ref_chosen_logits_detach[b, :, :].numpy().flatten())
        #         ref_loss = -wasserstein_matrix
        #     else:
        #         KL_pq = torch.sum(torch.exp(ref_chosen_logits) * (ref_chosen_logits - policy_chosen_logits)) 
        #         KL_qp = torch.sum(torch.exp(policy_chosen_logits) * (policy_chosen_logits - ref_chosen_logits))
        #         ref_loss = (KL_pq + KL_pq) / 2.0 / self.model_config.vocab_size
            
            
        #     print ('[] ref_loss.requires_grad', ref_loss.requires_grad, ref_loss, policy_chosen_logits.requires_grad, ref_chosen_logits.requires_grad )
        ref_loss_weight = self.beta
        
        ref_loss = ref_loss * ref_loss_weight
        losses = ref_loss_mask * ref_loss + sft_loss_mask * sft_loss
        #print (f"[zhaoykun debug loss] ref_loss_weight={ref_loss_weight}, ref_loss_mask={ref_loss_mask}, sft_loss_mask={sft_loss_mask}, ref_loss={ref_loss}, sft_loss={sft_loss},losses={losses} ")


        def compute_ref_loss(chosen_logps, ref_chosen_logps):
            device = self.accelerator.device
            # Get the log ratios for the chosen and rejected responses
            logratios = chosen_logps.to(device) - ref_chosen_logps.to(device)
            # beta = pref_beta = 0.1
            losses = -F.logsigmoid(self.beta * logratios)
            return losses
        # loss 1
        # loss 2
        # losses =  -ref_chosen_logps / ref_valid_length

        # # 这里几个loss 都可以试试
        
        # losses += sft_loss
        sft_loss_new = torch.tensor(losses)
        
        # if self.ftx_gamma > 1e-6:
        #     losses += self.ftx_gamma * sft_loss

        prefix = "eval_" if train_eval == "eval" else ""
        # metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        # metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        # metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
        # metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        # metrics[f"{prefix}logps/rejected"] = policy_chosen_logps.mean().item()
        # metrics[f"{prefix}logps/chosen"] = ref_chosen_logps.mean().item()
        # metrics[f"{prefix}logits/rejected"] = policy_chosen_logits.mean().item()
        # metrics[f"{prefix}logits/chosen"] = ref_chosen_logits.mean().item()

        metrics[f"{prefix}loss"] = sft_loss_new.mean().item() # nan
        if self.loss_type == "orpo":
            metrics[f"{prefix}sft_loss"] = sft_loss_new.mean().item()
            metrics[f"{prefix}odds_ratio_loss"] = ((losses - sft_loss_new) / self.beta).mean().item()

        # print ('[final loss]', losses)
        return losses.mean(), metrics

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r"""
        Fixes the loss value for transformers 4.46.0.
        https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/trainer.py#L3605
        """
        loss = super().compute_loss(model, inputs, return_outputs)
        if is_transformers_version_equal_to_4_46() and kwargs.pop("num_items_in_batch", False):
            if return_outputs:
                return (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
            else:
                return loss / self.args.gradient_accumulation_steps

        return loss

    @override
    def log(self, logs: Dict[str, float]) -> None:
        r"""
        Log `logs` on the various objects watching training, including stored metrics.
        """
        # logs either has "loss" or "eval_loss"
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        key_list, metric_list = [], []
        for key, metrics in self._stored_metrics[train_eval].items():
            key_list.append(key)
            metric_list.append(torch.tensor(metrics, dtype=torch.float).to(self.accelerator.device).mean().item())

        del self._stored_metrics[train_eval]
        if len(metric_list) < 10:  # pad to for all reduce
            for i in range(10 - len(metric_list)):
                key_list.append(f"dummy_{i}")
                metric_list.append(0.0)

        metric_list = torch.tensor(metric_list, dtype=torch.float).to(self.accelerator.device)
        metric_list = self.accelerator.reduce(metric_list, "mean").tolist()
        for key, metric in zip(key_list, metric_list):  # add remaining items
            if not key.startswith("dummy_"):
                logs[key] = metric

        return Trainer.log(self, logs)
