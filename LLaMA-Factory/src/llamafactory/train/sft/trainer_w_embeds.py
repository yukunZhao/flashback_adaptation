# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_equal_to_4_46
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments

logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

        # zhaoyukun02 新增的2行 匹配 compute_loss 函数
        self.config = self.model.config
        print ('self.model.config pad_token_id', self.model.config, self.model.config.pad_token_id)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

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
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r"""
        Fixes the loss value for transformers 4.46.0.
        https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/trainer.py#L3605
        """
        # loss = super().compute_loss(model, inputs, return_outputs, **kwargs)
        # zhaoyukun compute_Loss 展开
        labels = inputs.pop("labels")
        input_ids = inputs["input_ids"]

        # model 是 DistributedDataParallel，这里直接调用 model.get_input_embeddings() 会报错
        # 使用模型的嵌入层获取 inputs_embeds
        embedding_layer = self.model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids)

        # print ('compute_loss inputs_embeds.shape', inputs_embeds.shape)

        outputs = model(inputs_embeds=inputs_embeds, labels=labels)  # 返回 loss logits
        # print ('compute_loss output', outputs)

        loss = outputs.loss
        # print ('compute_loss loss', loss)
        # logits = logits[0]
        # print ('compute_loss logits2.shape', logits.shape)
        # print ('compute_loss logits3.shape', logits.view(-1, logits.shape[-1]).shape)
        # print ('compute_loss labels.shape', abels.view(-1).shape)
        
        # loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        # if self.args.label_smoothing == 0:
        #     if self.data_args is not None and self.data_args.ignore_pad_token_for_loss:
        #         # force training to ignore pad token
        #         logits = model(**inputs, use_cache=False)[0]
        #         loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        #     else:
        #         # compute usual loss via models
        #         loss, logits = model(**inputs, labels=labels, use_cache=False)[:2]
        # else:
        #     # compute label smoothed loss
        #     logits = model(**inputs, use_cache=False)[0]
        #     lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        #     loss, _ = self.loss_fn(lprobs, labels, self.args.label_smoothing, ignore_index=self.config.pad_token_id)
        #  展开 done
        if is_transformers_version_equal_to_4_46() and not getattr(self, "model_accepts_loss_kwargs", False):
            # other model should not scale the loss
            if return_outputs:
                return (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
            else:
                return loss / self.args.gradient_accumulation_steps

        return loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        #print ('zhaoyukun ', inputs.keys()) #dict_keys(['input_ids', 'attention_mask', 'labels'])
        labels = inputs["labels"] if "labels" in inputs else None
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            labels = labels.detach().clone() if labels is not None else None  # backup labels
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]
        # zhaoyukun 11.26 把prediction_step copy过来重写 代替123行super().prediction_step()逻辑，另外 compute_loss 可能也需要重写
        
        # loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
        #     model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        # )
        embedding_layer = model.get_input_embeddings()  # 获取嵌入层
        inputs_embeds = embedding_layer(inputs["input_ids"])  # 转换为嵌入向量
        # input_embeds = self.model.model.embed_tokens(inputs["input_ids"])
        inputs["input_embeds"] = inputs_embeds
   
        # copy super prediction step here
        inputs = self._prepare_inputs(inputs)

        # gen_kwargs = {
        #     "max_length": self.data_args.val_max_target_length
        #     if self.data_args is not None
        #     else self.config.max_length,
        #     "num_beams": self.data_args.eval_beams if self.data_args is not None else self.config.num_beams,
        # }
        gen_kwargs = {"max_length": 4096, "num_beams": 1}

        # if self.args.predict_with_generate and not self.args.prediction_loss_only:
        # From inputs_embeds -- exact same output if you also pass `input_ids`. If you don't
        # pass `input_ids`, you will get the same generated content but without the prompt
        generated_tokens = self.model.generate(
            inputs_embeds=inputs["input_embeds"], #inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )

        outputs_logits = model(inputs_embeds=inputs_embeds)
        generated_tokens_by_id = self.model.generate(
            inputs["input_ids"], #inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        print ('[zhaoyukun generated_tokens.shape]', generated_tokens.shape)
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        print ('[zhaoyukun inputs_id.shape]', inputs["input_ids"].shape) # torch.Size([1, 104])
        print ('[zhaoyukun label.shape]', inputs["labels"].shape) # torch.Size([1, 104])
        print ('[zhaoyukun generated_tokens_by_id.shape]', generated_tokens_by_id.shape) # torch.Size([1, 123])


        #labels = inputs.pop("labels")
        with torch.no_grad():
            # compute loss on predict data
            #loss, logits = self._compute_loss(model, inputs, labels)
            loss = self.compute_loss(model, inputs)

        loss = loss.mean().detach()
        # if self.args.prediction_loss_only:
        #     return (loss, None, None)

        logits = generated_tokens # if self.args.predict_with_generate else logits

        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        # copy done

        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: "torch.Tensor", tgt_tensor: "torch.Tensor") -> "torch.Tensor":
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))

            writer.write("\n".join(res))
