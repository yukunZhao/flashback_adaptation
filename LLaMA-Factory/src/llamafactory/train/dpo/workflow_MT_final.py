# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/examples/scripts/dpo.py
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

from typing import TYPE_CHECKING, List, Optional
import torch.nn as nn

from ...data import PairwiseDataCollatorWithPadding, SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.ploting import plot_loss
from ...hparams import ModelArguments
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push, create_ref_model
from .trainer_MT_final import CustomDPOTrainer
import torch
from .retriever import Retriever
if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments

def generate_orthogonal_matrix(rows, cols):
    tensor = torch.empty(rows, cols) # .uniform_(-1, 1) 均匀分布
    indexes = list(range(0, rows, cols))
    if cols not in indexes:
        indexes.append(cols)
    for i in range(len(indexes) - 1):
        nn.init.orthogonal_(tensor[indexes[i]: indexes[i+1], :])
    return tensor

def build_keys_and_weights(config, pool_size=12, group_nums=16):


    pool_size = pool_size # 12, 64
    hidden_size = config.hidden_size
    groups = 4
    num_hidden_layers = 1 #config.num_hidden_layers
    bert_model_hidden_size = 768
    low_rank = 8
    group_nums = group_nums

    print (f"[hypter parameters] group_nums={group_nums}, pool_size={pool_size}")

    low_rank_a = torch.zeros(group_nums, pool_size, hidden_size * low_rank * num_hidden_layers)
    low_rank_b = nn.init.normal_(torch.empty(group_nums, pool_size,  hidden_size * low_rank * num_hidden_layers))
    weight_offset = nn.parameter.Parameter(torch.stack([low_rank_a, low_rank_b], dim=-2))  # [pool_size, 2, channels*l*r]
    #weight_offset = torch.stack([low_rank_a, low_rank_b], dim=-2)  # [pool_size, 2, channels*l*r=5120*4*40]

    key_hidden_size = bert_model_hidden_size // groups
    if pool_size > key_hidden_size:
        warnings.warn("The pool size is larger than the key_hidden_size, may cause the generate unstable keys")
    
    keys = [generate_orthogonal_matrix(pool_size, key_hidden_size) for _ in range(groups)]
    keys = torch.stack(keys, dim=0).unsqueeze(0)

    return keys, weight_offset

def run_dpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    #  
    # print (', model config', model.config)
    retriever = None #Retriever(model.config)

    print (f"debug parapameters, pool_size={finetuning_args.pool_size}, group_nums={finetuning_args.group_nums}, top_k_JL={finetuning_args.top_k_JL}")
    retriever_keys, retriever_weight_offset = build_keys_and_weights(model.config, finetuning_args.pool_size, finetuning_args.group_nums)

    # DPO dataset
    # data_collator = PairwiseDataCollatorWithPadding(
    #     template=template,
    #     pad_to_multipRetriever
    #     label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    #     **tokenizer_module,
    # )

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    state_dict = None # torch.load("./saves/vicuna-13b-only-embeds/embeddings/input_embeddings.pth", map_location='cpu')

    # Create reference model
    if finetuning_args.use_ref_model:
        if finetuning_args.ref_model is None and (not training_args.do_train):  # use the model itself
            ref_model = model
        else:
            ref_model = create_ref_model(model_args, finetuning_args, add_valuehead=False, only_load_embeds=False)
            #ref_model = create_ref_model(model_args, finetuning_args, add_valuehead=True)
    else:
        ref_model = None

    # Update arguments
    training_args.remove_unused_columns = False  # important for multimodal and pairwise dataset

    # Initialize our Trainer
    trainer = CustomDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        embeds_state_dict=state_dict,
        retriever=retriever,
        act_tokenizer=tokenizer,
        retriever_keys=retriever_keys, 
        retriever_weight_offset=retriever_weight_offset,
        **dataset_module,
        **tokenizer_module,
        
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "rewards/accuracies"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        if id(model) == id(ref_model):  # unable to compute rewards if reference model is the model itself
            remove_keys = [key for key in metrics.keys() if "rewards" in key]
            for key in remove_keys:
                metrics.pop(key)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
