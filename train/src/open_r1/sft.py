# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
"""

import logging
import os
import sys

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
from open_r1.configs import SFTConfig
from open_r1.utils.callbacks import get_callbacks
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from qwen_vl_utils import process_vision_info
logger = logging.getLogger(__name__)


processor = None


def convert_example(example):
    """
    将原始 jsonl 格式转换为内部统一的 "messages" 格式。
    输入样例格式例如：
    {
      "system": "You are a helpful assistant.",
      "conversations": [
          {"from": "user", "value": "How many objects are included in this image?",
           "image_path": "/path/to/image.png"},
          {"from": "assistant", "value": "<|begin_of_thought|>\nI can see 10 objects\n<|end_of_thought|>\n<|begin_of_solution|>\n10\n<|end_of_solution|>"}
      ]
    }
    转换后会生成 "messages" 字段，其格式与 Qwen2-VL 期望的格式一致。
    """
    messages = []
    # 如果存在 system 消息，加入 system 消息
    if "system" in example:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": example["system"]}],
        })
    # 遍历 conversations 字段，逐条转换
    for conv in example.get("conversations", []):
        role = conv.get("from", "")
        content = []
        # 如果有图片，则添加图片项
        if "image_path" in conv and conv["image_path"] is not None:
            image_path = conv["image_path"]
            # 确保路径以 "file://" 开头
            if not image_path.startswith("file://"):
                image_path = "file://" + image_path
            content.append({"type": "image", "image": image_path})
        # 如果有文本，则添加文本项
        if "value" in conv:
            content.append({"type": "text", "text": conv["value"]})
        messages.append({
            "role": role,
            "content": content,
        })
    example["messages"] = messages
    return example


def collate_fn(examples):
    """
    自定义的 batch collate 函数，支持 Qwen2-VL 的多模态输入。
    主要流程：
      1. 利用 processor.apply_chat_template 对每个样本的 "messages" 格式进行处理，
         同时设置 add_generation_prompt=True 用于生成阶段。
      2. 调用 process_vision_info 提取每个样本中图片及视频信息，分别放入 image_inputs 和 video_inputs。
      3. 利用 processor 将文本和视觉信息转换为模型输入（包括 attention mask 等）。
      4. 复制 input_ids 为 labels，并对 pad token 及 image token 进行 mask（mask 值为 -100）。
    """
    # 使用 processor 生成文本输入，每个样本添加生成提示
    texts = [
        processor.apply_chat_template( convert_example(example)["messages"], tokenize=False, add_generation_prompt=True)
        for example in examples
    ]
    image_inputs = []
    # video_inputs = []
    for example in examples:
        # for each in example["messages"]:
        #     for contents in each["content"]:
        #         if "image" in contents and contents["image"] is None:
                    # contents.pop("image")
        imgs, vids = process_vision_info(example["messages"])
        image_inputs.append(imgs)
        # video_inputs.append(vids)
    # 生成 batch 输入：文本、图片、视频
    batch = processor(
        text=texts,
        images=image_inputs,
        # videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )
    # 构造标签，复制 input_ids
    labels = batch["input_ids"].clone()
    # 将 pad token 位置 mask 成 -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    # 将 image token 位置也 mask 成 -100（假设 processor.image_token 为对应的特殊标识）
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################

    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # Load from jsonl file
    dataset = load_dataset("json", data_files=script_args.dataset_name)
    # for split in dataset.keys():
    #     dataset[split] = dataset[split].map(convert_example)
    ################
    # Load tokenizer
    ################
    global processor
    if "vl" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        logger.info("Using AutoProcessor for vision-language model.")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
        )
        logger.info("Using AutoTokenizer for text-only model.")
    # 设置 pad_token（如果不存在，则用 eos_token 代替）
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # training_args.model_init_kwargs = model_kwargs
    from transformers import Qwen2VLForConditionalGeneration
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    ############################
    # Initialize the SFT Trainer
    ############################
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    training_args.remove_unused_columns = False
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(script_args.dataset_name),
        "dataset_tags": list(script_args.dataset_name),
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
