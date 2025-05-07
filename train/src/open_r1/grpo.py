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
import sys
print(sys.executable)
import logging
from datetime import datetime
import os
import re
import sys
from dataclasses import dataclass, field

import datasets
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from open_r1.configs import GRPOConfig
from open_r1.utils.callbacks import get_callbacks
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config


logger = logging.getLogger(__name__)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "influence", "length"],
        metadata={"help": "List of reward functions. "},
    )
    # do_long_answer: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to include long answer in the reward function"},
    # )


def accuracy_reward(completions, solution, **kwargs):
    """
    Reward function that checks if the completion is correct using either symbolic verification or exact string matching.
    
    1. 尝试符号验证：解析生成答案和正确答案，若验证成功则直接奖励 1.0。
    2. 如果符号验证失败，则从 <answer> 标签中提取答案部分进行字符串精确比较。
    3. 在 DEBUG_MODE 环境变量为 "true" 时，将每次的验证信息写入日志文件。
    
    Args:
        completions (list): 模型生成结果列表，每个元素预期为一个字典列表，其中第一个元素包含 "content" 键。
        solution (list): 正确答案列表，每个元素为一个字符串，可能包含 <answer> 标签。
    
    Returns:
        list: 每个样本的奖励分数列表，1.0 表示正确，0.0 表示不正确。
    """
    # 从生成结果中提取文本内容
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        
        # 1. 尝试符号验证
        try:
            # 解析生成答案和正确答案
            answer = parse(content)
            sol_parsed = parse(sol)
            if float(verify(answer, sol_parsed)) > 0:
                reward = 1.0
        except Exception:
            pass  # 如果符号验证失败，则进入字符串匹配
        
        # 2. 如果符号验证失败，则尝试字符串匹配
        if reward == 0.0:
            try:
                # 尝试从正确答案中提取 <answer> 标签内的内容，否则使用整个字符串（去除前后空白）
                sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # 尝试从生成答案中提取 <answer> 标签内的内容，否则使用整个字符串（去除前后空白）
                content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # 如果提取后的答案完全相同，则奖励 1.0
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # 保持 reward 为 0.0
        
        rewards.append(reward)
        
        # 3. 若开启调试模式，则将验证信息写入日志
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "accuracy_reward.log")
            try:
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n\n")
            except Exception as e:
                print(f"Logging failed: {e}")
                
    return rewards

def influence_reward(completions, solution, completions_long_answer, **kwargs):
    # 从生成结果中提取文本内容
    
    contents = [
        completion_long_answer[0]["content"] if completion_long_answer is not None else None
        for completion_long_answer in completions_long_answer
    ]

    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        if content is None:
            rewards.append(reward)
            continue
        try:
            # 尝试从正确答案中提取 <answer> 标签内的内容，否则使用整个字符串（去除前后空白）
            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            
            # 尝试从生成答案中提取 <answer> 标签内的内容，否则使用整个字符串（去除前后空白）
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            student_answer = content_match.group(1).strip() if content_match else content.strip()
            
            # 如果提取后的答案完全相同，则奖励 1.0
            if student_answer == ground_truth:
                reward = 1.0
        except Exception:
            pass  # 保持 reward 为 0.0
        
        rewards.append(reward)
        
        # 3. 若开启调试模式，则将验证信息写入日志
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "influence_accuracy_reward.log")
            try:
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} influence_Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n\n")
            except Exception as e:
                print(f"Logging failed: {e}")
                
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<long_answer>.*?</long_answer>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def len_reward(completions, **kwargs):
    """
    保证long answer 内的长度在 <low, high> * context_len 之间
    """
    contents = [completion[0]["content"] for completion in completions]
    # print(prompts)
    # print(kwargs.keys())
    contexts = [context for context in kwargs['problem']]
    # print('prompts',kwargs['prompts'])
    # print('problem',kwargs['problem'])
    rewards = []

    min_threshold = 0.2
    max_threshold = 0.8
    # max_threshold = 0.6

    
    for content, context in zip(contents, contexts):
        context_pattern = r'<context>(.*?)</context>'
        long_answer_pattern = r'<long_answer>(.*?)</long_answer>'

        match_context = re.search(context_pattern, context)
        match_long_answer = re.search(long_answer_pattern, content)
        # print(content)
        if match_context and match_long_answer:
            context_str = match_context.group(1).strip()
            long_answer_str = match_long_answer.group(1).strip()
            # print(context_str)
            # print(long_answer_str)
            if (len(long_answer_str) > min_threshold * len(context_str)) and (len(long_answer_str) < max_threshold * len(context_str)):
                reward = 1
            else:
                reward = 0
        else:
            reward = 0
        rewards.append(reward)

    return rewards

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "influence": influence_reward,
    "length": len_reward,
}

# SYSTEM_PROMPT = (
#     "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
#     "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
#     "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
#     "<think> reasoning process here </think><answer> answer here </answer>"
# )
# SYSTEM_PROMPT = (
#     "A conversation between User and Assistant. The user asks a question that consists of two parts: a context and the actual question, separated by two newline characters (\\n\\n). "
#     "The context is provided within <context> and </context> tags. The Assistant solves the question by first thinking about the reasoning process internally and then providing the answer. "
#     "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."
# )

# SYSTEM_PROMPT = ("A conversation between User and Assistant. The user asks a question that consists of two parts: a context and the actual question, separated by two newline characters (\\n\\n). "
# "The context is provided within <context> and </context> tags. The Assistant solves the question by first thinking about the reasoning process internally and then providing the answer. "
# "The reasoning process, the concise response that consists of syntactically complete and semantically complete sentences to answer the question, "
# "the concise response that directly answer the question are enclosed within <think> </think>, <long_answer> </long_answer>, <short_answer> </short_answer> tags, "
# "respectively, i.e., <think> reasoning process here </think><long_answer> the complete response here </long_answer><short_answer> the concise response here </short_answer>.")


SYSTEM_PROMPT = """
A conversation between User and Assistant. The user gives an instruction that consists of two parts: a passage and the actual instruction, separated by two newline characters (\\n\\n).

The passage is provided within <context> and </context> tags. The Assistant need to refer to the given passage and complete the instruction. 

The Assistant solves the question by first thinking about the reasoning process internally according to the given passage and then providing the response.

The response must be structured and include the following three sections, clearly marked by the respective tags:

- **Reasoning Process**: Explain your thought process or logical steps to derive the answer. Enclose this within <think> and </think> tags.
- **Long Answer**: Provide a long response that consists of syntactically complete and semantically complete sentences to answer the question. Enclose this within <long_answer> and </long_answer> tags.
- **Short Answer**: Present a concise response that directly answer the question. Enclose this within <answer> and </answer> tags.

Format your response exactly as follows:
<think> reasoning process here.</think><long_answer> detailed answer here.</long_answer><answer> the concise answer here.</answer>.
"""
# SYSTEM_PROMPT = """
# A conversation between User and Assistant. The user asks a question that consists of two parts: a context and the actual question, separated by two newline characters (\\n\\n).

# The context is provided within <context> and </context> tags. The Assistant solves the question by first thinking about the reasoning process internally and then providing the answer. 

# The response must be structured and include the following three sections, clearly marked by the respective tags:

# - **Reasoning Process**: Explain your thought process or logical steps to derive the answer. Enclose this within <think> and </think> tags.
# - **Long Answer**: Provide a detailed and complete answer in grammatically correct, semantically coherent sentences. Enclose this within <long_answer> and </long_answer> tags.
# - **Short Answer**: Present a concise, direct answer to the question in as few words as possible while retaining accuracy and clarity. Enclose this within <answer> and </answer> tags.

# Format your response exactly as follows:
# <think> reasoning process here.</think><long_answer> detailed answer here.</long_answer><answer> the concise answer here.</answer>.
# """

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

    # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    dataset = load_dataset("json", data_files=script_args.dataset_name)
    # shuffle dataset
    dataset = dataset.shuffle(seed=training_args.seed)
    if training_args.eval_strategy != "no":
        eval_dataset = load_dataset("json", data_files=script_args.dataset_test_split)

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    dataset = dataset.map(make_conversation)
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    if training_args.eval_strategy != "no":
        eval_dataset = eval_dataset.map(make_conversation)
        for split in eval_dataset:
            if "messages" in eval_dataset[split].column_names:
                eval_dataset[split] = eval_dataset[split].remove_columns("messages")

    #############################
    # Initialize the GRPO trainer
    #############################
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token 
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        processing_class = tokenizer,
        eval_dataset=eval_dataset[script_args.dataset_train_split] if training_args.eval_strategy != "no" else None,
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
        # "finetuned_from": model_args.model_name_or_path,
        # "dataset": list(script_args.dataset_name),
        # "dataset_tags": list(script_args.dataset_name),
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)
    ##########
    # Evaluate
    ##########
    if training_args.do_eval and training_args.eval_strategy != "no":
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset[script_args.dataset_train_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)




if __name__ == "__main__":
    # debugpy debug:
    # local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # if local_rank == 0:
    #     import debugpy
    #     debugpy.listen(("0.0.0.0", 11223))
    #     print("Waiting for debugger attach on rank=0")
    #     debugpy.wait_for_client()
    #     print("Debugger attached on rank=0")

    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
