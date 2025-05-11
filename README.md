# CANOE


The code of our paper "Teaching Language Models to Remain Context-Faithful via Synthetic Tasks and Reinforcement Learning".

## 🛶 Overview

Teaching large language models (LLMs) to be faithful in the provided context is crucial for building reliable information-seeking systems. Therefore, we propose a systematic framework **CANOE** to improve the faithfulness of LLMs in both short-form and long-form generation tasks without human annotations. Specifically, we first synthesize short-form question-answering (QA) data with four diverse tasks to construct high-quality and easily verifiable training data without human annotation. Also, we propose Dual-GRPO, a rule-based reinforcement learning method that includes three tailored rule-based rewards derived from synthesized short-form QA data, while simultaneously optimizing both short-form and long-form response generation. Notably, Dual-GRPO eliminates the need to manually label preference data to train reward models and avoids over-optimizing short-form generation when relying only on the synthesized short-form QA data. Experimental results show that **CANOE** greatly improves the faithfulness of LLMs across 11 different downstream tasks, even outperforming the most advanced LLMs, e.g., GPT-4o and OpenAI o1.



## 🎯 Usage

#### 🔎 Setup


Our code is based on open-r1, please follow the repo to get the right environments according to train/setup.py in our repo and [ReadMe](https://github.com/huggingface/open-r1) in open-r1 repo.

Meanwhile, we rewrite the TRL package, plz install our TRL by, 


```python
cd train/TRL

pip install -e .[dev]
```


### 📢 Train

You can find the corresponding code in `train`.

You can download and save the processed data through the [Tsinghua Drive/training_data/](https://cloud.tsinghua.edu.cn/d/38ed09b657584c01ae29/) to train the model. Please put the data into `train/train_data`. 


We provide training scripts under `/train`, e.g., llama_8b_10k_2epoch.sh and qwen_7b_10k_2epoch.sh.



## 🎲 Evaluation

You can download and save the processed data through the [Tsinghua Drive/datasets](https://cloud.tsinghua.edu.cn/d/38ed09b657584c01ae29/). Plz put the correct dataset files in the correct path, e.g., `eval/CNQ/dataset`.

### 🔍 ConFiQA & FiQA

There are two short-form QA tasks, including both the counterfactual QA and factual QA tasks.

```sh
sh eval.sh

sh eval_factual.sh
```

We also provide the scripts for API-based LLMs.

```sh
sh eval_api.sh

sh eval_factual_api.sh
```

### 🔍 CNQ

This is a short-form counterfactual multiple-choice questions task.

```sh
sh eval.sh
```

We also provide the scripts for API-based LLMs.
```sh
sh eval_api.sh
```

### 🔍 FaithEval

This is a short-form counterfactual QA task.

```sh
sh eval.sh
```

We also provide the scripts for API-based LLMs.
```sh
sh eval_api.sh
```

### 🔍 FollowRAG

They are short-form open-domain QA tasks for RAG generation.

```sh
sh eval.sh
```

We also provide the scripts for API-based LLMs.
```sh
sh eval_api.sh
```

### 🔍 CLAPNQ

This is a long-form QA task. Please install [MiniCheck-Flan-T5-Large](https://github.com/Liyan06/MiniCheck) for the correct evaluation. MiniCheck-Flan-T5-Large can be saved in `eval/CLAPNQ/ckpts`

```sh
sh eval.sh
```

We also provide the scripts for API-based LLMs.
```sh
sh eval_api.sh
```


### 🔍 XSum & WiKiLarge

They are two long-form tasks, including simplification and summarization.
Please install [MiniCheck-Flan-T5-Large](https://github.com/Liyan06/MiniCheck) for the correct evaluation. MiniCheck-Flan-T5-Large can be saved in `eval/CLAPNQ/ckpts`

```sh
sh eval_sim.sh

sh eval_sum.sh
```

We also provide the scripts for API-based LLMs.

```sh
sh eval_sim_api.sh

sh eval_sum_api.sh
```


## 🤖 All available models

Here is the full list of models we released:

|Model|Link|Description|
|---|---|---|
|**CANOE-LLaMA3-8B**| [🤗 HF](https://huggingface.co/ssz1111/CANOE-LLaMA3-8B) | Chat model, training based on LLaMA3-Instruct-8B. |
|**CANOE-Qwen2.5-7B**| [🤗 HF](https://huggingface.co/ssz1111/CANOE-Qwen2.5-7B) | Chat model, training based on Qwen2.5-Instruct-7B. |
|**CANOE-Qwen2.5-14B**| [🤗 HF](https://huggingface.co/ssz1111/CANOE-Qwen2.5-14B) | Chat model, training based on Qwen2.5-Instruct-7B. |


