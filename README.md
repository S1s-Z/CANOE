# CANOE


The code of our paper "Teaching Language Models to Remain Context-Faithful via Synthetic Tasks and Reinforcement Learning"

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

You can download and save the processed data through the [Tsinghua Drive](https://cloud.tsinghua.edu.cn/d/38ed09b657584c01ae29/) to train the model. Please put the data into `train/train_data`. 


We provide training scripts under `/train`, e.g., llama_8b_10k_2epoch.sh and qwen_7b_10k_2epoch.sh.



## 🎲 Evaluation

You can download and save the processed data through the [Tsinghua Drive](https://cloud.tsinghua.edu.cn/d/38ed09b657584c01ae29/)

### 🔍 ConFiQA & FiQA


### 🔍 CNQ


### 🔍 FaithEval


### 🔍 FollowRAG


### 🔍 CLAPNQ


### 🔍 XSum & WiKiLarge



## 🤖 All available models

Here is the full list of models we released:

|Model|Link|Description|
|---|---|---|
|**CANOE-LLaMA3-8B**| [🤗](https://huggingface.co/ssz1111/CANOE-LLaMA3-8B) | Chat model, training based on LLaMA3-Instruct-8B. |
|**CANOE-Qwen2.5-7B**| [🤗](https://huggingface.co/ssz1111/CANOE-Qwen2.5-7B) | Chat model, training based on Qwen2.5-Instruct-7B. |
|**CANOE-Qwen2.5-14B**| [🤗](https://huggingface.co/ssz1111/CANOE-Qwen2.5-14B) | Chat model, training based on Qwen2.5-Instruct-7B. |


