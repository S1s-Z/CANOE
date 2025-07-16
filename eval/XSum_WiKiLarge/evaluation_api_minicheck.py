import re
import ast
import string
import json
import re
import argparse
from tqdm import tqdm
import os
import torch
from transformers import AutoTokenizer, AutoModel, StoppingCriteria, StoppingCriteriaList, AutoModelForCausalLM
import logging
from datasets import load_dataset
import os
import time
import requests  
import json  
import numpy as np
import os
import nltk
# nltk.download('punkt_tab')
nltk.data.path.append("/root/nltk_data/")
from minicheck.minicheck import MiniCheck
scorer = MiniCheck(model_name='flan-t5-large', cache_dir='./ckpts', enable_prefix_caching=False)



def query_claude(full_prompt, llm_model_name='claude-3-7-sonnet-20250219'):

    # 替换成你的 OpenAI API 密钥  
    api_key = ' '  

    # 设置请求头  
    headers = {  
        'Content-Type': 'application/json',  
        'Authorization': f'Bearer {api_key}'  
    }  

    # 构建请求数据  
    data = {  
        "model": llm_model_name,  
        "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": full_prompt}],
        "max_tokens": 1000 
    }  

    # 发送 POST 请求  
    retries = 0
    max_retries = 5
    while retries < max_retries:
        response = requests.post(  
            'https://o.api.jisuancloud.cn/v1/chat/completions',  
            headers=headers,  
            data=json.dumps(data)  
        )  

        # 判断请求是否成功  
        if response.status_code == 200:  
            # 解析并打印回复  
            reply = response.json()  
            response_text = reply['choices'][0]['message']['content'].strip()
            # print(response)  
            return response_text 
        else:  
            print("Request failed:", response.status_code, response.text)  

    return response_text

def query_claude_think(full_prompt, llm_model_name='claude-3-7-sonnet-20250219'):

    # 替换成你的 OpenAI API 密钥  
    api_key = ' '  

    # 设置请求头  
    headers = {  
        'Content-Type': 'application/json',  
        'Authorization': f'Bearer {api_key}'  
    }  

    # 构建请求数据  
    # print(llm_model_name)
    data = {  
        "model": llm_model_name,  
        "thinking": {
            "type": "enabled",
        },
        "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": full_prompt}],
        # "max_tokens": 1000 
    }  

    # 发送 POST 请求  
    retries = 0
    max_retries = 5
    while retries < max_retries:
        response = requests.post(  
            'https://o.api.jisuancloud.cn/v1/chat/completions',  
            headers=headers,  
            data=json.dumps(data)  
        )  

        # 判断请求是否成功  
        if response.status_code == 200:  
            # 解析并打印回复  
            reply = response.json()  
            response_text = reply['choices'][0]['message']['content'].strip()
            # print(response)  
            return response_text 
        else:  
            print("Request failed:", response.status_code, response.text)  

    return response_text

def query_gpt(full_prompt, llm_model_name='gpt-4o-mini'):

    # 替换成你的 OpenAI API 密钥  
    api_key = ' '  

    # 设置请求头  
    headers = {  
        'Content-Type': 'application/json',  
        'Authorization': f'Bearer {api_key}'  
    }  

    # 构建请求数据  
    data = {  
        "model": llm_model_name,  
        "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": full_prompt}],
        "max_tokens": 1000  
    }  
    
    if llm_model_name == "o1-2024-12-17":
        data = {  
            "model": llm_model_name,  
            "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": full_prompt}],  
        }  

    # 发送 POST 请求  
    retries = 0
    max_retries = 5
    while retries < max_retries:
        response = requests.post(  
            'https://api.shubiaobiao.cn/v1/chat/completions',  
            headers=headers,  
            data=json.dumps(data)  
        )  

        # 判断请求是否成功  
        if response.status_code == 200:  
            # 解析并打印回复  
            reply = response.json()  
            response_text = reply['choices'][0]['message']['content'].strip()
            # print(response)  
            return response_text 
        else:  
            print("Request failed:", response.status_code, response.text)  

    return response_text


def query_deepseek(full_prompt, llm_model_name='deepseek-chat'):

    # 替换成你的 OpenAI API 密钥   
    api_key = ' ' 

    # 设置请求头  
    headers = {  
        'Content-Type': 'application/json',  
        'Authorization': f'Bearer {api_key}'  
    }  

    # 构建请求数据  
    data = {  
        "model": llm_model_name,  
        "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": full_prompt}],  
        "max_tokens": 1000  
    }  

    # 发送 POST 请求  
    retries = 0
    max_retries = 5
    while retries < max_retries:
        response = requests.post(  
            # 'https://mtu.mtuopenai.xyz/v1/chat/completions',  
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,  
            data=json.dumps(data)  
        )  

        # 判断请求是否成功  
        if response.status_code == 200:  
            # 解析并打印回复  
            reply = response.json()  
            response_text = reply['choices'][0]['message']['content'].strip()
            # print(response)  
            return response_text 
        else:  
            print("Request failed:", response.status_code, response.text)  

    return response_text


class LLamaQaStoppingCriteria(StoppingCriteria):
    def __init__(self, list_token_ids_sequence: list = []):
        self.token_ids_sequences = []
        self.lengths = []
        for token_ids_sequence in list_token_ids_sequence:
            self.token_ids_sequences.append(torch.tensor(token_ids_sequence, dtype=torch.long))
            self.lengths.append(len([token_ids_sequence]))
        
    # @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # check the final {self.length} tokens
        stop = False
        for token_ids_sequence, length in zip(self.token_ids_sequences, self.lengths):
            if input_ids.shape[-1] < length:
                continue
            else:
                if bool(torch.all(input_ids[0, -length:] == token_ids_sequence.to(input_ids.device))):
                    stop = True
                    break
        return stop

def set_stop_words(tokenizer, stop):
    stop_words = stop
    list_stop_word_ids = []
    for stop_word in stop_words:
            stop_word_ids = tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
    stopping_criteria = StoppingCriteriaList()
    stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))
    return stopping_criteria
            
def call_llama(model, tokenizer, prompt, stopping_criteria, stop):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    sequences = model.generate(input_ids.cuda(), stopping_criteria = stopping_criteria, max_new_tokens = 1024)[0, input_ids.shape[-1]:]
    decoded = tokenizer.decode(sequences, skip_special_tokens=True)
    for stop_word in stop:
        length_to_remove = len(stop_word)
        if decoded[-length_to_remove:] == stop_word:
            decoded = decoded[:-length_to_remove]
    # import pdb; pdb.set_trace()
    output_str = decoded.strip()
    return output_str


def qa_to_prompt_baseline(context, schema):
    # print(schema)
    def get_prompt(context, schema, answer=''):
        if schema == 'sum':
            template = open("./prompt_sum.txt").read()
            prompt = template.format(context)
        elif schema == 'sim':
            template = open("./prompt_sim.txt").read()
            prompt = template.format(context)      
        elif schema == 'dialog':
            template = open("./prompt_dialog.txt").read()
            prompt = template.format(context)         

        return prompt
    
    prompt = ''
    prompt = prompt + get_prompt(context, schema=schema)
    return prompt

    
def create_log_path(log_path):
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('') 
        logging.info(f"Log file {log_path} created.")
    else:
        logging.info(f"Log file {log_path} already exists.")



def main():
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="./Models/Qwen2-7B-Instruct", type=str)
    parser.add_argument("--schema", default="sum", type=str, help="Choose from the following prompting templates: sum, sim")
    parser.add_argument("--data_path", default="./data/summarization.json", type=str)
    parser.add_argument("--output_path", default='./result/Qwen2-7B-Instruct.json', type=str)
    parser.add_argument("--log_path", default='./log_ConFiQA/Qwen2-7B-Instruct.log', type=str)
    args = parser.parse_args()
    model_name = args.model_name

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_path),  # 写入日志文件
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logging.info("Evaluate Context-Faithfulness for the Model: %s" % model_name)
    

    with open(args.data_path, 'r') as fh:
        data = json.load(fh)
    logging.info('Loaded {} instances.'.format(len(data)))

    create_log_path(args.log_path)
    
    all_gold_answers, pred_answers = [], []
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage = True, torch_dtype=torch.float16, trust_remote_code=True)
    # model.cuda()
    stop = []
    # stopping_criteria = set_stop_words(tokenizer, stop)
    step = 0

    passages = []
    summaries = []

    final_score = 0

    if args.schema == 'sum':
        data = data[:1000]
        
    for d in tqdm(data):
        step += 1
        
        passage = d["text"]
        passages.append(passage)
        # print(passage)
        prompt = qa_to_prompt_baseline(passage, schema=args.schema)
        # print(prompt)

        if args.model_name == "claude-3-7-sonnet-20250219":
            pred = query_claude(prompt, llm_model_name=args.model_name)
        elif args.model_name == "claude-3-7-sonnet-20250219-think":
            pred = query_claude_think(prompt, llm_model_name="claude-3-7-sonnet-20250219")
        elif args.model_name == "deepseek-chat":
            pred = query_deepseek(prompt, llm_model_name=args.model_name)
            # print(pred)
        elif args.model_name == "deepseek-reasoner":
            pred = query_deepseek(prompt, llm_model_name=args.model_name)
            # print(pred)
        else:
            pred = query_gpt(prompt, llm_model_name=args.model_name)

        d['pred'] = pred

        # score_conv = model_conv.score([passage], [pred])
        # final_score += float(score_conv["scores"][0])

        pred_label, raw_prob, _, _ = scorer.score(docs=[passage], claims=[pred])
        final_score += float(pred_label[0])

        summaries.append(pred)

        if step % 50 == 0:
            logging.info('Score: {}.'.format(final_score/step))
            with open(args.output_path, 'w') as fh:
                json.dump(data, fh)
            

    with open(args.output_path, 'w') as fh:
        json.dump(data, fh)
        
    logging.info('Final score: {}.'.format(final_score/len(data)))
    

if __name__ == '__main__':
    main()
