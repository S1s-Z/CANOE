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
import requests
import time


def query_claude(full_prompt, llm_model_name='claude-3-7-sonnet-20250219'):

    # 替换成你的 OpenAI API 密钥  
    api_key = ''  

    # 设置请求头  
    headers = {  
        'Content-Type': 'application/json',  
        'Authorization': f'Bearer {api_key}'  
    }  

    # 构建请求数据  
    data = {  
        "model": llm_model_name,  
        "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": full_prompt}],
        "max_tokens": 150  
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
    api_key = ''  

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
    api_key = ''  

    # 设置请求头  
    headers = {  
        'Content-Type': 'application/json',  
        'Authorization': f'Bearer {api_key}'  
    }  

    # 构建请求数据  
    data = {  
        "model": llm_model_name,  
        "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": full_prompt}],
        "max_tokens": 150  
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
    api_key = ""

    # 设置请求头  
    headers = {  
        'Content-Type': 'application/json',  
        'Authorization': f'Bearer {api_key}'  
    }  

    # 构建请求数据  
    data = {  
        "model": llm_model_name,  
        "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": full_prompt}],  
        "max_tokens": 150  
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


def set_stop_words(tokenizer, stop):
    stop_words = stop
    list_stop_word_ids = []
    for stop_word in stop_words:
            stop_word_ids = tokenizer.encode(stop_word, add_special_tokens=False)[0]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
    stopping_criteria = StoppingCriteriaList()
    stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))
    return stopping_criteria
        

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):

    return (normalize_answer(prediction) == normalize_answer(ground_truth))    

def recall_score(prediction, ground_truth, is_cf):
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    
    contains_negation = any(word in prediction.split() for word in negation_words)
    
    return (ground_truth in prediction) and (not contains_negation if is_cf else True)

def get_score(preds, golds):
    em = 0
    # import pdb; pdb.set_trace()
    # print(preds)
    # print(golds)
    for pred, gold in zip(preds, golds):
        if isinstance(gold, list):
            _em = 0
            for g in gold:
                _em = max(exact_match_score(pred, g), _em)
        else:
            _em = exact_match_score(pred, gold)

        em += _em
        
    em = em * 100 / (len(preds) + 1e-5)
    return em

def qa_to_prompt(query, context):
    prompt = '{}\nQ: {}\nA: '.format(context, query)
    return prompt

def qa_to_prompt_baseline(query, context, schema):
    def get_prompt(query, context, schema, answer=''):
        if schema == 'base':
            prompt = '{}\nQ:{}\nA:{}'.format(context, query, answer)
        elif schema == 'opin':
            context = context.replace('"', "")
            prompt = 'Bob said "{}"\nQ: {} in Bob\'s opinion?\nA:{}'.format(context, query[:-1], answer)
        elif schema == 'instr+opin':
            context = context.replace('"', "")
            prompt = 'Bob said "{}"\nQ: {} in Bob\'s opinion?\nA:{}'.format(context, query[:-1], answer)
        elif schema == 'attr':
            prompt = '{}\nQ:{} based on the given tex?\nA:{}'.format(context, query[:-1], answer)
        elif schema == 'instr':
            prompt = '{}\nQ:{}\nA:{}'.format(context, query, answer)
        elif schema == 'our_prompt':
            template = open("./prompt_ours.txt").read()
            prompt = template.format(context, query)
        elif schema == 'our_prompt_test':
            template = open("./prompt_ours_test.txt").read()
            prompt = template.format(context, query)

        return prompt
    prompt = ''
    if schema in ('instr', 'instr+opin'):
        prompt = 'Instruction: read the given information and answer the corresponding question.\n\n'
    prompt = prompt + get_prompt(query, context, schema=schema)
    return prompt

    
def eval(pred_answers, gold_answers, step):
    em = get_score(pred_answers, gold_answers)
    # mr = po / (ps + po + 1e-10) * 100
    logging.info('Step: {}: em {}.'.format(step, em))
    
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
    parser.add_argument("--model_name", default="gpt-4o-mini", type=str)
    # args = parser.parse_args()
    parser.add_argument("--data_path", default="./followRAG/ConFiQA-QA.json", type=str)
    parser.add_argument("--schema", default="our_prompt", type=str, help="Choose from the following prompting templates: base, attr, instr, opin, instr+opin.")
    parser.add_argument("--output_path", default='./result/Qwen2-7B-Instruct.json', type=str)
    # args = parser.parse_args()
    # schema = args.schema
    parser.add_argument("--log_path", default='./log_followRAG/Qwen2-7B-Instruct.log', type=str)
    args = parser.parse_args()
    model_name = args.model_name
    # args.log_path = './log_ConFiQA/Qwen2-7B-Instruct.log' % schema,

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_path),  # 写入日志文件
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logging.info("Evaluate FollowRAG for the Model: %s" % model_name)
    with open(args.data_path, 'r') as fh:
        data = json.load(fh)
    logging.info('Loaded {} instances.'.format(len(data)))
    
    create_log_path(args.log_path)
    
    all_gold_answers, pred_answers = [], []


    stop = []
    step = 0
    
    for d in tqdm(data):
        step += 1
        query = d['question']
        context = d['prompt'].split('Given the following information:')[1].split('Answer the following question')[0].strip()
        # print(len(context))
        gold_answers = d['answer_gold']
        
        prompt = qa_to_prompt_baseline(query, context, schema=args.schema)
        # pred = query_gpt(prompt, args.model_name).split('.')[0].split('\n')[0]
        if args.model_name == "claude-3-7-sonnet-20250219":
            pred = query_claude(prompt, llm_model_name=args.model_name).split('.')[0].split('\n')[0]
        elif args.model_name == "claude-3-7-sonnet-20250219-think":
            pred = query_claude_think(prompt, llm_model_name="claude-3-7-sonnet-20250219").split('.')[0].split('\n')[0]
        elif args.model_name == "deepseek-chat":
            pred = query_deepseek(prompt, llm_model_name=args.model_name).split('.')[0].split('\n')[0]
            # print(pred)
        elif args.model_name == "deepseek-reasoner":
            pred = query_deepseek(prompt, llm_model_name=args.model_name).split('.')[0].split('\n')[0].replace("Answer:",'').strip()
            # print(pred)
        else:
            pred = query_gpt(prompt, llm_model_name=args.model_name).split('.')[0].split('\n')[0]
        pred_answers.append(pred)

        if '/' in gold_answers:
            gold_answers = gold_answers.split('/')
        else:
            gold_answers = [gold_answers]

        all_gold_answers.append(gold_answers)
        d['pred'] = pred
        
        if step % 50 == 0:
            eval(pred_answers, all_gold_answers, step)
            # time.sleep(60)
            with open(args.output_path, 'w') as fh:
                json.dump(data, fh)
    with open(args.output_path, 'w') as fh:
        json.dump(data, fh)
    

if __name__ == '__main__':
    main()
