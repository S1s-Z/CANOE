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


def query_claude(full_prompt, llm_model_name='claude-3-7-sonnet-20250219'):

    # 替换成你的 OpenAI API 密钥  
    api_key = 'sk-proj-NVa8Rr8cBTaA2lETfnoRmSQiAHuQu9XYf4yTkGL3OcKNVtWd'  

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
    api_key = 'sk-proj-NVa8Rr8cBTaA2lETfnoRmSQiAHuQu9XYf4yTkGL3OcKNVtWd'  

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
    api_key = 'sk-0qwyW6kPmaFjFBII2529Fa7e5cC54236A37987D14bE0A0F5'  

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
    # api_key = 'sk-FastAPI1mNDid0j1xzrXT8YBI9l1qmO0dJrmT3nd9re4BRlJ'  
    api_key = "sk-7045f85e6dd94330b2a4cf5e64c63bdd"

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
            stop_word_ids = tokenizer.encode('\n' + stop_word)[3:]
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

negation_words = [
    "no", "not", "never", "none", "cannot", "nobody", "nothing", "nowhere", 
    "neither", "nor", "without", "hardly"
]

def exact_match_score(prediction, ground_truth, is_cf):
    contains_negation = any(word in prediction.split() for word in negation_words)
    return (not contains_negation if is_cf else True) and (normalize_answer(prediction) == normalize_answer(ground_truth))    

def recall_score(prediction, ground_truth, is_cf):
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    
    contains_negation = any(word in prediction.split() for word in negation_words)
    
    return (ground_truth in prediction) and (not contains_negation if is_cf else True)

def get_score(preds, golds, origs):
    em, gold_recall, orig_recall = 0, 0, 0
    for pred, gold, orig in zip(preds, golds, origs):
        if isinstance(gold, list):
            _em, _recall = 0, 0
            for g in gold:
                _em = max(exact_match_score(pred, g, True), _em)
                _recall = max(recall_score(pred, g, True), _recall)
        else:
            _em = exact_match_score(pred, gold, True)
            _recall = recall_score(pred, gold, True)
        if isinstance(orig, list):
            _recall_orig = 0
            for o in orig:
                _recall_orig = max(recall_score(pred, o, False), _recall_orig)
        else:
            _recall_orig = recall_score(pred, orig, False)
        em += _em
        gold_recall += _recall and not _recall_orig
        orig_recall +=  _recall_orig
        
    em = em * 100 / (len(preds) + 1e-5)
    gold_recall = gold_recall * 100 / (len(preds) + 1e-5)
    orig_recall = orig_recall * 100 / (len(preds) + 1e-5)
    return em, gold_recall, orig_recall

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
            # bad
            # prompt = 'Context:{}\n Question:{}\n Please answer our Question based on the content of the given Context, please directly give the answer:{}'.format(context, query, answer)
            # prompt = 'Documents:{}\n Question:{}\n Answer our Question based on the content of the given Context, directly give the answer, e.g., Luis Abinader Corona:{}'.format(context, query, answer)
            
            # good
            # prompt = "Refer to the passages below and answer the following question with just a few words.\n {} \n Refer to the context above and answer the following question with just a few words. \n Question: {} \n The answer is".format(context, query, answer)

            # template = open("./prompt_ours_old.txt").read()
            # prompt = template.format(query, context, answer)

            template = open("./prompt_ours.txt").read()
            prompt = template.format(context, query)
            # print(prompt)
        return prompt
    
    prompt = ''
    if schema in ('instr', 'instr+opin'):
        prompt = 'Instruction: read the given information and answer the corresponding question.\n\n'
    prompt = prompt + get_prompt(query, context, schema=schema)
    return prompt

    
def eval(pred_answers, orig_answers, gold_answers, step):
    em, ps, po = get_score(pred_answers, gold_answers, orig_answers)
    mr = po / (ps + po + 1e-10) * 100
    logging.info('Step: {}: ps {}, po {}, mr {}, em {}.'.format(step, ps, po, mr, em))
    
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
    parser.add_argument("--data_path", default="./dataset/conflict_dev_filtered.json", type=str)
    parser.add_argument("--schema", default="our_prompt", type=str, help="Choose from the following prompting templates: base, attr, instr, opin, instr+opin.")
    parser.add_argument("--output_path", default='./result/Qwen2-7B-Instruct.json', type=str)
    # args = parser.parse_args()
    # schema = args.schema
    parser.add_argument("--log_path", default='./log_followRAG/Qwen2-7B-Instruct.log', type=str)
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
    
    logging.info("Evaluate CNQ for the Model: %s" % model_name)
    with open(args.data_path, 'r') as fh:
        data = json.load(fh)
    logging.info('Loaded {} instances.'.format(len(data)))
    
    create_log_path(args.log_path)
    
    # all_gold_answers, pred_answers = [], []
    stop = []
    step = 0
    gold_answers, pred_answers, orig_answers = [], [], []
    for d in tqdm(data):
        step += 1
        query = d['question']
        # context = d['cf_context']
        context = d['orig_context']
        cf_answer = d['cf_answer']
        orig_answer = d['orig_answer']
        
        prompt = qa_to_prompt_baseline(query, context, schema=args.schema)
        if args.model_name == "claude-3-7-sonnet-20250219":
            pred = query_claude(prompt, llm_model_name=args.model_name).split('.')[0].split('\n')[0]
        elif args.model_name == "claude-3-7-sonnet-20250219-think":
            pred = query_claude_think(prompt, llm_model_name="claude-3-7-sonnet-20250219").split('.')[0].split('\n')[0]
        elif args.model_name == "deepseek-chat":
            pred = query_deepseek(prompt, llm_model_name=args.model_name).split('.')[0].split('\n')[0]
        elif args.model_name == "deepseek-reasoner":
            pred = query_deepseek(prompt, llm_model_name=args.model_name).split('.')[0].split('\n')[0].replace("Answer:",'').strip()
        else:
            pred = query_gpt(prompt, llm_model_name=args.model_name).split('.')[0].split('\n')[0]
        pred_answers.append(pred)


        if len(d['cf_alias']) != 0:
            cf_answer = [cf_answer] + d['cf_alias']
        if len(d['orig_alias']) != 0:
            orig_answer = [orig_answer] + d['orig_alias']
        gold_answers.append(cf_answer)
        orig_answers.append(orig_answer)
        d['pred'] = pred
        d['normalized_answer'] = normalize_answer(pred)

        if step % 100 == 0:
            # eval(pred_answers, orig_answers, gold_answers, step)
            eval(pred_answers, gold_answers, orig_answers, step)
            with open(args.output_path, 'w') as fh:
                json.dump(data, fh)
    with open(args.output_path, 'w') as fh:
        json.dump(data, fh)
    

if __name__ == '__main__':
    main()
