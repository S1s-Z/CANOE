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
from summac.model_summac import SummaCConv
import nltk
# nltk.download('punkt')
# nltk.download('punkt', download_dir="/root/nltk_data/")
nltk.data.path.append("/root/nltk_data/")
# model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")

from minicheck.minicheck import MiniCheck
scorer = MiniCheck(model_name='flan-t5-large', cache_dir='./ckpts', enable_prefix_caching=False)





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
        "max_tokens": 1000 
    }  

    # 发送 POST 请求  
    retries = 0
    max_retries = 5
    while retries < max_retries:
        response = requests.post(  
            'correct value here',  
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
            '',  
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
        "max_tokens": 1000  
    }  
    
    if llm_model_name =="o1-2024-12-17":
        data = {  
            "model": llm_model_name,  
            "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": full_prompt}],  
        }  

    # 发送 POST 请求  
    retries = 0
    max_retries = 5
    while retries < max_retries:
        response = requests.post(  
            '',  
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
        "max_tokens": 1000  
    }  

    # 发送 POST 请求  
    retries = 0
    max_retries = 5
    while retries < max_retries:
        response = requests.post(  
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
    output_str = decoded.strip()
    return output_str

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
    parser.add_argument("--model_name", default="./Models/Qwen2-7B-Instruct", type=str)
    # args = parser.parse_args()
    parser.add_argument("--data_path", default="./dataset/clapnq.json", type=str)
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
    
    logging.info("Evaluate CNQ for the Model: %s" % model_name)
    with open(args.data_path, 'r') as fh:
        data = json.load(fh)
    logging.info('Loaded {} instances.'.format(len(data)))
    
    create_log_path(args.log_path)
    
    all_gold_answers, pred_answers = [], []
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage = True, torch_dtype=torch.float16, trust_remote_code=True)
    # model.cuda()
    # stop = [".", " .", "..."]
    # stop = ["."]
    stop = []
    # stopping_criteria = set_stop_words(tokenizer, stop)
    step = 0
    
    final_score = 0
    for d in tqdm(data):
        step += 1
        query = d['input']
        context = " ".join(item["text"] for item in d['passages'])  
        
        prompt = qa_to_prompt_baseline(query, context, schema=args.schema)
        # pred = call_llama(model, tokenizer, prompt, stopping_criteria, stop)

        if args.model_name == "claude-3-7-sonnet-20250219":
            pred = query_claude(prompt, llm_model_name=args.model_name)
        elif args.model_name == "deepseek-chat":
            pred = query_deepseek(prompt, llm_model_name=args.model_name)
            # print(pred)
        elif args.model_name == "claude-3-7-sonnet-20250219-think":
            pred = query_claude_think(prompt, llm_model_name="claude-3-7-sonnet-20250219")
            # print(pred)
        elif args.model_name == "deepseek-reasoner":
            pred = query_deepseek(prompt, llm_model_name=args.model_name)
            # print(pred)
        else:
            pred = query_gpt(prompt, llm_model_name=args.model_name)
        d['pred'] = pred

        pred_label, raw_prob, _, _ = scorer.score(docs=[context], claims=[pred])
        final_score += float(pred_label[0])

        if step % 50 == 0:
            logging.info('Score: {}.'.format(final_score/step))
            with open(args.output_path, 'w') as fh:
                json.dump(data, fh)

        
    with open(args.output_path, 'w') as fh:
        json.dump(data, fh)

    logging.info('Final score: {}.'.format(final_score/len(data)))
    

if __name__ == '__main__':
    main()
