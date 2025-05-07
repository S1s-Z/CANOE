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


def make_conversation(problem):
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
    # return f"{problem} First output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
    return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem}
        ]

class LLamaQaStoppingCriteria(StoppingCriteria):
    def __init__(self, list_token_ids_sequence: list = []):
        self.token_ids_sequences = []
        self.lengths = []
        for token_ids_sequence in list_token_ids_sequence:
            self.token_ids_sequences.append(torch.tensor(token_ids_sequence, dtype=torch.long))
            self.lengths.append(len(token_ids_sequence))
        
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
            
def call_llama(model, tokenizer, chat, stopping_criteria, stop):

    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    # input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt", add).input_ids.to("cuda")
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    # prompt = chat
    # print('Question:', prompt)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False).to("cuda")
    # sequences = model.generate(input_ids.cuda(), stopping_criteria = stopping_criteria, max_new_tokens = 1024)[0, input_ids.shape[-1]:]
    sequences = model.generate(inputs.input_ids.to("cuda"), max_new_tokens = 1024, attention_mask=inputs.attention_mask.to("cuda"))[0, inputs.input_ids.shape[-1]:]
    # sequences = model.generate(**inputs)
    # decoded = tokenizer.batch_decode(sequences, skip_special_tokens=False)
    decoded = tokenizer.decode(sequences, skip_special_tokens=True)
    # import pdb; pdb.set_trace()
    # print('Answer:', decoded)
    for stop_word in stop:
        length_to_remove = len(stop_word)
        if decoded[-length_to_remove:] == stop_word:
            decoded = decoded[:-length_to_remove]
    output_str = decoded.strip()


    think_pattern = r'<think>(.*?)</think>'
    match_think = re.search(think_pattern, output_str)
    if match_think:
        think_str = match_think.group(1).strip()
    else:
        think_str = output_str.strip()

    long_answer_pattern = r'<long_answer>(.*?)</long_answer>'
    match_long_answer = re.search(long_answer_pattern, output_str)
    if match_long_answer:
        long_answer_str = match_long_answer.group(1).strip()
    else:
        long_answer_str = output_str.strip()

    # r1 pattern
    r1_pattern = r'<answer>(.*?)</answer>'  
    match = re.search(r1_pattern, output_str)  
    if match:  
        output_str = match.group(1).strip()  
    else:  
        output_str = output_str.strip()
    # import pdb; pdb.set_trace()
    return output_str, long_answer_str, think_str

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
    # print(prediction)
    # print(ground_truth)
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def recall_score(prediction, ground_truth, is_cf):
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    
    contains_negation = any(word in prediction.split() for word in negation_words)
    
    return (ground_truth in prediction) and (not contains_negation if is_cf else True)

def get_score(preds, golds):
    em = 0
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
            # prompt = '{}\nQ:{}\nA:{}'.format(context, query, answer)
            # user_question = '<context> {} </context>\n\n{}'.format(context, query)
            # prompt = make_conversation(user_question)
            user_question = '<context>{}</context>\n\nRefer to the passages above, first think about the reasoning process internally and then answer the following question with just a few words. \n Question:{}'.format(context, query)
            prompt = make_conversation(user_question)
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
        return prompt
    # prompt = ''
    if schema in ('instr', 'instr+opin'):
        prompt = 'Instruction: read the given information and answer the corresponding question.\n\n'
    # prompt = prompt + get_prompt(query, context, schema=schema)
    prompt = get_prompt(query, context, schema=schema)
    # print(type(prompt))
    return prompt

    
def eval(pred_answers, gold_answers, step):
    em = get_score(pred_answers, gold_answers)

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
    parser.add_argument("--data_path", default="./followRAG/ConFiQA-QA.json", type=str)
    parser.add_argument("--schema", default="base", type=str, help="Choose from the following prompting templates: base, attr, instr, opin, instr+opin.")
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
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage = True, torch_dtype=torch.float16, trust_remote_code=True)
    model.cuda()
    stop = []
    stopping_criteria = set_stop_words(tokenizer, stop)
    step = 0
    
    for d in tqdm(data):
        step += 1
        query = d['question']
        context = d['prompt'].split('Given the following information:')[1].split('Answer the following question')[0].strip()
        # print(len(context))
        gold_answers = d['answer_gold']
        
        prompt = qa_to_prompt_baseline(query, context, schema=args.schema)
        # pred, think_str = call_llama(model, tokenizer, prompt, stopping_criteria, stop)
        pred, long_answer_str, think_str = call_llama(model, tokenizer, prompt, stopping_criteria, stop)
        pred_answers.append(pred)

        if '/' in gold_answers:
            gold_answers = gold_answers.split('/')
        else:
            gold_answers = [gold_answers]

        all_gold_answers.append(gold_answers)
        

        d['pred'] = pred
        d['think'] = think_str
        d['long_answer'] = long_answer_str

        if step % 100 == 0:
            eval(pred_answers, all_gold_answers, step)

        if step % len(data) == 0:
            eval(pred_answers, all_gold_answers, step)

        
    with open(args.output_path, 'w') as fh:
        json.dump(data, fh)
    

if __name__ == '__main__':
    main()
