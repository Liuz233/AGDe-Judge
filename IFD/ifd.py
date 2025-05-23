import os
import json
import torch
import argparse
from tqdm import tqdm
import re
from datasets import load_from_disk, concatenate_datasets, Dataset
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from transformers import AutoTokenizer, AutoModelForCausalLM

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='./data/pairwise_traindata_gpt4_wo_tie.json')
    parser.add_argument("--save_path", type=str, default='./data/sft_llama_gpt4_ppl.jsonl')
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()
    return args

# Used to get the ppl and emb for the whole input
def get_perplexity_and_embedding_whole_text(tokenizer, model, text, max_length):

    try:
        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

        with torch.no_grad(): 
            outputs = model(input_ids, labels=input_ids.contiguous())
        loss = outputs.loss
        perplexity = torch.exp(loss)

        return perplexity.to('cpu').item()

    except:
        return 0, 0

# Used to get the ppl and emb for part of input, used in conditional version, and token-wise loss
def get_perplexity_and_embedding_part_text(tokenizer, model, text, target_span, max_length):

    try:
        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

        start_index = text.rfind(target_span)
        start_token = len(tokenizer.encode(text[:start_index]))
        end_token = input_ids.shape[1]

        labels = input_ids.clone()
        labels[0, :start_token] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)

        loss = outputs.loss
        perplexity = torch.exp(loss)

        return perplexity.to('cpu').item(), loss.to('cpu').item() * -(end_token - start_token)
    
    except:
        return 0, 0

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]

def process(raw_text):
    system_prompt = raw_text.split('[BEGIN DATA]')[0].strip()

    question_match = re.search(r"\[Query\]:(.*?)\[Response 1\]", raw_text, re.DOTALL)
    response1_match = re.search(r"\[Response 1\]:(.*?)\[Response 2\]", raw_text, re.DOTALL)
    response2_match = re.search(r"\[Response 2\]:(.*?)\[END DATA\]", raw_text, re.DOTALL)
    criteria_match  = raw_text.split('[END DATA]')[-1].strip()

    # 去掉换行和前后空格
    question = question_match.group(1).strip() if question_match else ''
    response1 = re.sub(r'^\*{3,}\s*', '', response1_match.group(1).strip()) if response1_match else ''
    response2 = response2_match.group(1).strip() if response2_match else ''
    clean_text = lambda x: re.sub(r'^\*{3,}\s*|\s*\*{3,}$', '', x.strip(), flags=re.MULTILINE)

    question = clean_text(question_match.group(1)) if question_match else ''
    response1 = clean_text(response1_match.group(1)) if response1_match else ''
    response2 = clean_text(response2_match.group(1)) if response2_match else ''
    return system_prompt, question, response1, response2, criteria_match

def transform_to_preference_data(data):
        
    system_prompt, question, response1, response2, criteria_match = process(data["usrmsg"])
    if data["pred_label"] == 0:
        chosen = response1
        rejected = response2
    elif data["pred_label"] == 1:
        chosen = response2
        rejected = response1
    else:
        raise ValueError("Error: pred_label is not 0 or 1")

    return question, chosen, rejected

def main():

    args = parse_args()
    print(args)
    # attn_implementation="flash_attention_2", 
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda") # for llama series
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)


    # chat_template = open('***/model_tmp/llama-2-chat.jinja').read()
    # chat_template = chat_template.replace('    ', '').replace('\n', '')
    # tokenizer.chat_template = chat_template

    model.eval()
    dataset = json.load(open(args.dataset_path, "r", encoding="utf-8"))
    ds = Dataset.from_list(dataset)
    
    print(ds)

    if not os.path.exists(args.save_path):
        with open(args.save_path, "w") as file:
            pass  # Creates an empty file

    with open(args.save_path, "r") as file:
        exsisting_num =  sum(1 for _ in file)
        print(f'existing row:{exsisting_num}')
    ds = ds.select(range(exsisting_num, len(ds)))


    for i in tqdm(range(len(ds))):

        data_i = ds[i]
        prompt, chosen, rejected = transform_to_preference_data(data_i)
        prompt_messages = [{'role':'user','content':prompt}]
        # Now we extract the final turn to define chosen/rejected responses
        chosen_messages = chosen
        rejected_messages = rejected
        instruct_i = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True).replace(tokenizer.bos_token, "")
        whole_text_chosen = instruct_i + chosen_messages
        whole_text_rejected = instruct_i + rejected_messages

        
        ppl_chosen_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, chosen_messages, args.max_length)
        ppl_rejected_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, rejected_messages, args.max_length)
        ppl_chosen_condition, loss_chosen_condition = get_perplexity_and_embedding_part_text(tokenizer, model, whole_text_chosen, chosen_messages, args.max_length)
        ppl_rejected_condition, loss_rejected_condition = get_perplexity_and_embedding_part_text(tokenizer, model, whole_text_rejected, rejected_messages, args.max_length)


        temp_data_i = {}
        temp_data_i['ppl_chosen'] = [ppl_chosen_alone,ppl_chosen_condition]
        temp_data_i['ppl_rejected'] = [ppl_rejected_alone,ppl_rejected_condition]
        temp_data_i['reward'] = [loss_chosen_condition - loss_rejected_condition]

        with open(args.save_path, "a") as file:
            file.write(json.dumps(temp_data_i) + '\n')

    model.cpu()
    
    print('Done: Data Analysis:',args.save_path)

if __name__ == "__main__":
    main()
