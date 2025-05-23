import argparse
import sys
from pathlib import Path
import json
sys.path.append(str(Path(__file__).resolve().parent.parent))
# from utils import generate_critique_by_gpt, generate_critique_by_llm
from judge_utils.vllm import VLLM
from judge_utils.judge import AGDEval
from judge_utils.prompts import PAIREWISE_AUTOJ
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the model")
    parser.add_argument("--dataset", type=str, default="OffsetBias", help="[OffsetBias, JudgeLM, UltraFeedback, Arena-Human, Reward-Bench, MT-Bench, Preference-Bench]")
    parser.add_argument("--output_path", type=str, default="./data/res.json", help="Path for the output result")
    args = parser.parse_args()
    return args

def set_seeds(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  

def process_dataset_offsetbias(dataset):
    new_dataset = []
    for data in dataset:
        if 'tie' in data["label"] or 'TIE' in data["label"]:
            continue
        new_dataset.append(data)
    
    dataset = new_dataset
    instructions =[]
    responses_A = []
    responses_B = []
    # rubric = []
    # reference_answers = []

    for i in range(len(dataset)):
        instructions.append(dataset[i]["instruction"])
        responses_A.append(dataset[i]["response_A"])
        responses_B.append(dataset[i]["response_B"])

    return instructions, responses_A, responses_B, dataset



def process_dataset_arena(dataset, model_name):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    new_dataset = []
    for data in dataset:
        prompt = PAIREWISE_AUTOJ.format(
            instruction=data["conversation_a"][0]["content"],
            response_A=data["conversation_a"][1]["content"],
            response_B=data["conversation_b"][1]["content"]
        )
        tokens = tokenizer(prompt, return_tensors="pt").input_ids[0]
        if len(tokens) > 4096:
            continue
        if 'tie' in data["label"] or 'TIE' in data["label"]:
            continue
        new_dataset.append(data)
    
    dataset = new_dataset
    instructions =[]
    responses_A = []
    responses_B = []

    for i in range(len(dataset)):
        instructions.append(dataset[i]["conversation_a"][0]["content"])
        responses_A.append(dataset[i]["conversation_a"][1]["content"])
        responses_B.append(dataset[i]["conversation_b"][1]["content"])

    return instructions, responses_A, responses_B, dataset

def process_dataset_reward_bench(dataset, model_name):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    new_dataset = []
    for data in dataset:
        prompt = PAIREWISE_AUTOJ.format(
            instruction=data["instruction"],
            response_A=data["response_A"],
            response_B=data["response_B"],
        )
        tokens = tokenizer(prompt, return_tensors="pt").input_ids[0]
        if len(tokens) > 4096:
            continue
        # print("Token length: ", len(tokens))
        if 'tie' in data["label"] or 'TIE' in data["label"]:
            continue
        new_dataset.append(data)
    
    dataset = new_dataset
    instructions =[]
    responses_A = []
    responses_B = []


    for i in range(len(dataset)):
        instructions.append(dataset[i]["instruction"])
        responses_A.append(dataset[i]["response_A"])
        responses_B.append(dataset[i]["response_B"])

    return instructions, responses_A, responses_B, dataset

if __name__ == "__main__":

    args = parse_args()
    print(args)
    # set_seeds(42)
    if args.dataset == "OffsetBias":
        dataset = json.load(open("./data/OffsetBias.json"))
        instructions, responses_A, responses_B, dataset = process_dataset_offsetbias(dataset)
    elif args.dataset == "JudgeLM":
        dataset = json.load(open("./data/JudgeLM.json"))
        instructions, responses_A, responses_B, dataset = process_dataset_offsetbias(dataset)
    elif args.dataset == "UltraFeedback":
        dataset = json.load(open("./data/UltraFeedback.json"))
        instructions, responses_A, responses_B, dataset = process_dataset_offsetbias(dataset)
    elif args.dataset == "Arena-Human":
        dataset = json.load(open("./data/Arena-Human.json"))
        instructions, responses_A, responses_B, dataset = process_dataset_arena(dataset, args.model)
    elif args.dataset == "Reward-Bench":
        dataset = json.load(open("./data/Reward-Bench.json"))
        instructions, responses_A, responses_B, dataset = process_dataset_reward_bench(dataset, args.model)
    elif args.dataset == "MT-Bench":
        dataset = json.load(open("./data/MT-Bench.json"))
        instructions, responses_A, responses_B, dataset = process_dataset_arena(dataset, args.model)
    elif args.dataset == "Preference-Bench":
        dataset = json.load(open("./data/Preference-Bench.json"))
        instructions, responses_A, responses_B, dataset = process_dataset_offsetbias(dataset)
    else:
        raise ValueError("dataset not supported")

    model = VLLM(model = args.model, lora_config=None, max_model_len = 4096, gpu_memory_utilization = 0.8)
    judge = AGDEval(model=model, relative_grade_template= PAIREWISE_AUTOJ)

    params = {
    "max_tokens": 1024,
    "repetition_penalty": 1.03,
    "best_of": 1,
    "temperature": 1.0,
    "top_p": 0.9,
    "seed": 42,
    }
    feedbacks, scores = judge.relative_grade(
        instructions=instructions,
        responses_A=responses_A,
        responses_B=responses_B,
        params=params,
        mode = "Auto-J",
        model_name = "mistral"
    )
    data_list = []
    total_num = len(dataset)
    acc_num = 0
    for idx,(feedback, label) in enumerate(zip(feedbacks, scores)):
        if label == dataset[idx]["label"]:
            acc_num += 1
        data_list.append({
            ** dataset[idx],
            "result_feedback": feedback,
            "result_label": label,
        })


    print("Accuracy num: ", acc_num)
    print("Accuracy: ", acc_num / total_num)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)
    