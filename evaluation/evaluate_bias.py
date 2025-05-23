import os
import json
import re
import sys
from pathlib import Path
import json
sys.path.append(str(Path(__file__).resolve().parent.parent))
# from utils import generate_critique_by_gpt, generate_critique_by_llm
from judge_utils.vllm import VLLM
from judge_utils.judge import AGDEval
from judge_utils.prompts import PAIREWISE_AUTOJ
from datasets import load_dataset
import argparse
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the model")
    parser.add_argument("--output_path", type=str, default="./data/res.json", help="Path for the output result")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    set_seeds(42)
    model = VLLM(model = args.model, lora_config=None, max_model_len = 4096, gpu_memory_utilization = 0.8)
    judge = AGDEval(model=model, relative_grade_template=PAIREWISE_AUTOJ)

    dataset = json.load(open("./data/gpt4_vs_other.json"))
    print(len(dataset))
    target_model = "gpt-4"

    instructions =[]
    responses_A = []
    responses_B = []

    real_win = 0
    real_tie_or_lose = 0
    win_human_dict = {}
    win_model_dict = {}
    for i in range(len(dataset)):
        instructions.append(dataset[i]["conversation_a"][0]["content"])
        responses_A.append(dataset[i]["conversation_a"][1]["content"])
        responses_B.append(dataset[i]["conversation_b"][1]["content"])

        if dataset[i]["label"] == "tie":
            real_tie_or_lose += 1
        else:
            winner = dataset[i]["label"]
            if dataset[i][winner] == target_model:
                real_win += 1
            else:
                real_tie_or_lose += 1
            win_human_dict[dataset[i][winner]] = win_human_dict.get(dataset[i][winner], 0) + 1

    params = {
    "max_tokens": 1024,
    "repetition_penalty": 1.03,
    "best_of": 1,
    "temperature": 1.0,
    "top_p": 0.9,
    "seed": 41,
    }
    feedbacks, scores = judge.relative_grade(
        instructions=instructions,
        responses_A=responses_A,
        responses_B=responses_B,
        params=params,
        mode = "Auto-J",
        model_name="mistral",
    )
    data_list = []

    total_num = len(dataset)
    acc_num = 0
    win_num = 0
    loss_or_tie_num = 0
    for idx,(feedback, label) in enumerate(zip(feedbacks, scores)):

        if label == "A":
            label = 'model_a'
        elif label == "B":
            label = 'model_b'
        else:
            label = 'tie'

        data_list.append({
            **dataset[idx],
            "result_feedback": feedback,
            "result_label": label,
        })
        if label == dataset[idx]["label"]:
            acc_num += 1


        if label == 'tie':
            loss_or_tie_num += 1
        else:
            try:
                if dataset[idx][label] == target_model:
                    win_num += 1
                else:
                    loss_or_tie_num += 1
                win_model_dict[dataset[idx][label]] = win_model_dict.get(dataset[idx][label], 0) + 1
            except:
                print("Error: ", label)


    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)




    print("Real win: ", real_win)
    print("Real tie or lose: ", real_tie_or_lose)
    print("Pred win: ", win_num)
    print("Pred tie or lose: ", loss_or_tie_num)

    print("Accuracy num: ", acc_num)
    print("Accuracy: ", acc_num / total_num)

    print("-------------------")
    print("human win: ", win_human_dict)
    print("model win: ", win_model_dict)