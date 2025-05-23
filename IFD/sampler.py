import json
# import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo_ppl_path", type=str, default='./data/dpo_llama_gpt4_ppl.jsonl')
    parser.add_argument("--sft_ppl_path", type=str, default='./data/sft_llama_gpt4_ppl.jsonl')
    parser.add_argument("--input_path", type=str, default='./data/pairwise_traindata_gpt4_wo_tie.json')
    parser.add_argument("--output_path", type=str, default='./data/pairwise_traindata_gpt4_wo_tie_margin.json')
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = parse_args()
    print(args)
    # def Sampling(name, ds, strategy, part, Num, input_path, output_path):
    rewards = []

    f_dpo = open(args.dpo_ppl_path, 'r')
    f_sft = open(args.sft_ppl_path, 'r')

    for dpo_line, sft_line in zip(f_dpo, f_sft):
        dpo_data = json.loads(dpo_line)
        sft_data = json.loads(sft_line)
        
        reward = dpo_data['reward'][0] - sft_data['reward'][0]
        rewards.append(reward)

    f_dpo.close()
    f_sft.close()
    rewards = np.array(rewards)
    if np.isnan(rewards).any():
        rewards = np.nan_to_num(rewards, nan=0)
    else:
        print("no nan in rewards")

    margin = rewards
    mid_edge = 1

    raw_data_json = json.load(open(args.input_path, "r", encoding="utf-8"))
    assert len(raw_data_json) == len(margin), f"Length mismatch: {len(raw_data_json)} vs {len(margin)}, check the data!"
    for i in range(len(raw_data_json)):
        raw_data_json[i]['margin'] = margin[i]
    
    print(f"length of raw_data_json: {len(raw_data_json)}")
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(raw_data_json, f, indent=4, ensure_ascii=False)