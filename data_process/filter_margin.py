import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import filter_margin_for_refine

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_dataset_raw", type=str, default="./data/pairwise_traindata.json", help="Path to the raw dataset(teacher).")
    parser.add_argument("--assistant_dataset_raw", type=str, default="./data/pairwise_traindata_gpt3.5_raw.json", help="Path to the raw dataset(assistant).")
    parser.add_argument("--teacher_dataset_margin", type=str, default="./data/pairwise_traindata_gpt4_wo_tie_margin.json", help="Path to the dataset with margin(teacher).")
    parser.add_argument("--assistant_dataset_margin", type=str, default="./data/pairwise_traindata_gpt3.5_wo_tie_margin.json", help="Path to the dataset with margin(assistant).")
    parser.add_argument("--threshold", type=float, default=5, help="Threshold for implicit reward margin filtering.")
    parser.add_argument("--output_path", type=str, default="./data/filtered_margin_same_label.json", help="Path to the output dataset.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)
    filter_margin_for_refine(args.teacher_dataset_raw, args.assistant_dataset_raw, args.teacher_dataset_margin, args.assistant_dataset_margin, args.threshold, args.output_path)