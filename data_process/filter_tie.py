import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import filter_raw

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./data/pairwise_traindata_gpt3.5_raw.json", help="Path to the raw dataset")
    parser.add_argument("--output_path", type=str, default="./data/pairwise_traindata_gpt3.5_wo_tie.json", help="Path to the new dataset")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)
    filter_raw(args.dataset_path, args.output_path)