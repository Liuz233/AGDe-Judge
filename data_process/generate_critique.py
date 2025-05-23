import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import generate_critique_by_gpt, generate_critique_by_llm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="model path or name")
    parser.add_argument("--dataset_path", type=str, default="./data/filtered_margin_same_label.json", help="Path to the input dataset")
    parser.add_argument("--output_path", type=str, default="./data/critique.json", help="Path to the output dataset")
    parser.add_argument("--open_source", default=False, action="store_true", help="Use open source model or not")
    parser.add_argument("--model_name", type=str, default="qwen-7b-chat", help="Model template name for open source model")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)
    if args.open_source:
        generate_critique_by_llm(args.model, args.dataset_path, args.output_path, args.model_name)
    else:
        generate_critique_by_gpt(args.model, args.dataset_path, args.output_path)
   