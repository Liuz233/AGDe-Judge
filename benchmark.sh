export CUDA_VISIBLE_DEVICES=0

echo "Using GPU device: $CUDA_VISIBLE_DEVICES"
datasets=("OffsetBias"  "JudgeLM"  "UltraFeedback"  "Arena-Human"  "Reward-Bench"  "MT-Bench"  "Preference-Bench")
for dataset in "${datasets[@]}"; do
    echo "Running benchmark for $dataset"
    python evaluation/evaluate_acc.py --model /path/to/your/proxy_judge_model --dataset $dataset
done