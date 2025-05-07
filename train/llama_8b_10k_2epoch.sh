export DEBUG_MODE="true"

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/LLama/LLama-Instruct/llama3_8b_2epoch_10k.yaml



