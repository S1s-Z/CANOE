export DEBUG_MODE="true"

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero2_offload.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/Qwen/Qwen2.5-Instruct/qwen_14b_2epoch_10k.yaml