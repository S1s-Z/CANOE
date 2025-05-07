

MODEL_NAMES="#model name here"

# 定义对应的 GPU ID 列表
GPU_IDS="0"

# 定义对应的评估脚本列表（前4个用 evaluation_ours.py）
EVAL_SCRIPTS="evaluation_ours.py"

# 计数器初始化
count=1

# 循环执行所有任务
for model_name in $MODEL_NAMES; do
    # 提取对应的 GPU ID 和评估脚本
    gpu_id=$(echo "$GPU_IDS" | cut -d ' ' -f $count)
    eval_script=$(echo "$EVAL_SCRIPTS" | cut -d ' ' -f $count)
    
    # 打印执行信息（便于调试）
    echo "[$(date)] Starting $model_name on GPU $gpu_id using $eval_script"
    
    # 执行推理任务
    CUDA_VISIBLE_DEVICES=$gpu_id python "./$eval_script" \
        --model_name "/mnt/public/share/users/sishuzheng/models/$model_name" \
        --output_path "./result/${model_name}.json" \
        --log_path "./log/${model_name}.log"
    
    count=$((count + 1))
done

# 等待所有后台任务完成
wait
echo "All tasks completed at $(date)"