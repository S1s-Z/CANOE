



# MODEL_NAMES="claude-3-7-sonnet-20250219"
# MODEL_NAMES="o1-2024-12-17"
MODEL_NAMES="gpt-4o"
# MODEL_NAMES="deepseek-reasoner"


# 定义对应的评估脚本列表（前4个用 evaluation_ours.py）
# EVAL_SCRIPTS="evaluation_api.py"

# 计数器初始化
count=1

# 循环执行所有任务
for model_name in $MODEL_NAMES; do
    # 提取对应的 GPU ID 和评估脚本
    # eval_script=$(echo "$EVAL_SCRIPTS" | cut -d ' ' -f $count)
    
    # 打印执行信息（便于调试）
    echo "[$(date)] Starting $model_name"
    
    # 执行推理任务
    
    python "./evaluation_api.py" \
        --model_name "${model_name}" \
        --output_path "./result/${model_name}.json" \
        --log_path "./log/${model_name}.log" &
    
    count=$((count + 1))
done

# 等待所有后台任务完成
wait
echo "All tasks completed at $(date)"