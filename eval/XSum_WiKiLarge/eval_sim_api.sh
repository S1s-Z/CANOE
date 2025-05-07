
MODEL_NAMES="o1-2024-12-17"
# 定义对应的评估脚本列表
EVAL_SCRIPTS="evaluation_api_minicheck.py"


# 计数器初始化
count=1

# 循环执行所有任务
for model_name in $MODEL_NAMES; do
    # 提取对应的 GPU ID 和评估脚本
    eval_script=$(echo "$EVAL_SCRIPTS" | cut -d ' ' -f $count)
    
    # 打印执行信息（便于调试）
    echo "[$(date)] Starting $model_name"
    
    # 执行推理任务
    python "./$eval_script" \
        --model_name "${model_name}" \
        --schema "sim" \
        --data_path "./data/simplification.json" \
        --output_path "./sim_result/${model_name}.json" \
        --log_path "./sim_log/${model_name}.log" &
    
    count=$((count + 1))
done

# 等待所有后台任务完成
wait
echo "All tasks completed at $(date)"




