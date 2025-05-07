# 定义任务配置（模型名称 + 评估脚本）


MODELS="#model name here::evaluation_ours_factual.py"


TASK_TYPES="
QA:./dataset/ConFiQA-QA-test.json:0
MR:./dataset/ConFiQA-MR-test.json:1 
MC:./dataset/ConFiQA-MC-test.json:2
"

# 基础路径
BASE_OUTPUT="./result_factual"
BASE_LOG="./log_factual"

# 执行所有任务
for model_config in $MODELS; do
    # 解析模型配置
    model_name=$(echo "$model_config" | cut -d ':' -f 1)
    eval_script=$(echo "$model_config" | cut -d ':' -f 2)
    
    for task_config in $TASK_TYPES; do
        # 解析任务配置
        task_suffix=$(echo "$task_config" | cut -d ':' -f 1)
        data_path=$(echo "$task_config" | cut -d ':' -f 2)
        gpu_id=$(echo "$task_config" | cut -d ':' -f 3)
        
        # 生成输出文件名
        output_file="${model_name}-${task_suffix}.json"
        log_file="${model_name}-${task_suffix}.log"
        
        # 打印执行信息
        echo "[$(date)] Starting $model_name $task_suffix on GPU $gpu_id"
        
        # 执行评估任务
        CUDA_VISIBLE_DEVICES=$gpu_id python "$eval_script" \
            --model_name "/mnt/public/share/users/sishuzheng/models/$model_name" \
            --data_path "$data_path" \
            --output_path "$BASE_OUTPUT/$output_file" \
            --log_path "$BASE_LOG/$log_file" &
    done

    wait
done

# 等待所有后台任务完成
wait
echo "All tasks completed at $(date)"