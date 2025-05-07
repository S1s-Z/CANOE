


MODELS="gpt-4o:evaluation_api.py"


BASE_OUTPUT="./result"
BASE_LOG="./log"

# 任务配置（数据集名称:数据路径）
TASKS="
hq:./followRAG/hq.json
nq:./followRAG/nq.json
tq:./followRAG/tq.json
webq:./followRAG/webq.json
"


# 遍历所有模型
for model_config in $MODELS; do
    # 提取模型名称（从路径中）
    
    # 当前模型的GPU计数器
    model_name=$(echo "$model_config" | cut -d ':' -f 1)
    eval_script=$(echo "$model_config" | cut -d ':' -f 2)
    
    echo "================================================================"
    echo "[$(date)] 开始处理模型: $model_name"
    echo "================================================================"
    

    # 遍历所有任务
    for task_config in $TASKS; do
        # 解析任务配置
        task_name=$(echo "$task_config" | cut -d ':' -f 1)
        data_path=$(echo "$task_config" | cut -d ':' -f 2)
        
        # 生成输出文件名
        output_file="${task_name}-${model_name}.json"
        log_file="${task_name}-${model_name}.log"
        
        # 打印任务信息
        echo "[$(date)] 启动任务: ${task_name}"
        
        # 执行评估命令
        python "$eval_script" \
            --model_name "$model_name" \
            --data_path "$data_path" \
            --output_path "${BASE_OUTPUT}/${output_file}" \
            --log_path "${BASE_LOG}/${log_file}" &
        
    done
    # 等待当前模型的所有任务完成
    wait
    echo "[$(date)] 已完成模型: $model_name 的所有任务"
done

echo "全部任务完成于: $(date)"