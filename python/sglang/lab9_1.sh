#!/bin/bash

# 定义日志文件路径
LOG_FILE="benchmark_output.log"

# 清空或创建日志文件
> "$LOG_FILE"

# 循环不同的CUDA_MPS线程百分比设置
for percentage in $(seq 100 -10 10); do
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Running with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$percentage" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    # 设置环境变量并运行命令
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$percentage
    python -m sglang.bench_one_batch \
        --model-path /model/Meta-Llama-3-8B-Instruct/ \
        --batch 8 \
        --input-len 1024 \
        --output-len 10 \
        --mem-fraction-static 0.8 >> "$LOG_FILE" 2>&1

    echo "" >> "$LOG_FILE"
done

echo "所有测试完成！结果已保存到 $LOG_FILE"