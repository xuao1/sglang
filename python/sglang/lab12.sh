#!/bin/bash

# 循环 10~100，步长 10
for i in {10..100..10}; do
    echo "========================================"
    echo "Running with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$i"
    echo "Log file: lab12_${i}.log"
    echo "========================================"

    # 运行命令并记录日志
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$i python -m sglang.bench_one_batch \
        --model-path /model/Meta-Llama-3-8B-Instruct/ \
        --batch 1 4 16 \
        --input-len 128 \
        --output-len 2050 \
        --mem-fraction-static 0.8 \
        > "lab12_${i}.log" 2>&1  # 将 stderr 也重定向到日志

    echo "Completed run for $i"
    echo ""
done
