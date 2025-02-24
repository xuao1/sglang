#!/bin/bash

# 遍历finetune的配额（0, 10, 20,..., 90）
for finetune_percent in {0..90..10}; do
    echo -e "\n\033[34m===== 开始 finetune ${finetune_percent}% 测试 =====\033[0m"
    
    train_pid=""
    # 跳过0%的finetune启动
    if [ $finetune_percent -ne 0 ]; then
        echo "启动后台训练任务（配额 ${finetune_percent}%）..."
        CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$finetune_percent nohup llamafactory-cli train /workspace/LLaMA-Factory/examples/train_lora/llama3_lora_sft.yaml > /dev/null 2>&1 &
        train_pid=$!
        echo "训练进程已启动 PID: $train_pid"
        
        echo "等待20秒让训练任务稳定运行..."
        sleep 20
    fi

    max_infer_percent=$((100 - finetune_percent))
    echo -e "\n开始推理测试（可用配额 ${max_infer_percent}%）:"

    # 遍历inference配额
    for infer_percent in $(seq 10 10 $max_infer_percent); do
        echo "测试组合：finetune=${finetune_percent}% inference=${infer_percent}%"
        
        log_file="lab13_forward_${finetune_percent}_${infer_percent}.log"
        CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$infer_percent python -m sglang.bench_one_batch \
            --model-path /model/Meta-Llama-3-8B-Instruct/ \
            --batch 1 4 16 \
            --input-len 128 \
            --output-len 2050 \
            --mem-fraction-static 0.8 > $log_file
            
        echo "测试完成！日志保存至: $log_file"
    done

    # 终止训练进程（如果存在）
    if [ -n "$train_pid" ]; then
        echo -e "\n 终止训练进程 PID: $train_pid"
        kill $train_pid
        wait $train_pid 2>/dev/null
        echo " 进程已终止"
    fi

    echo -e "\033[32m===== finetune ${finetune_percent}% 测试完成 =====\n\033[0m"
done

echo -e "\n\033[1;35m所有测试任务已完成！\033[0m"
