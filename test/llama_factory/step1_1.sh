#!/bin/bash

# 定义batch size数组
bs_values=(1 2 3)
bs_values+=($(seq 4 2 64))  # 生成4到64的偶数序列

# 清空旧日志文件（可选）
# > test.log

for bs in "${bs_values[@]}"; do
    echo "正在运行 batch_size: $bs [时间: $(date +'%T')]"
    
    # 执行命令并添加错误处理
    if ! python -m sglang.bench_one_batch \
        --model-path /model/Qwen2.5-7B-Instruct/ \
        --batch $bs \
        --input-len 128 \
        --output-len 4096 \
        --mem-fraction-static 0.95 \
        --disable-cuda-graph >> test.log 2>&1
    then
        # 记录错误信息到日志文件
        echo "[ERROR] batch_size $bs 执行失败 at $(date +'%F %T')" >> test.log
        # 在终端显示警告
        echo -e "\033[31m警告：batch_size $bs 运行失败，继续下一个...\033[0m" >&2
    fi
    
    # 添加分隔线到日志
    echo "--------------------------------------------------" >> test.log
done

echo "所有测试任务完成，最终结果见 test.log"
