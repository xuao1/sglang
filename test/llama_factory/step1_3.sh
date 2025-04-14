#!/bin/bash

for batch_val in 1 4 16 32 64; do
    for stream_a_val in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1; do
        # 计算stream_b的最大值（保留1位小数）
        max_stream_b=$(echo "1 - $stream_a_val" | bc -l | xargs printf "%.1f")
        
        # 生成stream_b的步进序列
        for stream_b_val in $(seq 0.1 0.1 $max_stream_b); do
            # 格式化数字（避免出现.0结尾）
            stream_a_fmt=$(printf "%.1f" $stream_a_val)
            stream_b_fmt=$(printf "%.1f" $stream_b_val)
            
            # 修改Python文件中的两个比例参数
            sed -i "s/\(create_greenctx_stream_by_percent(\)[0-9.]\+,[ ]*[0-9.]\+/\1$stream_a_fmt, $stream_b_fmt/" /workspace/sglang/python/sglang/bench_one_batch.py
            
            # 记录参数组合
            echo "===== Batch: $batch_val | Stream_a: $stream_a_fmt | Stream_b: $stream_b_fmt ===== [时间: $(date +'%T')]"
            echo "===== Batch: $batch_val | Stream_a: $stream_a_fmt | Stream_b: $stream_b_fmt =====" >> test.log
            
            # 执行命令（增加超时时间参数示例）
            timeout 300 taskset -c 0,2,4,6,8 python -m sglang.bench_one_batch \
                --model-path /model/Qwen2.5-7B-Instruct/ \
                --batch $batch_val \
                --input-len 128 \
                --output-len 2050 \
                --mem-fraction-static 0.95 \
                --disable-cuda-graph >> test.log 2>&1 || true
        done
    done
done
