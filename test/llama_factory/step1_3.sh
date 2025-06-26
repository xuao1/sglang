#!/bin/bash

for batch_val in 16 32 64; do
    for stream_a_val in 12 24 32 44 52 64 76 88 96; do      
        # 计算stream_b的最大值（保留1位小数）
        max_stream_b=$(echo "108 - $stream_a_val" | bc -l | xargs printf "%.1f")
        
        # 生成stream_b的步进序列
        for stream_b_val in 12 24 32 44 52 64 76 88 96; do     
            if (( $(echo "$stream_b_val > $max_stream_b" | bc -l) )); then
                break
            fi
            
            # 修改Python文件中的两个比例参数
            # sed -i "s/\(create_greenctx_stream_by_percent(\)[0-9.]\+,[ ]*[0-9.]\+/\1$stream_a_fmt, $stream_b_fmt/" /workspace/sglang/python/sglang/bench_one_batch.py
            sed -i "s/\(stream_a, stream_b = freeslots.create_greenctx_stream_by_value(\)[0-9.]\+,[ ]*[0-9.]\+/\1$stream_a_val, $stream_b_val/" /workspace/sglang/python/sglang/bench_one_batch.py

            # 记录参数组合
            echo "===== Batch: $batch_val | Stream_a: $stream_a_val | Stream_b: $stream_b_val ===== [时间: $(date +'%T')]"
            echo "===== Batch: $batch_val | Stream_a: $stream_a_val | Stream_b: $stream_b_val =====" >> test1_3.log
            
            # 执行命令（增加超时时间参数示例）
            timeout 300 taskset -c 0,2,4,6,8 python -m sglang.bench_one_batch \
                --model-path /model/Qwen2.5-7B-Instruct/ \
                --batch $batch_val \
                --input-len 128 \
                --output-len 2050 \
                --mem-fraction-static 0.95 \
                --disable-radix-cache >> test1_3.log 2>&1 || true
        done
    done
done