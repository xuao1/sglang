#!/bin/bash

# 外层循环遍历不同的batch值
for batch_val in 1 4 16 32 64; do
    # 内层循环遍历stream_a的比例，从1.0到0.1，每次减少0.1
    for stream_a_val in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1; do        
        # 修改bench_one_batch.py中的stream_a比例
        sed -i "s/\(stream_a, stream_b = freeslots.create_greenctx_stream_by_percent(\)[0-9.]\+/\1$stream_a_val/" /workspace/sglang/python/sglang/bench_one_batch.py
        
        # 输出当前参数到日志文件
        echo "===== Batch: $batch_val, Stream_a: $stream_a_val ===== [时间: $(date +'%T')]"
        echo "===== Batch: $batch_val, Stream_a: $stream_a_val =====" >> test.log
        
        # 执行命令并追加输出到日志，加入容错处理
        taskset -c 0,2,4,6,8 python -m sglang.bench_one_batch \
            --model-path /model/Qwen2.5-7B-Instruct/ \
            --batch $batch_val \
            --input-len 128 \
            --output-len 4096 \
            --mem-fraction-static 0.95 \
            --disable-cuda-graph >> test.log 2>&1 || true
    done
done
