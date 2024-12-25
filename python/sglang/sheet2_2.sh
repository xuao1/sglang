#!/bin/bash

# 设置模型路径
MODEL_PATH="/model/Meta-Llama-3-8B-Instruct/"

# 定义批处理大小和输入长度数组
# batch_sizes=(1 2 4 8 16)
# input_lengths=(128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048)
batch_sizes=(16 8 4 2 1)
input_lengths=(2048 1920 1792 1664 1536 1408 1280 1152 1024 896 768 640 512 384 256 128)

# 设定一个足够大的输出长度
output_length=10000000

# 定义输出文件名
output_file=output.txt

# 循环遍历每个批处理大小
for batch in "${batch_sizes[@]}"; do
    # 循环遍历每个输入长度
    for input_len in "${input_lengths[@]}"; do
        echo "Testing with batch_size=$batch and input_len=$input_len" >> "$output_file"

        # 运行推理命令，只将标准输出重定向到文件中
        python -m sglang.bench_one_batch --model-path "$MODEL_PATH" --batch "$batch" --input-len "$input_len" --output-len "$output_length" >> "$output_file"         
        # 检查是否有错误发生
        if [ $? -ne 0 ]; then
            echo "Error occurred with batch_size=$batch and input_len=$input_len" >> "$output_file"
            # 如果发生错误，追加出错信息到文件
            echo "Output of the failed command:" >> "$output_file"
        fi
    done
done
