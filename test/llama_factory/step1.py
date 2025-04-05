import re
import os

def process_log_file(log_path):
    with open(log_path, 'r') as f:
        content = f.read()

    # 用正则表达式分割不同batch_size的块
    blocks = re.split(r'batch_size: (\d+), input_len: \d+, output_len: \d+', content)
    
    current_bs = None
    for i in range(1, len(blocks)):
        if i % 2 == 1:  # 奇数索引是batch_size值
            current_bs = blocks[i]
        else:           # 偶数索引是对应的日志内容
            process_block(current_bs, blocks[i])

def process_block(bs, block_content):
    decode_lines = re.findall(
        r'Decode\. i:(\d+),\s+latency:\s+([\d.]+)\s+ms', 
        block_content
    )
    
    if not decode_lines:
        return
    
    csv_filename = f"{bs}.csv"
    # 追加模式写入，防止多次运行重复覆盖
    with open(csv_filename, 'a') as f:
        for i, latency in decode_lines:
            f.write(f"{i},{latency}\n")

if __name__ == "__main__":
    # 先清理可能存在的旧文件（如果需要）
    # for f in os.listdir():
    #     if f.endswith('.csv'):
    #         os.remove(f)
    
    process_log_file('step1_llama.log')
