import re
import os

def process_log_file(log_path):
    with open(log_path, 'r') as f:
        content = f.read()

    # 使用正则表达式分割日志块，并捕获batch和stream_a值
    pattern = r'===== Batch: (\d+), Stream_a: ([\d.]+) ====='
    blocks = re.split(pattern, content)
    
    # blocks结构: [前缀, batch1, stream_a1, 内容1, batch2, stream_a2, 内容2...]
    for i in range(1, len(blocks), 3):
        batch = blocks[i]
        stream_a = blocks[i+1]
        block_content = blocks[i+2]
        process_block(batch, stream_a, block_content)

def process_block(batch, stream_a, content):
    decode_lines = re.findall(
        r'Decode\. i:(\d+),\s+latency:\s+([\d.]+)\s+ms', 
        content
    )
    
    if not decode_lines:
        return
    
    # 生成带参数的文件名，例如：32_0.7.csv
    csv_filename = f"{batch}_{stream_a}.csv"
    
    # 写入数据（首次写入时包含header）
    write_header = not os.path.exists(csv_filename)
    with open(csv_filename, 'a') as f:
        if write_header:
            f.write("i,latency(ms)\n")
        for i, latency in decode_lines:
            f.write(f"{i},{latency}\n")

if __name__ == "__main__":
    # 清理旧文件（可选）
    # for f in os.listdir():
    #     if f.endswith('.csv'):
    #         os.remove(f)
    
    process_log_file('test.log')
