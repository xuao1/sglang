import re
import os
import sys

def parse_log_to_csv(log_path):
    # 定义匹配模式
    header_pattern = re.compile(r"^batch_size: (\d+), input_len: 128, output_len: 2048")
    decode_pattern = re.compile(
        r"Decode\. i:(\d+),\s+latency: ([\d.]+) s,\s+throughput:\s+([\d.]+) token/s"
    )

    current_chunk = []
    file_counter = 1
    in_target_section = False

    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            # print(line)
            
            # 检测新块的开始
            if header_pattern.match(line):
                # print(line)
                if current_chunk:
                    write_csv(current_chunk, file_counter)
                    file_counter += 1
                    current_chunk = []
                in_target_section = True
                continue
                
            # 收集目标数据行
            # if in_target_section:
            match = decode_pattern.search(line)
            if match:
                i, latency, throughput = match.groups()
                # print(i, latency, throughput)
                current_chunk.append({
                    'i': int(i),
                    'latency': float(latency),
                    'throughput': float(throughput)
                })
                # else:
                #     # 遇到非Decode行时结束当前块
                #     if current_chunk:
                #         write_csv(current_chunk, file_counter)
                #         file_counter += 1
                #         current_chunk = []
                #     in_target_section = False

        # 处理最后一个块
        if current_chunk:
            write_csv(current_chunk, file_counter)

def write_csv(data, counter):
    filename = f"decode_{counter}.csv"
    with open(filename, 'w') as f:
        # 写入CSV头
        f.write("i,latency(s),throughput(token/s)\n")
        # 写入数据
        for row in data:
            f.write(f"{row['i']},{row['latency']},{row['throughput']}\n")
    print(f"Generated {filename} with {len(data)} records")

# 使用示例
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Missing log file path")
        print(f"Usage: {sys.argv[0]} <log_file_path>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    # 添加文件存在性检查
    if not os.path.exists(log_file):
        print(f"Error: File '{log_file}' not found")
        sys.exit(1)
    
    try:
        parse_log_to_csv(log_file)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)