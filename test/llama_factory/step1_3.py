import re
import csv
from collections import defaultdict

def parse_log(log_path):
    # 正则表达式匹配模式
    header_pattern = re.compile(
        r'===== Batch: (\d+) \| Stream_a: ([\d.]+) \| Stream_b: ([\d.]+) ====='
    )
    decode_pattern = re.compile(
        r'Decode\. i:(\d+),\s+latency: ([\d.]+) ms'
    )
    
    # 数据结构：{(batch, i, stream_a): {stream_b: latency}}
    data = defaultdict(lambda: defaultdict(dict))
    # 收集所有stream_b值用于列排序
    stream_b_values = set()

    current_batch = None
    current_stream_a = None
    current_stream_b = None

    with open(log_path, 'r') as f:
        for line in f:
            # 匹配参数头
            header_match = header_pattern.match(line.strip())
            if header_match:
                current_batch = header_match.group(1)
                current_stream_a = header_match.group(2)
                current_stream_b = header_match.group(3)
                stream_b_values.add(float(current_stream_b))
                continue

            # 匹配解码延迟
            decode_match = decode_pattern.match(line.strip())
            if decode_match and current_batch:
                i = decode_match.group(1)
                latency = float(decode_match.group(2))
                
                key = (int(current_batch), int(i), float(current_stream_a))
                data[key][float(current_stream_b)] = latency

    return data, sorted(stream_b_values)

def write_csv(data, stream_b_list, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写表头
        header = ['batch', 'i', 'stream_a'] + [f"b_{sb:.1f}" for sb in stream_b_list]
        writer.writerow(header)
        
        # 按自然顺序排序键
        sorted_keys = sorted(data.keys(), key=lambda x: (x[0], x[1], x[2]))
        
        for key in sorted_keys:
            batch, i, stream_a = key
            row = [batch, i, f"{stream_a:.1f}"]
            
            # 按stream_b顺序填充数据
            for sb in stream_b_list:
                row.append(f"{data[key].get(sb, 'N/A')}")
            
            writer.writerow(row)

if __name__ == "__main__":
    input_log = "test1_3.log"
    output_csv = "output.csv"
    
    log_data, stream_b_values = parse_log(input_log)
    write_csv(log_data, stream_b_values, output_csv)
    print(f"解析完成，已输出到 {output_csv}")
