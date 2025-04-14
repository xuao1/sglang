import re
import csv

def parse_log_to_csv(log_path, csv_path):
    batch_pattern = re.compile(r'===== Batch: (\d+), Stream_a: ([\d.]+) =====')
    decode_pattern = re.compile(r'Decode\. i:(\d+),\s+latency: ([\d.]+) ms')
    
    with open(log_path, 'r') as f_in, open(csv_path, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['Triple (batch,i,stream_a)', 'Latency(ms)'])
        
        current_batch = None
        current_stream_a = None
        
        for line in f_in:
            # 匹配批次信息
            batch_match = batch_pattern.match(line.strip())
            if batch_match:
                current_batch = batch_match.group(1)
                current_stream_a = batch_match.group(2)
                continue
            
            # 匹配延迟信息
            decode_match = decode_pattern.match(line.strip())
            if decode_match and current_batch and current_stream_a:
                i = decode_match.group(1)
                latency = decode_match.group(2)
                
                # 构建三元组字符串
                triple = f'({current_batch},{i},{current_stream_a})'
                writer.writerow([triple, latency])

if __name__ == "__main__":
    parse_log_to_csv('test1_2.log', 'output.csv')
