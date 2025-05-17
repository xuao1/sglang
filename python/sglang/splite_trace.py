input_file = 'filtered_output.csv'
output_file = 'filtered_output2.csv'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    # 读取并保留表头
    header = infile.readline()
    outfile.write(header)
    
    # 逐行处理数据部分（从第二行开始）
    line_count = 0
    for line in infile:
        line_count += 1
        # 当行号为奇数时保留（数据行的第1、3、5...行）
        if line_count % 2 == 1:
            outfile.write(line)
