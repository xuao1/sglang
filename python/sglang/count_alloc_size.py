import re
import sys

def count_sizes(filename):
    size_counts = {}
    
    try:
        with open(filename, 'r') as file:
            for line in file:
                # 使用正则表达式匹配目标模式
                # match = re.search(r'In allocate, size\s*=\s*(\d+)', line, re.IGNORECASE)
                match = re.search(r'In my_malloc, size\s*=\s*(\d+)', line, re.IGNORECASE)
                if match:
                    size = int(match.group(1))
                    size_counts[size] = size_counts.get(size, 0) + 1
                    
    except FileNotFoundError:
        print(f"Error: 文件 '{filename}' 未找到")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        return

    # 按数字大小排序结果
    sorted_sizes = sorted(size_counts.items(), key=lambda x: x[0])
    
    # 打印结果
    print("数字统计结果（按大小排序）：")
    print("-" * 30)
    for size, count in sorted_sizes:
        print(f"Size: {size:8d} | 出现次数: {count}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python script.py <文件名>")
        sys.exit(1)
    
    count_sizes(sys.argv[1])
