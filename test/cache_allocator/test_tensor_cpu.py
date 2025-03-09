import torch

# 创建大张量
big_tensor = torch.empty(100, 100, dtype=torch.float32)
ptr = big_tensor.data_ptr()
print("ptr is: ", ptr)
bytes_per_element = big_tensor.element_size()  # 4（float32）
print("bytes_per_element is: ", bytes_per_element)

# 创建子张量视图
sub_tensor = big_tensor[10:20, 20:30]
offset_elements = sub_tensor.storage_offset()
print("offset_elements is: ", offset_elements)
offset_bytes = offset_elements * bytes_per_element
print("offset_bytes is: ", offset_bytes)

# 子张量的内存地址
sub_ptr = ptr + offset_bytes
print("sub_ptr is: ", sub_ptr)
print("sub_tensor is: ", sub_tensor)

# 验证指针正确性（通过随机写入）
import ctypes
# 将指针转换为可操作的ctypes对象
array_type = ctypes.c_float * (100 * 10)  # 10x10 的float数组
print("array_type is: ", array_type)
buffer = array_type.from_address(sub_ptr)
print("buffer is: ", buffer)

# 写入数据（例如全1）
for i in range(100 * 10):
    buffer[i] = 1.0

print("sub_tensor is: ", sub_tensor)
# 验证子张量的值是否全为1
print(torch.all(sub_tensor == 1.0))  # 输出：tensor(True)
