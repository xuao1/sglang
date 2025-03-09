import torch
import numpy as np
torch.set_printoptions(threshold=np.inf)

cnt = 1

def get_gpu_memory(device):
    """
    Returns the current GPU memory usage in MB.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) # byte
    return 0

class MemoryOwner:
    def __init__(self, size_bytes):
        self._tensor = torch.empty(size_bytes, dtype=torch.uint8, device='cuda')
        self.storage = self._tensor.untyped_storage()
        self.total_bytes = size_bytes
        self.allocated = 0

    def allocate(self, size_bytes, alignment=1):
        aligned_start = (self.allocated + alignment - 1) // alignment * alignment
        if aligned_start + size_bytes > self.total_bytes:
            raise RuntimeError("Insufficient CUDA memory")
        self.allocated = aligned_start + size_bytes
        return aligned_start

    def memory_usage(self):
        return f"Used: {self.allocated}/{self.total_bytes} bytes"

    def print_tensor(self):
        print(self._tensor)

class MemoryUser:
    def __init__(self, memory_owner, num_elements, dtype, init_value):
        self.dtype = dtype
        self.element_size = torch.tensor([], dtype=dtype).element_size()
        print("self.element_size is: ", self.element_size)
        required_bytes = num_elements * self.element_size
        
        self.offset = memory_owner.allocate(required_bytes, alignment=self.element_size)
        
        # 使用正确的位置参数调用顺序
        self.tensor = torch.tensor([], dtype=dtype, device='cuda').set_(
            memory_owner.storage,  # storage参数
            self.offset,           # storage_offset参数
            (num_elements,)        # size参数
        )
        print("In MemoryUser, self.tensor is: ", self.tensor)
        # print("In MemoryUser, memory: ", get_gpu_memory(device='cuda'))
        self.tensor.fill_(init_value)
        print("In MemoryUser, self.tensor is: ", self.tensor)

if __name__ == "__main__":
    # print("Memory: ", get_gpu_memory(device='cuda'))
    owner = MemoryOwner(64)  
    # print("Memory: ", get_gpu_memory(device='cuda'))
    owner.print_tensor()
    
    user1 = MemoryUser(owner, 10, torch.float32, 1.6)   
    # print(f"User1 tensor shape: {user1.tensor.shape}")
    print(f"User1 tensor storage offset: {user1.offset} bytes")
    # print("Memory: ", get_gpu_memory(device='cuda'))
    owner.print_tensor()
    
    user2 = MemoryUser(owner, 20, torch.uint8, 2)
    # print(f"\nUser2 tensor shape: {user2.tensor.shape}")
    print(f"User2 tensor storage offset: {user2.offset} bytes")
    # print("Memory: ", get_gpu_memory(device='cuda'))
    owner.print_tensor()
    
    print(f"\nMemory usage: {owner.memory_usage()}")
