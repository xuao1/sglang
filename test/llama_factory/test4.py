import torch
import os
import ctypes
from torch.cuda.memory import CUDAPluggableAllocator

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

script_dir = os.path.dirname(os.path.abspath(__file__))
alloc_path = os.path.join(script_dir, '/workspace/freeslots/python/freeslots/_C.cpython-310-x86_64-linux-gnu.so')

# 加载自定义分配器
alloc = CUDAPluggableAllocator(
    alloc_path,
    'my_malloc',
    'my_free'
)

# 绑定回调函数
dll = ctypes.CDLL(alloc_path)
alloc._allocator.set_begin_allocate_to_pool(
    ctypes.cast(dll.beginAllocateToPool, ctypes.c_void_p).value
)
alloc._allocator.set_end_allocate_to_pool_fn(
    ctypes.cast(dll.endAllocateToPool, ctypes.c_void_p).value
)
alloc._allocator.set_release_pool(
    ctypes.cast(dll.releasePool, ctypes.c_void_p).value
)

# 设置为当前分配器
torch.cuda.memory.change_current_allocator(alloc)

def test_cuda_graph():
    device = torch.device('cuda')
    
    # ===== 预分配所有内存 =====
    size = 1024
    a = torch.randn(size, device=device)  # 输入向量A
    b = torch.randn(size, device=device)  # 输入向量B
    c = torch.empty_like(a)              # 输出向量C

    # CUDA图捕获
    print("捕获CUDA图...")
    graph = torch.cuda.CUDAGraph()
    print("0")
    with torch.cuda.graph(graph):
        c.copy_(a + b)
        temp = torch.empty(256, device=device)
        temp.fill_(1.0)  # 简单操作验证内存可用性
    print("CUDA图捕获完成")

    print("CUDA图重放...")
    # 重放验证
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        print(f"输出向量范数: {torch.norm(c).item():.4f}")
        print(f"临时内存值: {temp[0].item()}")


if __name__ == "__main__":
    print("==== 开始测试 ====")
    test_cuda_graph()
    print("==== 测试完成 ====")
