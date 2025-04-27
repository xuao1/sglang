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
    
    # 阶段5：数学库预热
    print("预热数学库...")
    a = torch.randn(512, 512, device=device)
    b = torch.randn(512, 512, device=device)
    c = torch.matmul(a, b)  # 显式初始化cuBLAS
    torch.cuda.synchronize()
    print("数学库预热完成")

    # CUDA图捕获
    print("捕获CUDA图...")
    graph = torch.cuda.CUDAGraph()
    print("0")
    with torch.cuda.graph(graph):
        print("1")
        temp = torch.empty(512, device=device)
        print("2")
        c.copy_(a @ b)
        print("3")
        temp.fill_(1.0)
        print("4")
    print("CUDA图捕获完成")

    print("CUDA图重放...")
    # 重放验证
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        print(f"结果范数: {torch.norm(c).item():.4f}")

if __name__ == "__main__":
    print("==== 开始测试 ====")
    test_cuda_graph()
    print("==== 测试完成 ====")
