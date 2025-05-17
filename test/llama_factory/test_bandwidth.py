import torch
import time
from torch.profiler import profile, ProfilerActivity

def test_pcie_bandwidth(device='cuda', size_gb=1, num_tests=10):
    # 转换为字节
    size = int(size_gb * 1024**3)  # 1 GB
    dtype = torch.float32  # 4 bytes per element
    num_elements = size // dtype.itemsize
    
    # 创建数据（使用连续内存）
    host_data = torch.rand(num_elements, device='cpu').pin_memory()
    gpu_data = torch.rand(num_elements, device=device)
    
    # 预热
    for _ in range(2):
        _ = host_data.to(device, non_blocking=True)
        _ = gpu_data.cpu()
    torch.cuda.synchronize()
    
    # 测试Host -> Device带宽
    h2d_times = []
    for _ in range(num_tests):
        start = time.perf_counter()
        _ = host_data.to(device, non_blocking=True)
        torch.cuda.synchronize()
        h2d_times.append(time.perf_counter() - start)
    
    # 测试Device -> Host带宽
    d2h_times = []
    for _ in range(num_tests):
        start = time.perf_counter()
        _ = gpu_data.cpu()
        torch.cuda.synchronize()
        d2h_times.append(time.perf_counter() - start)
    
    # 计算结果（GB/s）
    size_gb = size / 1024**3
    h2d_bw = size_gb / (sum(h2d_times)/num_tests)
    d2h_bw = size_gb / (sum(d2h_times)/num_tests)
    
    print(f"测试数据大小: {size_gb:.2f} GB")
    print(f"Host -> Device 带宽: {h2d_bw:.2f} GB/s")
    print(f"Device -> Host 带宽: {d2h_bw:.2f} GB/s")
    print(f"双向平均带宽: {(h2d_bw + d2h_bw)/2:.2f} GB/s")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA不可用，请检查GPU驱动和PyTorch安装")
    else:
        print(f"测试设备: {torch.cuda.get_device_name(0)}")
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], with_stack=True) as prof:
            test_pcie_bandwidth()
        prof.export_chrome_trace(f"/workspace/sglang/test/llama_factory/colocation_overlap_trace.json")
 
