"""检查当前PyTorch环境是否可以使用GPU"""

import torch

print("=" * 60)
print("PyTorch 设备检查")
print("=" * 60)

# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA 可用: {cuda_available}")

if cuda_available:
    print(f"CUDA 设备数量: {torch.cuda.device_count()}")
    print(f"当前 CUDA 设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")
    
    # 测试创建一个张量
    test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print(f"测试张量设备: {test_tensor.device}")
    print("✓ GPU 可以正常使用")
else:
    print("⚠ 未检测到可用的 GPU，将使用 CPU")
    print("如果您的服务器有 GPU，请检查：")
    print("  1. 是否正确安装了 CUDA 版本的 PyTorch")
    print("  2. CUDA 驱动是否正确安装")
    print("  3. 运行 nvidia-smi 查看 GPU 状态")

print("=" * 60)

# 模拟 resolve_device 的行为
from device_utils import resolve_device
device = resolve_device()
print(f"\nresolve_device() 返回的设备: {device}")
print(f"设备类型: {device.type}")

if device.type == 'cuda':
    print("✓ 模型训练将使用 GPU")
else:
    print("⚠ 模型训练将使用 CPU")

