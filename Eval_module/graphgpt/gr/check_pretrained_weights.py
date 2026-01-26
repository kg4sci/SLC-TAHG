#!/usr/bin/env python3
"""
检查预训练权重文件中的 nan/inf 问题
"""
import torch
import os
import glob
import json
from pathlib import Path

def check_pretrained_weights(pretrain_path):
    """检查预训练权重文件"""
    pretrain_path = Path(pretrain_path)
    
    if not pretrain_path.exists():
        print(f"错误: 路径不存在: {pretrain_path}")
        return False
    
    print(f"检查预训练权重目录: {pretrain_path}")
    print("=" * 60)
    
    # 查找 .pkl 文件
    pkl_files = glob.glob(str(pretrain_path / "*.pkl"))
    if not pkl_files:
        print("未找到 .pkl 文件")
        return False
    
    print(f"找到 {len(pkl_files)} 个 .pkl 文件")
    
    total_invalid = 0
    total_params = 0
    critical_weights = []
    
    for pkl_file in pkl_files:
        print(f"\n检查文件: {os.path.basename(pkl_file)}")
        try:
            state_dict = torch.load(pkl_file, map_location="cpu")
            
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor):
                    total_params += 1
                    nan_count = torch.isnan(v).sum().item()
                    inf_count = torch.isinf(v).sum().item()
                    
                    if nan_count > 0 or inf_count > 0:
                        total_invalid += 1
                        total_elements = v.numel()
                        invalid_ratio = (nan_count + inf_count) / total_elements
                        
                        print(f"  ⚠️  {k}:")
                        print(f"     形状: {v.shape}")
                        print(f"     nan: {nan_count} ({nan_count/total_elements:.2%})")
                        print(f"     inf: {inf_count} ({inf_count/total_elements:.2%})")
                        print(f"     总无效率: {invalid_ratio:.2%}")
                        
                        if invalid_ratio > 0.5:
                            critical_weights.append((k, invalid_ratio))
                            print(f"     ⚠️⚠️⚠️  严重损坏！超过 50% 的值无效！")
                        elif invalid_ratio > 0.1:
                            print(f"     ⚠️⚠️  中度损坏！超过 10% 的值无效！")
        except Exception as e:
            print(f"  错误: 无法加载文件: {e}")
    
    print("\n" + "=" * 60)
    print(f"总结:")
    print(f"  总参数数量: {total_params}")
    print(f"  有问题的参数: {total_invalid}")
    print(f"  问题参数比例: {total_invalid/total_params:.2%}" if total_params > 0 else "  问题参数比例: N/A")
    
    if critical_weights:
        print(f"\n⚠️  严重损坏的权重（>50% 无效）:")
        for k, ratio in critical_weights:
            print(f"    - {k}: {ratio:.2%}")
        print("\n建议: 这些权重将被完全重新初始化")
    
    return total_invalid == 0

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python check_pretrained_weights.py <pretrain_path>")
        print("示例: python check_pretrained_weights.py ./clip_gt_arxiv")
        sys.exit(1)
    
    pretrain_path = sys.argv[1]
    is_ok = check_pretrained_weights(pretrain_path)
    
    if not is_ok:
        print("\n⚠️  警告: 预训练权重文件包含 nan/inf 值！")
        print("   代码会自动修复这些问题，但建议检查权重文件的来源。")
    else:
        print("\n✓ 预训练权重文件正常，没有发现 nan/inf 值。")
    
    sys.exit(0 if is_ok else 1)

