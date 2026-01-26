import os
import sys
sys.path.append(os.getcwd())
from tape.models.core.LMs.lm_trainer import LMTrainer
from tape.models.core.config import cfg, update_cfg
import pandas as pd

def run(cfg):
    print(f"Using dataset: {cfg.dataset}")  # 打印出使用的数据集
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    all_results = []  # 存储所有评估指标的结果
    for seed in seeds:
        cfg.seed = seed
        trainer = LMTrainer(cfg)
        trainer.train()
        results = trainer.eval_and_save()  # 获取所有评估指标
        all_results.append(results)  # 将每次实验的结果添加到列表中

    if len(all_results) > 1:
        # 将所有结果转换为DataFrame，这样可以对多个实验的结果进行统计
        df = pd.DataFrame(all_results)
        
        # 计算并打印每个评估指标的平均值和标准差
        for k, v in df.items():
            print(f"{k}: {v.mean():.4f} ± {v.std():.4f}")
    else:
        # 如果只有一个结果，直接打印
        print(all_results[0])

if __name__ == '__main__':
    print("LM*******************")
    cfg = update_cfg(cfg)
    run(cfg)