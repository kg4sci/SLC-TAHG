import os
import sys
sys.path.append(os.getcwd())

from core.config import cfg, update_cfg
from core.GNNs.ensemble_trainer import EnsembleTrainer
import pandas as pd

import time

import csv

from sklearn.metrics import f1_score


def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    all_acc = []
    start = time.time()
    results = []

    for seed in seeds:
        cfg.seed = seed
        ensembler = EnsembleTrainer(cfg)
        acc = ensembler.train()

        # 保存到结果列表中
        results.append(acc)

        # 打印并保存评估结果
        print(f"[{cfg.dataset}] Finished Training with Seed {cfg.seed}")
        for feature_type, acc_results in acc.items():
            print(f"Feature type: {feature_type}")
            print(f"Acc results keys: {acc_results.keys()}")  # 打印字段名，确保包含所有指标
            for key, value in acc_results.items():
                print(f"{key}: {value}")

        # 定义 CSV 文件的字段名，确保包含所有评估指标字段
        fieldnames = [
            'dataset', 'num_layers', 'hidden_dim', 'dropout', 'test_acc',
            'test_macrof1', 'test_microf1', 'test_weightedf1', 'test_macrof1', 
            'val_acc', 'val_f1', 'val_microf1', 'val_weightedf1', 'val_macrof1'
        ]

        # 将结果保存在 CSV 文件中
        with open(f'{cfg.dataset}_results.csv', 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            # 如果是首次写入文件，写入字段名
            if file.tell() == 0:  # 文件为空，写入表头
                writer.writeheader()

            # 将评估结果写入 CSV 文件
            for feature_type, acc_results in acc.items():
                row = {
                    'dataset': cfg.dataset,
                    'num_layers': cfg.gnn.model.num_layers,
                    'hidden_dim': cfg.gnn.model.hidden_dim,
                    'dropout': cfg.gnn.train.dropout,
                    'test_acc': acc_results['test_acc'],
                    'test_macrof1': acc_results['test_macrof1'],
                    'test_microf1': acc_results['test_microf1'],
                    'test_weightedf1': acc_results['test_weightedf1'],
                    'test_macrof1': acc_results['test_macrof1'],
                    'val_acc': acc_results['val_acc'],
                    'val_f1': acc_results['val_f1'],
                    'val_microf1': acc_results['val_microf1'],
                    'val_weightedf1': acc_results['val_weightedf1'],
                    'val_macrof1': acc_results['val_macrof1'],
                }
                writer.writerow(row)

    end = time.time()

    # 打印总的运行时间
    print(f"Running time: {round((end-start)/len(seeds), 2)}s")



if __name__ == '__main__':
    cfg = update_cfg(cfg)
    print(cfg.seed, "+++++++++++cfg.seed")
    run(cfg)
