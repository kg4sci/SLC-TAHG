import os
import json
import torch
import csv
from tape.models.core.data_utils.dataset import CustomDGLDataset


def load_gpt_preds(dataset, topk):
    preds = []
    # fn = f'gpt_preds/{dataset}.csv'**********************更改
    fn = f'{dataset}.csv'
    print(f"Loading topk preds from {fn}")
    with open(fn, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            inner_list = []
            for value in row:
                inner_list.append(int(value))
            preds.append(inner_list)

    pl = torch.zeros(len(preds), topk, dtype=torch.long)
    for i, pred in enumerate(preds):
        pl[i][:len(pred)] = torch.tensor(pred[:topk], dtype=torch.long)+1
    return pl


def load_data(
    dataset,
    use_dgl=False,
    use_text=False,
    use_gpt=False,
    seed=0,
    few_shot_k=None,
    few_shot_balance=None,
):
    if dataset == 'SLC_database':
        from tape.models.core.data_utils.load_slc_database import get_raw_text_slc_database as get_raw_text
        # 节点标签类别数来自 Neo4j 标签，先设占位；实际由 loader 内部确定 y，但这里保持兼容返回 num_classes
        num_classes = None
    else:
        exit(f'Error: Dataset {dataset} not supported')

    # for training GNN
    if not use_text:
        data, _ = get_raw_text(use_text=False, seed=seed, few_shot_k=few_shot_k, few_shot_balance=few_shot_balance)
        if num_classes is None and hasattr(data, "y") and data.y is not None:
            num_classes = int(torch.unique(data.y).numel())
        if use_dgl:
            data = CustomDGLDataset(dataset, data)
        return data, num_classes

    #****************llama评估用下面
    if use_gpt:
        # 1) 先加载原始 data 和 total count
        data, _ = get_raw_text(use_text=False, seed=seed, few_shot_k=few_shot_k, few_shot_balance=few_shot_balance)
        if num_classes is None and hasattr(data, "y") and data.y is not None:
            num_classes = int(torch.unique(data.y).numel())
        folder_path = os.path.join('/mnt/data/lxy/benchmark_paper/llama_response', dataset)
        print(f"Using GPT outputs from: {folder_path}")

        # 2) 列出所有已有的 json 文件，提取出它们的索引
        all_files = [
            f for f in os.listdir(folder_path)
            if f.endswith('.json') and f[:-5].isdigit()
        ]
        available_idxs = sorted(int(f[:-5]) for f in all_files)
        print(f"Found {len(available_idxs)} GPT files / {data.y.shape[0]} total samples")

        # 3) 只加载这些已有的文件，并同步构建 text 列表
        text = []
        for idx in available_idxs:
            fp = os.path.join(folder_path, f"{idx}.json")
            with open(fp, 'r') as fin:
                js = json.load(fin)
                text.append(js.get('answer', ''))

        # 4) 把 data.y（和 data.x，如果有的话）也只保留这些索引
        data.y = data.y[available_idxs]
        if hasattr(data, 'x') and data.x is not None:
            data.x = data.x[available_idxs]
        # 同步裁剪 mask，保持长度一致
        for mask_name in ['train_mask', 'val_mask', 'test_mask']:
            if hasattr(data, mask_name):
                m = getattr(data, mask_name)
                if m is not None:
                    setattr(data, mask_name, m[available_idxs])
        # 更新 num_nodes
        if hasattr(data, 'num_nodes'):
            data.num_nodes = len(available_idxs)
    else:
        # 标准文本加载
        data, text = get_raw_text(use_text=True, seed=seed, few_shot_k=few_shot_k, few_shot_balance=few_shot_balance)
        if num_classes is None and hasattr(data, "y") and data.y is not None:
            num_classes = int(torch.unique(data.y).numel())

    # 最后返回 data、类别数和 text
    return data, num_classes, text