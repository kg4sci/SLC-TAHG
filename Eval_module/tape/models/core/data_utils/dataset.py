
import dgl
import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Dict, List, Any

# convert PyG dataset to DGL dataset


class CustomDGLDataset(TorchDataset):
    def __init__(self, name, pyg_data):
        self.name = name
        self.pyg_data = pyg_data

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        data = self.pyg_data
        g = dgl.DGLGraph()
        if self.name == 'ogbn-arxiv' or self.name == 'ogbn-products':
            edge_index = data.edge_index.to_torch_sparse_coo_tensor().coalesce().indices()
        else:
            edge_index = data.edge_index
        g.add_nodes(data.num_nodes)
        g.add_edges(edge_index[0], edge_index[1])

        if data.edge_attr is not None:
            g.edata['feat'] = torch.FloatTensor(data.edge_attr)
        if self.name == 'ogbn-arxiv' or self.name == 'ogbn-products':
            g = dgl.to_bidirected(g)
            print(
                f"Using GAT based methods,total edges before adding self-loop {g.number_of_edges()}")
            g = g.remove_self_loop().add_self_loop()
            print(f"Total edges after adding self-loop {g.number_of_edges()}")
        if data.x is not None:
            g.ndata['feat'] = torch.FloatTensor(data.x)
        g.ndata['label'] = torch.LongTensor(data.y)
        return g

    @property
    def train_mask(self):
        return self.pyg_data.train_mask

    @property
    def val_mask(self):
        return self.pyg_data.val_mask

    @property
    def test_mask(self):
        return self.pyg_data.test_mask


# Create torch dataset for LM finetuning
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['node_id'] = idx
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


class CascadingPathDataset(TorchDataset):
    """
    针对级联预测任务 (rel_AB, rel_BC) 的路径级 Dataset。

    每个样本对应一条 N 元路径，包含：
    - A, B, C, Event: 节点 id
    - y_ab, y_bc: 两个阶段的关系标签 id
    未来 TAPE 侧的级联 Trainer 会直接消费该 Dataset。
    """

    def __init__(self, paths: List[Dict[str, Any]], name_to_id: Dict[str, int]):
        self.A = torch.tensor([p["A"] for p in paths], dtype=torch.long)
        self.B = torch.tensor([p["B"] for p in paths], dtype=torch.long)
        self.C = torch.tensor([p["C"] for p in paths], dtype=torch.long)
        self.Event = torch.tensor(
            [p.get("Event", p["A"]) for p in paths], dtype=torch.long
        )
        self.y_ab = torch.tensor(
            [name_to_id[p["rel_AB"]] for p in paths], dtype=torch.long
        )
        self.y_bc = torch.tensor(
            [name_to_id[p["rel_BC"]] for p in paths], dtype=torch.long
        )

    def __len__(self):
        return self.A.size(0)

    def __getitem__(self, idx):
        return {
            "A": self.A[idx],
            "B": self.B[idx],
            "C": self.C[idx],
            "Event": self.Event[idx],
            "y_ab": self.y_ab[idx],
            "y_bc": self.y_bc[idx],
        }
