import os
from time import time

import numpy as np
import torch
from sklearn.metrics import f1_score

from tape.models.core.GNNs.gnn_utils import EarlyStopping
from tape.models.core.data_utils.load import load_data, load_gpt_preds
from tape.models.core.utils import time_logger

LOG_FREQ = 10


class GNNTrainer():

    def __init__(self, cfg, feature_type):
        # 保留完整 cfg，便于后续读取路径等配置
        self.cfg = cfg
        self.seed = cfg.seed
        self.device = cfg.device
        self.dataset_name = cfg.dataset
        self.gnn_model_name = cfg.gnn.model.name
        self.lm_model_name = cfg.lm.model.name
        self.hidden_dim = cfg.gnn.model.hidden_dim
        self.num_layers = cfg.gnn.model.num_layers
        self.dropout = cfg.gnn.train.dropout
        self.lr = cfg.gnn.train.lr
        requested_feature_type = feature_type
        normalized_feature_type = feature_type
        if isinstance(normalized_feature_type, str) and "_" in normalized_feature_type:
            parts = [part for part in normalized_feature_type.split("_") if part]
            if parts:
                print(
                    f"Composite feature_type '{normalized_feature_type}' detected. "
                    f"Using '{parts[0]}' for single-model GNN training. "
                    "To train an ensemble over multiple feature types, run trainEnsemble.py."
                )
                normalized_feature_type = parts[0]
        self.feature_type = normalized_feature_type
        self.epochs = cfg.gnn.train.epochs

        # few-shot 控制（默认 None 不截断）
        self.few_shot_k = getattr(cfg, "few_shot_k", None)
        self.few_shot_balance = getattr(cfg, "few_shot_balance", None)

        # ! Load data 加载数据
        data, num_classes = load_data(
            self.dataset_name,
            use_dgl=False,
            use_text=False,
            seed=self.seed,
            few_shot_k=self.few_shot_k,
            few_shot_balance=self.few_shot_balance,
        )

        self.num_nodes = data.y.shape[0]
        self.num_classes = num_classes
        data.y = data.y.squeeze()

        # ! Init gnn feature 加载特征
        topk = 3 if self.dataset_name == 'pubmed' else 5
        # 与 LMTrainer 中保持一致：使用 basename 作为模型 tag，
        # 并允许通过 cfg.paths.output_base 控制绝对/相对根目录
        model_tag = os.path.basename(self.lm_model_name.rstrip("/"))
        output_base = getattr(getattr(cfg, "paths", None), "output_base", "")

        features = None
        if self.feature_type == 'ogb':
            print("Loading OGB features...")
            features = data.x
        elif self.feature_type == 'TA':
            print("Loading pretrained LM features (title and abstract) ...")
            LM_emb_path = os.path.join(
                output_base,
                f"prt_lm/{self.dataset_name}/{model_tag}-seed{self.seed}.emb"
            )
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(self.num_nodes, 768)))
            ).to(torch.float32)
        elif self.feature_type == 'E':
            print("Loading pretrained LM features (explanations) ...")
            LM_emb_path = os.path.join(
                output_base,
                f"prt_lm/{self.dataset_name}2/{model_tag}-seed{self.seed}.emb"
            )
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(self.num_nodes, 768)))
            ).to(torch.float32)
        elif self.feature_type == 'P':
            print("Loading top-k prediction features ...")
            features = load_gpt_preds(self.dataset_name, topk)
        else:
            print(
                f'Feature type {self.feature_type} not supported. Loading OGB features...')
            self.feature_type = 'ogb'
            features = data.x

        if features is None:
            raise ValueError(
                "No node features were found for dataset "
                f"'{self.dataset_name}' (requested feature_type='{requested_feature_type}', "
                f"resolved feature_type='{self.feature_type}'). "
                "Please generate the corresponding embeddings (e.g., run trainLM.py) "
                "or choose a supported feature_type with available data.x."
            )

        self.features = features.to(self.device)
        self.data = data.to(self.device)

        # ! Trainer init
        use_pred = self.feature_type == 'P'

        if self.gnn_model_name == "GCN":
            from tape.models.core.GNNs.GCN.model import GCN as GNN
        elif self.gnn_model_name == "SAGE":
            from tape.models.core.GNNs.SAGE.model import SAGE as GNN
        elif self.gnn_model_name == "MLP":
            from tape.models.core.GNNs.MLP.model import MLP as GNN
        else:
            print(f"Model {self.gnn_model_name} is not supported! Loading MLP ...")
            from tape.models.core.GNNs.MLP.model import MLP as GNN

        self.model = GNN(in_channels=self.hidden_dim*topk if use_pred else self.features.shape[1],
                         hidden_channels=self.hidden_dim,
                         out_channels=self.num_classes,
                         num_layers=self.num_layers,
                         dropout=self.dropout,
                         use_pred=use_pred).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=0.0)

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        print(f"\nNumber of parameters: {trainable_params}")
        # GNN 模型保存路径，同样允许通过 cfg.paths.output_base 控制根目录
        gnn_output_base = getattr(getattr(cfg, "paths", None), "output_base", "")
        self.ckpt = os.path.join(
            gnn_output_base,
            f"output/{self.dataset_name}/{self.gnn_model_name}.pt"
        )
        self.stopper = EarlyStopping(
            patience=cfg.gnn.train.early_stop, path=self.ckpt) if cfg.gnn.train.early_stop > 0 else None
        self.loss_func = torch.nn.CrossEntropyLoss()

        from tape.models.core.GNNs.gnn_utils import Evaluator
        self._evaluator = Evaluator(name=self.dataset_name)
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True),
             "y_true": labels.view(-1, 1)}
        )["acc"]

    def _forward(self, x, edge_index):
        logits = self.model(x, edge_index)  # small-graph
        return logits

    def _train(self):
        # ! Shared
        self.model.train()
        self.optimizer.zero_grad()
        # ! Specific
        logits = self._forward(self.features, self.data.edge_index)
        
    
        loss = self.loss_func(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        train_acc = self.evaluator(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()

        return loss.item(), train_acc

    @ torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        logits = self._forward(self.features, self.data.edge_index)
        val_preds = logits[self.data.val_mask].argmax(dim=-1)
        test_preds = logits[self.data.test_mask].argmax(dim=-1)
        val_acc = self.evaluator(
            logits[self.data.val_mask], self.data.y[self.data.val_mask])
        test_acc = self.evaluator(
            logits[self.data.test_mask], self.data.y[self.data.test_mask])
        val_f1 = f1_score(self.data.y[self.data.val_mask].cpu().numpy(), val_preds.cpu().numpy(), average='macro')
        test_f1 = f1_score(self.data.y[self.data.test_mask].cpu().numpy(), test_preds.cpu().numpy(), average='macro')
        return val_acc, test_acc, logits, val_f1, test_f1

    @time_logger
    def train(self):
        # ! Training
        for epoch in range(self.epochs):
            t0, es_str = time(), ''
            loss, train_acc = self._train()
            val_acc, test_acc, _, val_f1, test_f1 = self._evaluate()
            if self.stopper is not None:
                es_flag, es_str = self.stopper.step(val_acc, self.model, epoch)
                if es_flag:
                    print(
                        f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break
            if epoch % LOG_FREQ == 0:
                print(
                    f'Epoch: {epoch}, Time: {time()-t0:.4f}, Loss: {loss:.4f}, TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, ES: {es_str}')

        # ! Finished training, load checkpoints
        if self.stopper is not None:
            self.model.load_state_dict(torch.load(self.stopper.path))

        return self.model

    @ torch.no_grad()
    def eval_and_save(self):
        torch.save(self.model.state_dict(), self.ckpt)
        val_acc, test_acc, logits, val_f1, test_f1 = self._evaluate()

        # 可选导出节点表示（使用最终 logits 作为节点 embedding）
        export_path = getattr(self.cfg.gnn, "export_emb_path", "") if hasattr(self, "cfg") else ""
        if not export_path and hasattr(self, "cfg"):
            # 兼容外部未传 cfg 情况
            export_path = getattr(self.cfg, "export_emb_path", "")
        if export_path:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            emb = np.memmap(export_path, dtype=np.float16, mode='w+', shape=logits.shape)
            emb[:] = logits.cpu().numpy().astype(np.float16)
            print(f"[GNN] Exported node embeddings to {export_path}, shape={emb.shape}")

        print(
            f'[{self.gnn_model_name} + {self.feature_type}] ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n')
        res = {'val_acc': val_acc, 'test_acc': test_acc, 'val_f1': val_f1 ,'test_f1': test_f1}
        return logits, res
