import torch

from tape.models.core.GNNs.gnn_trainer import GNNTrainer
from tape.models.core.GNNs.dgl_gnn_trainer import DGLGNNTrainer
from tape.models.core.data_utils.load import load_data
from sklearn.metrics import f1_score


LOG_FREQ = 10


class EnsembleTrainer():
    def __init__(self, cfg):
        self.cfg = cfg #初始化超参数
        self.device = cfg.device
        self.dataset_name = cfg.dataset
        self.gnn_model_name = cfg.gnn.model.name
        self.lm_model_name = cfg.lm.model.name
        self.hidden_dim = cfg.gnn.model.hidden_dim
        self.num_layers = cfg.gnn.model.num_layers

        self.dropout = cfg.gnn.train.dropout
        self.lr = cfg.gnn.train.lr
        self.feature_type = cfg.gnn.train.feature_type
        self.epochs = cfg.gnn.train.epochs
        self.weight_decay = cfg.gnn.train.weight_decay

        # few-shot 控制（默认 None 不截断）
        self.few_shot_k = getattr(cfg, "few_shot_k", None)
        self.few_shot_balance = getattr(cfg, "few_shot_balance", None)

        # ! Load data
        data, _ = load_data(
            self.dataset_name,
            use_dgl=False,
            use_text=False,
            seed=cfg.seed,
            few_shot_k=self.few_shot_k,
            few_shot_balance=self.few_shot_balance,
        )

        data.y = data.y.squeeze()
        self.data = data.to(self.device)

        from tape.models.core.GNNs.gnn_utils import Evaluator
        self._evaluator = Evaluator(name=self.dataset_name)#创建评估器
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True),
             "y_true": labels.view(-1, 1)}
        )["acc"]

        if cfg.gnn.model.name == 'RevGAT':
            self.TRAINER = DGLGNNTrainer
        else:
            self.TRAINER = GNNTrainer

    @torch.no_grad()
    def _evaluate(self, logits):
        # 获取预测值
        val_preds = logits[self.data.val_mask].argmax(dim=-1)
        test_preds = logits[self.data.test_mask].argmax(dim=-1)

        # 计算准确率
        val_acc = self.evaluator(
            logits[self.data.val_mask], self.data.y[self.data.val_mask])
        test_acc = self.evaluator(
            logits[self.data.test_mask], self.data.y[self.data.test_mask])

        # 计算不同的F1分数
        val_f1 = f1_score(self.data.y[self.data.val_mask].cpu().numpy(), val_preds.cpu().numpy(), average='macro')
        test_f1 = f1_score(self.data.y[self.data.test_mask].cpu().numpy(), test_preds.cpu().numpy(), average='macro')

        # 计算micro F1和weighted F1
        val_microf1 = f1_score(self.data.y[self.data.val_mask].cpu().numpy(), val_preds.cpu().numpy(), average='micro')
        test_microf1 = f1_score(self.data.y[self.data.test_mask].cpu().numpy(), test_preds.cpu().numpy(), average='micro')

        val_weightedf1 = f1_score(self.data.y[self.data.val_mask].cpu().numpy(), val_preds.cpu().numpy(), average='weighted')
        test_weightedf1 = f1_score(self.data.y[self.data.test_mask].cpu().numpy(), test_preds.cpu().numpy(), average='weighted')

        # 计算macro F1（如果需要的话）
        val_macrof1 = f1_score(self.data.y[self.data.val_mask].cpu().numpy(), val_preds.cpu().numpy(), average='macro')
        test_macrof1 = f1_score(self.data.y[self.data.test_mask].cpu().numpy(), test_preds.cpu().numpy(), average='macro')

        # 返回所有评估结果
        return val_acc, test_acc, logits, val_f1, test_f1, val_microf1, test_microf1, val_weightedf1, test_weightedf1, val_macrof1, test_macrof1



    @torch.no_grad()
    def eval(self, logits):
        val_acc, test_acc, _, val_f1, test_f1, val_microf1, test_microf1, val_weightedf1, test_weightedf1, val_macrof1, test_macrof1 = self._evaluate(logits)

        # 打印评估结果
        print(
            f'({self.feature_type}) '
            f'ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}, '
            f'ValF1: {val_f1:.4f}, TestF1: {test_f1:.4f}, '
            f'ValMicroF1: {val_microf1:.4f}, TestMicroF1: {test_microf1:.4f}, '
            f'ValWeightedF1: {val_weightedf1:.4f}, TestWeightedF1: {test_weightedf1:.4f}, '
            f'ValMacroF1: {val_macrof1:.4f}, TestMacroF1: {test_macrof1:.4f}'
        )

        # 将结果保存到字典中
        res = {
            'val_acc': val_acc, 
            'test_acc': test_acc,
            'val_f1': val_f1, 
            'test_f1': test_f1,
            'val_microf1': val_microf1, 
            'test_microf1': test_microf1,
            'val_weightedf1': val_weightedf1, 
            'test_weightedf1': test_weightedf1,
            'val_macrof1': val_macrof1,
            'test_macrof1': test_macrof1
        }

        return res


    def train(self):
        all_pred = []
        all_acc = {}
        feature_types = self.feature_type.split('_')
        for feature_type in feature_types:
            trainer = self.TRAINER(self.cfg, feature_type)
            trainer.train()
            pred, acc = trainer.eval_and_save()
            all_pred.append(pred)
            all_acc[feature_type] = acc

        # 计算集成模型的预测结果
        pred_ensemble = sum(all_pred) / len(all_pred)
        acc_ensemble = self.eval(pred_ensemble)
        all_acc['ensemble'] = acc_ensemble

        # 将所有结果保存到文件
        with open(f'{self.dataset_name}_results.out', 'w') as f:
            for feature_type, acc_results in all_acc.items():
                f.write(f"Feature type: {feature_type}\n")
                for key, value in acc_results.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

        return all_acc


