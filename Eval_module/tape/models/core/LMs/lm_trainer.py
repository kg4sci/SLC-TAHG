import torch
import numpy as np
from typing import Optional, List
import os

from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, IntervalStrategy
from tape.models.core.LMs.model import BertClassifier, BertClaInfModel
from tape.models.core.data_utils.dataset import Dataset
from tape.models.core.data_utils.load import load_data
from tape.models.core.utils import init_path, time_logger
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef



def _safe_roc_auc(y_true, logits):
    try:
        # multi-class: macro One-vs-Rest；二分类时自动使用 positive class 概率
        if logits.shape[1] > 2:
            return roc_auc_score(y_true, logits, multi_class="ovr", average="macro")
        else:
            probs_pos = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1)[:, 1].numpy()
            return roc_auc_score(y_true, probs_pos)
    except Exception:
        return 0.0


def _safe_mcc(y_true, y_pred):
    try:
        return matthews_corrcoef(y_true, y_pred)
    except Exception:
        return 0.0


def compute_metrics(p):
    logits, labels = p
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(y_true=labels, y_pred=preds)
    micro_f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    auc = _safe_roc_auc(labels, logits)
    mcc = _safe_mcc(labels, preds)

    return {
        "accuracy": acc,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "auc": auc,
        "mcc": mcc,
    }

class LMTrainer():
    def __init__(self, cfg):
        self.dataset_name = cfg.dataset
        self.seed = cfg.seed

        self.model_name = cfg.lm.model.name
        self.model_tag = os.path.basename(self.model_name.rstrip("/"))
        self.feat_shrink = cfg.lm.model.feat_shrink

        self.weight_decay = cfg.lm.train.weight_decay
        self.dropout = cfg.lm.train.dropout
        self.att_dropout = cfg.lm.train.att_dropout
        self.cla_dropout = cfg.lm.train.cla_dropout
        self.batch_size = cfg.lm.train.batch_size
        self.epochs = cfg.lm.train.epochs
        self.warmup_epochs = cfg.lm.train.warmup_epochs
        self.eval_patience = cfg.lm.train.eval_patience
        self.grad_acc_steps = cfg.lm.train.grad_acc_steps
        self.lr = cfg.lm.train.lr

        # few-shot 控制（默认 None 不截断，修改对应参数可实现few-shot）
        self.few_shot_k = getattr(cfg, "few_shot_k", None)
        self.few_shot_balance = getattr(cfg, "few_shot_balance", None)

        self.use_gpt_str = "2" if cfg.lm.train.use_gpt else ""
        output_base = getattr(cfg.paths, "output_base", "output")
        tag = f"{self.model_tag}-seed{self.seed}"
        self.output_dir = os.path.join(output_base, f'output/{self.dataset_name}{self.use_gpt_str}/{tag}')
        self.ckpt_dir = os.path.join(output_base, f'prt_lm/{self.dataset_name}{self.use_gpt_str}/{tag}')

        # Preprocess data数据加载和预处理
        data, num_classes, text = load_data(
            dataset=self.dataset_name,
            use_text=True,
            use_gpt=cfg.lm.train.use_gpt,
            seed=self.seed,
            few_shot_k=self.few_shot_k,
            few_shot_balance=self.few_shot_balance,
        )
        self.data = data
        self.num_nodes = data.y.size(0)
        self.n_labels = num_classes
        # AB/BC 证据文本（仅事件节点有，实体节点为空串），供后续独立编码
        self.rel_ab_texts: Optional[List[str]] = getattr(self.data, "rel_ab_texts", None)
        self.rel_bc_texts: Optional[List[str]] = getattr(self.data, "rel_bc_texts", None)

        # if len(self.data.train_mask)== 10:
        #     self.data.train_mask = self.data.train_mask[0]
        #     self.data.val_mask = self.data.val_mask[0]
        #     self.data.test_mask = self.data.test_mask[0]
        # 加载分词器，对文本数据标记化，处理的是摘要数据
        # tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-distilbert-cos-v1")************更改
        # 将文本转换为bert可以理解的序列
        tokenizer = AutoTokenizer.from_pretrained("/mnt/data/lxy/benchmark_paper/Eval_module/tape/models/multi-qa-distilbert-cos-v1")
        # tokenizer = AutoTokenizer.from_pretrained("/mnt/data/zch/projects/models/enhancer/TAPE/multi-qa-distilbert-cos-v1", local_files_only=True)

        
        if type(text)!=list:
            text = text.tolist()
        X = tokenizer(text, padding=True, truncation=True, max_length=512)#最大编码token
        # Dataset 类将文本数据和标签封装为一个数据集对象
        dataset = Dataset(X, data.y.tolist())
        self.inf_dataset = dataset
        # 对训练数据的划分
        self.train_dataset = torch.utils.data.Subset(
            dataset, self.data.train_mask.nonzero().squeeze().tolist())
        self.val_dataset = torch.utils.data.Subset(
            dataset, self.data.val_mask.nonzero().squeeze().tolist())
        self.test_dataset = torch.utils.data.Subset(
            dataset, self.data.test_mask.nonzero().squeeze().tolist())

        # 为 AB/BC 证据文本单独构建 Dataset
        self.inf_dataset_ab = None
        self.inf_dataset_bc = None
        if self.rel_ab_texts and len(self.rel_ab_texts) == self.num_nodes:
            X_ab = tokenizer(self.rel_ab_texts, padding=True, truncation=True, max_length=512)
            self.inf_dataset_ab = Dataset(X_ab, data.y.tolist())
        if self.rel_bc_texts and len(self.rel_bc_texts) == self.num_nodes:
            X_bc = tokenizer(self.rel_bc_texts, padding=True, truncation=True, max_length=512)
            self.inf_dataset_bc = Dataset(X_bc, data.y.tolist())

        # Define pretrained tokenizer and model
        # 模型初始化，预训练bert模型
        # bert_model = AutoModel.from_pretrained(self.model_name)
        # bert_model = AutoModel.from_pretrained("sentence-transformers/multi-qa-distilbert-cos-v1")************更改
        bert_model = AutoModel.from_pretrained("/mnt/data/lxy/benchmark_paper/Eval_module/tape/models/multi-qa-distilbert-cos-v1")
        # 自定义分类器，封装了bert_model,用于将bert的输出映射到分类标签空间
        self.model = BertClassifier(bert_model,
                                    n_labels=self.n_labels,
                                    feat_shrink=self.feat_shrink)# 是否降维

        # prev_ckpt = f'prt_lm/{self.dataset_name}/{self.model_name}.ckpt'
        # if self.use_gpt_str and os.path.exists(prev_ckpt):
        #     print("Initialize using previous ckpt...")
        #     self.model.load_state_dict(torch.load(prev_ckpt))

        self.model.config.dropout = self.dropout
        self.model.config.attention_dropout = self.att_dropout

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(f"\nNumber of parameters: {trainable_params}")

    @time_logger
    def train(self):
        # Define training parameters
        eq_batch_size = self.batch_size * 4
        train_steps = self.num_nodes // eq_batch_size + 1
        eval_steps = self.eval_patience // eq_batch_size
        warmup_steps = int(self.warmup_epochs * train_steps)

        # Define Trainer
        args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=True,
            do_eval=True,
            eval_steps=eval_steps,
            evaluation_strategy=IntervalStrategy.STEPS,
            save_steps=eval_steps,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            save_total_limit=1,
            load_best_model_at_end=True,
            gradient_accumulation_steps=self.grad_acc_steps,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size*8,
            warmup_steps=warmup_steps,
            num_train_epochs=self.epochs,
            dataloader_num_workers=1,
            fp16=True,
            dataloader_drop_last=False,
        )
        #训练过程
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train pre-trained model
        self.trainer.train()
        torch.save(self.model.state_dict(), init_path(f"{self.ckpt_dir}.ckpt"))
        print(f'LM saved to {self.ckpt_dir}.ckpt')

    #评估和保存
    @time_logger
    @torch.no_grad()
    def eval_and_save(self):
        def run_inference(dataset, emb_path, pred_path):
            emb = np.memmap(init_path(emb_path),
                            dtype=np.float16,
                            mode='w+',
                            shape=(self.num_nodes, self.feat_shrink if self.feat_shrink else 768))
            pred = np.memmap(init_path(pred_path),
                             dtype=np.float16,
                             mode='w+',
                             shape=(self.num_nodes, self.n_labels))

            inf_model = BertClaInfModel(
                self.model, emb, pred, feat_shrink=self.feat_shrink)
            inf_model.eval()

            inference_args = TrainingArguments(
                output_dir=self.output_dir,
                do_train=False,
                do_predict=True,
                per_device_eval_batch_size=self.batch_size*8,
                dataloader_drop_last=False,
                dataloader_num_workers=1,
                fp16_full_eval=True,
            )

            trainer = Trainer(model=inf_model, args=inference_args)
            trainer.predict(dataset)
            return emb, pred

        # 主文本（节点属性文本）推理
        emb, pred = run_inference(
            self.inf_dataset,
            f"{self.ckpt_dir}.emb",
            f"{self.ckpt_dir}.pred",
        )

        # 若有 AB/BC 证据文本，分别推理并保存
        emb_ab = pred_ab = emb_bc = pred_bc = None
        if self.inf_dataset_ab is not None:
            emb_ab, pred_ab = run_inference(
                self.inf_dataset_ab,
                f"{self.ckpt_dir}_ab.emb",
                f"{self.ckpt_dir}_ab.pred",
            )
        if self.inf_dataset_bc is not None:
            emb_bc, pred_bc = run_inference(
                self.inf_dataset_bc,
                f"{self.ckpt_dir}_bc.emb",
                f"{self.ckpt_dir}_bc.pred",
            )

        if "ogbn" in self.dataset_name:
            from ogb.nodeproppred import Evaluator
            _evaluator = Evaluator(name=self.dataset_name)
        else:
            from tape.models.core.GNNs.gnn_utils import Evaluator
            _evaluator = Evaluator(name=self.dataset_name)

        def evaluator(preds, labels):
            return _evaluator.eval({
                "y_true": torch.tensor(labels).view(-1, 1),
                "y_pred": torch.tensor(preds).view(-1, 1),
            })["acc"]

        def eval_acc(x):
            return evaluator(np.argmax(pred[x], -1), self.data.y[x])

        def slice_metrics(x_mask):
            labels = self.data.y[x_mask]
            logits_slice = np.array(pred[x_mask], dtype=np.float32)
            preds_slice = np.argmax(logits_slice, axis=1)
            probs_slice = torch.softmax(torch.tensor(logits_slice), dim=1).numpy()

            acc = evaluator(preds_slice, labels)
            macro_f1 = f1_score(labels, preds_slice, average='macro')
            micro_f1 = f1_score(labels, preds_slice, average='micro')
            auc = _safe_roc_auc(labels, probs_slice)
            mcc = _safe_mcc(labels, preds_slice)
            return acc, macro_f1, micro_f1, auc, mcc

        # 计算训练集、验证集和测试集的指标（Acc / F1 / AUC / MCC）
        train_acc, train_macro_f1, train_micro_f1, train_auc, train_mcc = slice_metrics(self.data.train_mask)
        val_acc, val_macro_f1, val_micro_f1, val_auc, val_mcc = slice_metrics(self.data.val_mask)
        test_acc, test_macro_f1, test_micro_f1, test_auc, test_mcc = slice_metrics(self.data.test_mask)

        # 打印结果
        print(
            f'[LM] '
            f'TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}, '
            f'TrainMacroF1: {train_macro_f1:.4f}, ValMacroF1: {val_macro_f1:.4f}, TestMacroF1: {test_macro_f1:.4f}, '
            f'TrainMicroF1: {train_micro_f1:.4f}, ValMicroF1: {val_micro_f1:.4f}, TestMicroF1: {test_micro_f1:.4f}, '
            f'TrainAUC: {train_auc:.4f}, ValAUC: {val_auc:.4f}, TestAUC: {test_auc:.4f}, '
            f'TrainMCC: {train_mcc:.4f}, ValMCC: {val_mcc:.4f}, TestMCC: {test_mcc:.4f}\n'
        )
        
        return {
            'TrainAcc': train_acc, 'ValAcc': val_acc, 'TestAcc': test_acc,
            'TrainMacroF1': train_macro_f1, 'ValMacroF1': val_macro_f1, 'TestMacroF1': test_macro_f1,
            'TrainMicroF1': train_micro_f1, 'ValMicroF1': val_micro_f1, 'TestMicroF1': test_micro_f1,
            'TrainAUC': train_auc, 'ValAUC': val_auc, 'TestAUC': test_auc,
            'TrainMCC': train_mcc, 'ValMCC': val_mcc, 'TestMCC': test_mcc,
            # 记录额外的 emb/pred 路径，方便下游使用
            'EmbMain': f"{self.ckpt_dir}.emb",
            'PredMain': f"{self.ckpt_dir}.pred",
            'EmbAB': f"{self.ckpt_dir}_ab.emb" if self.inf_dataset_ab is not None else None,
            'PredAB': f"{self.ckpt_dir}_ab.pred" if self.inf_dataset_ab is not None else None,
            'EmbBC': f"{self.ckpt_dir}_bc.emb" if self.inf_dataset_bc is not None else None,
            'PredBC': f"{self.ckpt_dir}_bc.pred" if self.inf_dataset_bc is not None else None,
        }
