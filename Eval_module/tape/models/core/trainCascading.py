import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from typing import Dict, List, Optional
from tape.models.core.config import cfg, update_cfg
from tape.models.core.utils import init_random_state, mkdir_p
from tape.models.core.data_utils.load_hype_paths import load_hype_cascading_paths
from tape.models.core.data_utils.dataset import CascadingPathDataset
from tape.models.core.cascading_model import CascadingModel
from Eval_module.path_data import select_few_shot


def _load_memmap_auto(path, num_nodes):
    if not path or not os.path.isfile(path):
        return None
    size_bytes = os.path.getsize(path)
    dim = size_bytes // num_nodes // 2  # float16 -> 2 bytes
    try:
        arr = np.memmap(path, mode="r", dtype=np.float16, shape=(num_nodes, dim))
        return torch.from_numpy(np.array(arr)).to(torch.float32)
    except Exception as e:
        print(f"[Cascade] Failed to load memmap {path}: {e}")
        return None


def build_dataloader(paths, name_to_id, batch_size, shuffle=False):
    dataset = CascadingPathDataset(paths, name_to_id)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, device, id_to_name: Optional[Dict[int, str]] = None):
    model.eval()
    all_true_ab, all_true_bc = [], []
    all_pred_ab, all_pred_bc = [], []
    all_prob_ab, all_prob_bc = [], []
    all_path_true, all_path_pred = [], []
    with torch.no_grad():
        for batch in loader:
            A = batch["A"].to(device)
            B = batch["B"].to(device)
            C = batch["C"].to(device)
            Event = batch["Event"].to(device)
            y_ab = batch["y_ab"].to(device)
            y_bc = batch["y_bc"].to(device)

            logits_ab, logits_bc, _ = model(A, B, C, Event, training=False)
            pred_ab = torch.argmax(logits_ab, dim=-1)
            pred_bc = torch.argmax(logits_bc, dim=-1)
            prob_ab = torch.softmax(logits_ab, dim=-1).cpu()
            prob_bc = torch.softmax(logits_bc, dim=-1).cpu()

            all_true_ab.append(y_ab.cpu())
            all_true_bc.append(y_bc.cpu())
            all_pred_ab.append(pred_ab.cpu())
            all_pred_bc.append(pred_bc.cpu())
            all_prob_ab.append(prob_ab)
            all_prob_bc.append(prob_bc)
            # path 级别二分类：两段都预测正确记为 1，否则 0
            all_path_true.append(((y_ab == y_ab) & (y_bc == y_bc)).cpu())  # 全 1 占位
            all_path_pred.append(((pred_ab == y_ab) & (pred_bc == y_bc)).cpu().long())

    true_ab = torch.cat(all_true_ab).numpy()
    true_bc = torch.cat(all_true_bc).numpy()
    pred_ab = torch.cat(all_pred_ab).numpy()
    pred_bc = torch.cat(all_pred_bc).numpy()
    prob_ab = torch.cat(all_prob_ab).numpy()
    prob_bc = torch.cat(all_prob_bc).numpy()

    # 基本准确率
    ab_acc = (true_ab == pred_ab).mean()
    bc_acc = (true_bc == pred_bc).mean()
    path_acc = ((true_ab == pred_ab) & (true_bc == pred_bc)).mean()

    # 四类路径情况：0=(T,T), 1=(T,F), 2=(F,T), 3=(F,F)
    is_ab_correct = (true_ab == pred_ab)
    is_bc_correct = (true_bc == pred_bc)
    path_case = np.full_like(true_ab, fill_value=3)
    path_case[is_ab_correct & is_bc_correct] = 0  # (T,T)
    path_case[is_ab_correct & (~is_bc_correct)] = 1  # (T,F)
    path_case[(~is_ab_correct) & is_bc_correct] = 2  # (F,T)
    # 剩余保持为 3 (F,F)

    # 各类比例，便于分析 AB 误差对 BC 预测的影响
    total_paths = len(true_ab)
    frac_TT = (path_case == 0).sum() / total_paths
    frac_TF = (path_case == 1).sum() / total_paths
    frac_FT = (path_case == 2).sum() / total_paths
    frac_FF = (path_case == 3).sum() / total_paths
    ab_f1 = f1_score(true_ab, pred_ab, average="macro", zero_division=0)
    bc_f1 = f1_score(true_bc, pred_bc, average="macro", zero_division=0)
    # path_f1（binary）：两段都对为正类
    path_true_bin = np.ones_like(true_ab)  # all ones (参考 HypE 只统计“全对”为正)
    path_pred_bin = ((true_ab == pred_ab) & (true_bc == pred_bc)).astype(np.int64)
    path_f1 = f1_score(path_true_bin, path_pred_bin, average="macro", zero_division=0)
    # AUC（仅在类别数>1 且标签多样时计算），二分类取正类概率，多分类 macro-ovr
    try:
        ab_auc = roc_auc_score(true_ab, prob_ab[:, 1] if prob_ab.shape[1] > 1 else prob_ab[:, 0],
                               multi_class="ovr" if prob_ab.shape[1] > 2 else "raise")
    except Exception:
        ab_auc = 0.0
    try:
        bc_auc = roc_auc_score(true_bc, prob_bc[:, 1] if prob_bc.shape[1] > 1 else prob_bc[:, 0],
                               multi_class="ovr" if prob_bc.shape[1] > 2 else "raise")
    except Exception:
        bc_auc = 0.0
    # MCC
    try:
        ab_mcc = matthews_corrcoef(true_ab, pred_ab)
    except Exception:
        ab_mcc = 0.0
    try:
        bc_mcc = matthews_corrcoef(true_bc, pred_bc)
    except Exception:
        bc_mcc = 0.0

    labels_ab = sorted(set(true_ab) | set(pred_ab))
    ab_cm = confusion_matrix(true_ab, pred_ab, labels=labels_ab).tolist()
    ab_prec, ab_rec, ab_f1_cls, ab_sup = precision_recall_fscore_support(
        true_ab,
        pred_ab,
        labels=labels_ab,
        zero_division=0,
    )
    ab_bal_acc = balanced_accuracy_score(true_ab, pred_ab) if len(labels_ab) > 1 else float(ab_acc)
    ab_label_names = [str((id_to_name or {}).get(int(lbl), str(lbl))) for lbl in labels_ab]

    labels_bc = sorted(set(true_bc) | set(pred_bc))
    bc_cm = confusion_matrix(true_bc, pred_bc, labels=labels_bc).tolist()
    bc_prec, bc_rec, bc_f1_cls, bc_sup = precision_recall_fscore_support(
        true_bc,
        pred_bc,
        labels=labels_bc,
        zero_division=0,
    )
    bc_bal_acc = balanced_accuracy_score(true_bc, pred_bc) if len(labels_bc) > 1 else float(bc_acc)
    bc_label_names = [str((id_to_name or {}).get(int(lbl), str(lbl))) for lbl in labels_bc]

    return {
        "ab_acc": ab_acc,
        "bc_acc": bc_acc,
        "ab_f1": ab_f1,
        "bc_f1": bc_f1,
        "path_acc": path_acc,
        "path_f1": path_f1,
        "ab_auc": ab_auc,
        "bc_auc": bc_auc,
        "ab_mcc": ab_mcc,
        "bc_mcc": bc_mcc,
        # 路径四分类比例
        "path_frac_TT": frac_TT,  # AB 对 & BC 对
        "path_frac_TF": frac_TF,  # AB 对 & BC 错
        "path_frac_FT": frac_FT,  # AB 错 & BC 对
        "path_frac_FF": frac_FF,  # AB 错 & BC 错
        "ab_labels": labels_ab,
        "ab_label_names": ab_label_names,
        "ab_confusion_matrix": ab_cm,
        "ab_precision_by_class": ab_prec.tolist(),
        "ab_recall_by_class": ab_rec.tolist(),
        "ab_f1_by_class": ab_f1_cls.tolist(),
        "ab_support_by_class": ab_sup.tolist(),
        "ab_balanced_acc": float(ab_bal_acc),
        "bc_labels": labels_bc,
        "bc_label_names": bc_label_names,
        "bc_confusion_matrix": bc_cm,
        "bc_precision_by_class": bc_prec.tolist(),
        "bc_recall_by_class": bc_rec.tolist(),
        "bc_f1_by_class": bc_f1_cls.tolist(),
        "bc_support_by_class": bc_sup.tolist(),
        "bc_balanced_acc": float(bc_bal_acc),
    }


def _build_relation_maps_from_paths(paths: List[Dict]) -> Dict[str, int]:
    rel_types = set()
    for p in paths:
        rel_ab = p.get("rel_AB")
        rel_bc = p.get("rel_BC")
        if rel_ab is not None:
            rel_types.add(rel_ab)
        if rel_bc is not None:
            rel_types.add(rel_bc)
    sorted_types = sorted(rel_types)
    return {name: idx for idx, name in enumerate(sorted_types)}


def train(
    cfg,
    *,
    train_paths: Optional[List[Dict]] = None,
    val_paths: Optional[List[Dict]] = None,
    test_paths: Optional[List[Dict]] = None,
    few_shot_k: Optional[int] = None,
    few_shot_balance: Optional[str] = None,
):
    device = f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu"
    init_random_state(cfg.seed)

    # 1) 加载路径数据（只依赖根目录 path_data）或使用外部切分
    if train_paths is None or val_paths is None:
        paths_dict = load_hype_cascading_paths(
            cache_file="hype_paths.pt",
            use_cache=False,
        )
        train_split = paths_dict["train"]
        val_split = paths_dict["val"]
        test_split = paths_dict["test"]
        name_to_id = paths_dict["name_to_id"]
    else:
        train_split = list(train_paths)
        # few-shot 仅截断训练正样本，保持验证/测试与图结构不变
        if few_shot_k:
            before = len(train_split)
            train_split = select_few_shot(train_split, few_shot_k, seed=cfg.seed, balance_by=few_shot_balance)
            after = len(train_split)
            print(f"[Cascade] few_shot_k={few_shot_k}, balance_by={few_shot_balance}, train size: {before} -> {after}")

        val_split = val_paths
        test_split = test_paths
        combined_paths = train_split + val_split + [p for p in test_split if p not in val_split]
        if not combined_paths:
            raise ValueError("[Cascade] No paths provided for external training splits.")

        name_to_id = _build_relation_maps_from_paths(combined_paths)
        if not name_to_id:
            raise ValueError("[Cascade] Unable to build relation mappings from provided paths.")

        paths_dict = {
            "train": train_split,
            "val": val_split,
            "test": test_split,
            "name_to_id": name_to_id,
        }

    num_rel_classes = len(name_to_id)
    id_to_name = {idx: name for name, idx in name_to_id.items()}

    train_loader = build_dataloader(
        paths_dict["train"], name_to_id, batch_size=cfg.cascade.batch_size, shuffle=True
    )
    val_loader = build_dataloader(
        paths_dict["val"], name_to_id, batch_size=cfg.cascade.batch_size, shuffle=False
    )
    test_loader = build_dataloader(
        paths_dict["test"], name_to_id, batch_size=cfg.cascade.batch_size, shuffle=False
    )

    # 2) 统计节点总数
    all_ids = []
    for split_name in ("train", "val", "test"):
        for p in paths_dict[split_name]:
            all_ids.extend([p["A"], p["B"], p["C"], p.get("Event", p["A"])])
    num_nodes = max(all_ids) + 1
    event_indices = torch.tensor(sorted({p.get("Event", p["A"]) for s in ("train", "val", "test") for p in paths_dict[s]}), dtype=torch.long)

    # 3) 加载节点特征（base / ab / bc / gnn）
    base_feat = ab_feat = bc_feat = gnn_feat = None
    lm_base = getattr(cfg.cascade, "lm_base", getattr(cfg.paths, "output_base", ""))
    model_tag = os.path.basename(cfg.lm.model.name.rstrip("/"))
    tag = f"{model_tag}-seed{cfg.seed}"
    lm_emb_base = os.path.join(lm_base, f"prt_lm/{cfg.dataset}/{tag}.emb")
    lm_emb_ab = os.path.join(lm_base, f"prt_lm/{cfg.dataset}/{tag}_ab.emb")
    lm_emb_bc = os.path.join(lm_base, f"prt_lm/{cfg.dataset}/{tag}_bc.emb")

    if cfg.cascade.use_node_feat:
        # base 优先用显式路径，否则用 LM 主文本 emb
        if cfg.cascade.node_feat_path:
            try:
                base_feat = torch.load(cfg.cascade.node_feat_path)
                if isinstance(base_feat, np.ndarray):
                    base_feat = torch.from_numpy(base_feat)
                base_feat = base_feat.to(torch.float32)
                if base_feat.size(0) < num_nodes:
                    pad_rows = num_nodes - base_feat.size(0)
                    base_feat = torch.cat([base_feat, torch.zeros(pad_rows, base_feat.size(1))], dim=0)
                print(f"[Cascade] base_feat loaded from {cfg.cascade.node_feat_path}, shape={tuple(base_feat.shape)}")
            except Exception as e:
                print(f"[Cascade] Failed to load node_feat_path {cfg.cascade.node_feat_path}: {e}")

        if base_feat is None:
            base_feat = _load_memmap_auto(lm_emb_base, num_nodes)
            if base_feat is not None:
                print(f"[Cascade] base_feat loaded from {lm_emb_base}, shape={tuple(base_feat.shape)}")

        # 可控：是否使用基于文本的 AB/BC 证据特征
        if getattr(cfg.cascade, "use_text_feature", True):
            ab_feat = _load_memmap_auto(lm_emb_ab, num_nodes)
            bc_feat = _load_memmap_auto(lm_emb_bc, num_nodes)
        else:
            ab_feat = None
            bc_feat = None
        # 可选 GNN embedding（用户提供），会加到 base_feat 上
        gnn_feat = _load_memmap_auto(cfg.cascade.gnn_feat_path, num_nodes) if cfg.cascade.gnn_feat_path else None

        # 只保留事件节点的 ab/bc 特征，其余置零
        if ab_feat is not None:
            mask = torch.zeros_like(ab_feat)
            mask[event_indices] = ab_feat[event_indices]
            ab_feat = mask
            print(f"[Cascade] ab_feat loaded (event-only), shape={tuple(ab_feat.shape)}")
        if bc_feat is not None:
            mask = torch.zeros_like(bc_feat)
            mask[event_indices] = bc_feat[event_indices]
            bc_feat = mask
            print(f"[Cascade] bc_feat loaded (event-only), shape={tuple(bc_feat.shape)}")

        # 将 GNN embedding 叠加到 base_feat（需同维度）
        if gnn_feat is not None:
            if base_feat is not None and gnn_feat.size(1) == base_feat.size(1):
                base_feat = base_feat + gnn_feat
                print(f"[Cascade] gnn_feat added to base_feat, shape={tuple(base_feat.shape)}")
            elif base_feat is None:
                base_feat = gnn_feat
                print(f"[Cascade] base_feat replaced by gnn_feat, shape={tuple(base_feat.shape)}")
            else:
                print(f"[Cascade] gnn_feat dim {gnn_feat.size(1)} != base_feat dim {base_feat.size(1)}, skip add.")

        if base_feat is None:
            print("[Cascade] No base_feat loaded; using pure id embeddings.")

    # 4) 构建模型
    model = CascadingModel(
        num_nodes=num_nodes,
        num_rel_classes=num_rel_classes,
        emb_dim=cfg.cascade.emb_dim,
        hidden_dim=cfg.cascade.hidden_dim,
        dropout=cfg.cascade.dropout,
        node_feat_base=base_feat.to(device) if base_feat is not None else None,
        node_feat_ab=ab_feat.to(device) if ab_feat is not None else None,
        node_feat_bc=bc_feat.to(device) if bc_feat is not None else None,
        use_node_feat=cfg.cascade.use_node_feat,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.cascade.lr)
    ce_loss = torch.nn.CrossEntropyLoss()

    best_path_acc = 0.0
    best_metrics = None  # 记录最优 path_acc 对应的一整套指标（val & test）
    save_dir = "/mnt/data/lxy/benchmark_paper/Eval_module/tape/models/output_data/output/cascade"
    mkdir_p(save_dir)
    best_ckpt = os.path.join(save_dir, "best.pt")

    for epoch in range(cfg.cascade.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            A = batch["A"].to(device)
            B = batch["B"].to(device)
            C = batch["C"].to(device)
            Event = batch["Event"].to(device)
            y_ab = batch["y_ab"].to(device)
            y_bc = batch["y_bc"].to(device)

            logits_ab, logits_bc, _ = model(A, B, C, Event, training=True)
            loss_ab = ce_loss(logits_ab, y_ab)
            loss_bc = ce_loss(logits_bc, y_bc)
            loss = loss_ab + loss_bc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * A.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        if (epoch + 1) % cfg.cascade.val_every == 0:
            val_metrics = evaluate(model, val_loader, device, id_to_name=id_to_name)
            test_metrics = evaluate(model, test_loader, device, id_to_name=id_to_name)
            path_acc = val_metrics["path_acc"]

            def _coerce_metric_value(value):
                if isinstance(value, (np.floating, np.integer)):
                    return value.item()
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, dict):
                    return {k: _coerce_metric_value(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return [_coerce_metric_value(v) for v in value]
                if isinstance(value, (float, int, bool)) or value is None:
                    return value
                try:
                    return float(value)
                except Exception:
                    return value

            print(
                f"[Epoch {epoch+1}/{cfg.cascade.epochs}] "
                f"Loss: {avg_loss:.4f} | \n"
                f"Val path_acc: {path_acc:.4f},Val path_f1: {val_metrics['path_f1']:.4f} | \n"
                f"ab_acc: {val_metrics['ab_acc']:.4f}, bc_acc: {val_metrics['bc_acc']:.4f},\n "
                f"ab_f1: {val_metrics['ab_f1']:.4f}, bc_f1: {val_metrics['bc_f1']:.4f}"
                # f"ab_auc: {val_metrics['ab_auc']:.4f}, bc_auc: {val_metrics['bc_auc']:.4f}, "
                # f"ab_mcc: {val_metrics['ab_mcc']:.4f}, bc_mcc: {val_metrics['bc_mcc']:.4f}"
            )

            if path_acc > best_path_acc:
                best_path_acc = path_acc
                # 记录此时的完整指标（val / test）
                best_metrics = {
                    "epoch": epoch + 1,
                    "val": {k: _coerce_metric_value(v) for k, v in val_metrics.items()},
                    "test": {k: _coerce_metric_value(v) for k, v in test_metrics.items()},
                }
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "name_to_id": name_to_id,
                        "config": cfg,
                    },
                    best_ckpt,
                )
                print(f"[Cascade] New best path_acc={best_path_acc:.4f}, saved to {best_ckpt}")
            # 也打印一下 test，便于观察
            print(
                f"[Test] path_acc: {test_metrics['path_acc']:.4f}, path_f1: {test_metrics['path_f1']:.4f} | \n"
                f"ab_acc: {test_metrics['ab_acc']:.4f}, bc_acc: {test_metrics['bc_acc']:.4f}, \n"
                f"ab_f1: {test_metrics['ab_f1']:.4f}, bc_f1: {test_metrics['bc_f1']:.4f}"
                # f"ab_auc: {test_metrics['ab_auc']:.4f}, bc_auc: {test_metrics['bc_auc']:.4f}, "
                # f"ab_mcc: {test_metrics['ab_mcc']:.4f}, bc_mcc: {test_metrics['bc_mcc']:.4f}"
            )

    # 训练结束后，将最优 path_acc 对应的完整指标写入文件
    if best_metrics is not None:
        metrics_path = os.path.join(save_dir, "best_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, ensure_ascii=False, indent=2)
        print(f"[Cascade] Best metrics saved to {metrics_path}")

    # 返回最优指标（包含 val/test），便于外部框架（如 Optuna）统一读取
    return best_metrics


def run(cfg):
    train(
        cfg,
        few_shot_k=getattr(cfg, "few_shot_k", None),
        few_shot_balance=getattr(cfg, "few_shot_balance", None),
    )


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    train(
        cfg,
        few_shot_k=getattr(cfg, "few_shot_k", None),
        few_shot_balance=getattr(cfg, "few_shot_balance", None),
    )
