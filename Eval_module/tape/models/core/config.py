import os
import argparse
from yacs.config import CfgNode as CN


def set_cfg(cfg):

    # ------------------------------------------------------------------------ #
    # Path options (absolute base for outputs/embeddings)
    # ------------------------------------------------------------------------ #
    cfg.paths = CN()
    cfg.paths.output_base = "/mnt/data/lxy/benchmark_paper/Eval_module/tape/models/output_data"

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    # Dataset name
    # 新的级联预测任务使用基于 SLC 图的路径数据集名称
    cfg.dataset = 'SLC_database'
    # Cuda device number, used for machine with multiple gpus
    cfg.device = 2
    # Whether fix the running seed to remove randomness
    # cfg.seed = None# ************************更改
    cfg.seed = 0
    # Number of runs with random init
    cfg.runs = 4
    cfg.gnn = CN()
    cfg.lm = CN()

    # ------------------------------------------------------------------------ #
    # GNN Model options
    # ------------------------------------------------------------------------ #
    cfg.gnn.model = CN()
    # GNN model name
    cfg.gnn.model.name = 'GCN'
    # Number of gnn layers
    cfg.gnn.model.num_layers = 4
    # Hidden size of the model
    cfg.gnn.model.hidden_dim = 128

    # ------------------------------------------------------------------------ #
    # GNN Training options
    # ------------------------------------------------------------------------ #
    cfg.gnn.train = CN()
    # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights
    cfg.gnn.train.weight_decay = 0.0
    # Maximal number of epochs
    cfg.gnn.train.epochs = 200
    # Node feature type, options: ogb, TA, P, E
    cfg.gnn.train.feature_type = 'TA_P_E'
    # Number of epochs with no improvement after which training will be stopped
    cfg.gnn.train.early_stop = 50
    # Base learning rate
    cfg.gnn.train.lr = 0.01
    # L2 regularization, weight decay
    cfg.gnn.train.wd = 0.0
    # Dropout rate
    cfg.gnn.train.dropout = 0.0

    # ------------------------------------------------------------------------ #
    # LM Model options
    # ------------------------------------------------------------------------ #
    cfg.lm.model = CN()
    # LM model name
    # cfg.lm.model.name = 'sentence-transformers/multi-qa-distilbert-cos-v1'**************更改
    cfg.lm.model.name = '/mnt/data/lxy/benchmark_paper/Eval_module/tape/models/multi-qa-distilbert-cos-v1'
    cfg.lm.model.feat_shrink = ""

    # ------------------------------------------------------------------------ #
    # LM Training options
    # ------------------------------------------------------------------------ #
    cfg.lm.train = CN()
    #  Number of samples computed once per batch per device
    cfg.lm.train.batch_size = 128
    # Number of training steps for which the gradients should be accumulated
    cfg.lm.train.grad_acc_steps = 1
    # Base learning rate
    cfg.lm.train.lr = 2e-5
    # Maximal number of epochs
    cfg.lm.train.epochs = 10
    # The number of warmup steps
    cfg.lm.train.warmup_epochs = 0.6
    # Number of update steps between two evaluations
    cfg.lm.train.eval_patience = 50000
    # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights
    cfg.lm.train.weight_decay = 0.0
    # The dropout probability for all fully connected layers in the embeddings, encoder, and pooler
    cfg.lm.train.dropout = 0.3
    # The dropout ratio for the attention probabilities
    cfg.lm.train.att_dropout = 0.1
    # The dropout ratio for the classifier
    cfg.lm.train.cla_dropout = 0.4
    # Whether or not to use the gpt responses (i.e., explanation and prediction) as text input
    # If not, use the original text attributes (i.e., title and abstract)
    cfg.lm.train.use_gpt = True

    # ------------------------------------------------------------------------ #
    # Add handling for missing 1-shot, 3-shot, and 5-shot values
    # ------------------------------------------------------------------------ #
    cfg.dataset_params = CN()
    cfg.dataset_params.shot_1 = None  # 可以用默认值填充
    cfg.dataset_params.shot_3 = None
    cfg.dataset_params.shot_5 = None

    cfg.few_shot_k = 1

    # ------------------------------------------------------------------------ #
    # Cascading path prediction (AB -> BC) options
    # ------------------------------------------------------------------------ #
    cfg.cascade = CN()
    # Embedding dim for node ids
    cfg.cascade.emb_dim = 128
    # Hidden dim inside stage MLP / cond proj
    cfg.cascade.hidden_dim = 256
    # Dropout
    cfg.cascade.dropout = 0.1
    # Batch size for cascading training
    cfg.cascade.batch_size = 512
    # Learning rate
    cfg.cascade.lr = 1e-3
    # Epochs
    cfg.cascade.epochs = 100
    # Validation frequency (epochs)
    cfg.cascade.val_every = 1
    # Whether to use text-based node features from LM (AB/BC 证据等)
    # True: 使用 LM 产生的基于文本的特征（默认）
    # False: 忽略这些文本特征，只依赖其他节点特征或纯 ID embedding
    cfg.cascade.use_text_feature = True
    # Optional node feature file (torch.save Tensor [num_nodes, feat_dim])
    cfg.cascade.node_feat_path = ""
    # Whether to use node features if provided
    cfg.cascade.use_node_feat = True
    # Optional GNN node embedding path (torch.load or memmap float16), will be added to base feat
    cfg.cascade.gnn_feat_path = f"{cfg.paths.output_base}/gnn_emb.pt"
    # LM base path for cascading to locate LM embeddings
    cfg.cascade.lm_base = f"{cfg.paths.output_base}"

    # ------------------------------------------------------------------------ #
    # GNN export options (for cascading to reuse GNN node embeddings)
    # ------------------------------------------------------------------------ #
    cfg.gnn.export_emb_path = f"{cfg.paths.output_base}/gnn_emb.pt"  # 如果非空，将在 gnn_trainer.eval_and_save 时导出 logits 作为节点表示

    return cfg


# Principle means that if an option is defined in a YACS config object,
# then your program should set that configuration option using cfg.merge_from_list(opts) and not by defining,
# for example, --train-scales as a command line argument that is then used to set cfg.TRAIN.SCALES.


def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="",
                        metavar="FILE", help="Path to config file")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args()
    # Clone the original cfg
    cfg = cfg.clone()

    # Update from config file
    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    # Update from command line
    cfg.merge_from_list(args.opts)

    return cfg


"""
    Global variable
"""
cfg = set_cfg(CN())
