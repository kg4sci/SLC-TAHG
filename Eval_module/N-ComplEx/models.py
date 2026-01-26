"""
NComplEx: N-ary ComplEx Model for Knowledge Graphs
核心思想：继承 ComplEx 基础，扩展支持N-ary路径，Batch式处理与Cascading结构兼容。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NComplEx(nn.Module):
    """
    ComplEx主模型，支持节点嵌入、关系嵌入，兼容N-ary路径输入
    """
    def __init__(self, num_entities, num_relations, embedding_dim, dropout=0.1, node_feat_dim=0, use_node_features=False, use_text_features=False, text_dim=0):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.use_node_features = use_node_features
        self.node_feat_dim = node_feat_dim
        self.use_text_features = use_text_features
        self.text_dim = text_dim

        # ComplEx实体、关系 均为复数（分别为实部/虚部两个 nn.Embedding）
        self.ent_re = nn.Embedding(num_entities, embedding_dim)
        self.ent_im = nn.Embedding(num_entities, embedding_dim)
        self.rel_re = nn.Embedding(num_relations, embedding_dim)
        self.rel_im = nn.Embedding(num_relations, embedding_dim)

        # 节点特征/文本特征投影
        if use_node_features and node_feat_dim > 0:
            self.node_feat_proj = nn.Linear(node_feat_dim, embedding_dim)
        else:
            self.node_feat_proj = None
        if use_text_features and text_dim > 0:
            self.text_proj = nn.Linear(text_dim, embedding_dim)
        else:
            self.text_proj = None

        self.dropout = nn.Dropout(dropout)
        self._init_params()

    def _init_params(self):
        nn.init.xavier_uniform_(self.ent_re.weight)
        nn.init.xavier_uniform_(self.ent_im.weight)
        nn.init.xavier_uniform_(self.rel_re.weight)
        nn.init.xavier_uniform_(self.rel_im.weight)
        if self.node_feat_proj is not None:
            nn.init.xavier_uniform_(self.node_feat_proj.weight)
        if self.text_proj is not None:
            nn.init.xavier_uniform_(self.text_proj.weight)

    def _fetch_entity_emb(self, idx, feat=None, text_feat=None):
        """返回实体的复数嵌入(real, imag)"""
        re = self.ent_re(idx)
        im = self.ent_im(idx)
        if self.use_node_features and feat is not None and self.node_feat_proj is not None:
            feat_proj = self.node_feat_proj(feat)
            re = re + feat_proj
            im = im + feat_proj
        if self.use_text_features and text_feat is not None and self.text_proj is not None:
            t_proj = self.text_proj(text_feat)
            re = re + t_proj
            im = im + t_proj
        return re, im

    def _score_complex(self, s_re, s_im, r_re, r_im, o_re, o_im):
        """按照 ComplEx 定义计算复数内积得分"""
        return (
            torch.sum(s_re * r_re * o_re, dim=-1) +
            torch.sum(s_im * r_re * o_im, dim=-1) +
            torch.sum(s_re * r_im * o_im, dim=-1) -
            torch.sum(s_im * r_im * o_re, dim=-1)
        )

    def forward(self, subj, rel, obj, subj_feat=None, obj_feat=None, text_feat=None):
        """
        subj, rel, obj: [B] int64 (index)
        subj_feat/obj_feat: [B, node_feat_dim]，可选
        text_feat: [B, text_dim]，可选（一般为三元组文本证据）
        返回三元组打分: [B]
        """
        s_re, s_im = self._fetch_entity_emb(subj, subj_feat, text_feat)
        o_re, o_im = self._fetch_entity_emb(obj, obj_feat, text_feat)
        r_re = self.rel_re(rel)
        r_im = self.rel_im(rel)
        return self._score_complex(s_re, s_im, r_re, r_im, o_re, o_im)

    def score_all_relations(self, subj, obj, subj_feat=None, obj_feat=None, text_feat=None):
        """返回所有关系下的得分 -> [B, num_relations]"""
        s_re, s_im = self._fetch_entity_emb(subj, subj_feat, text_feat)
        o_re, o_im = self._fetch_entity_emb(obj, obj_feat, text_feat)
        r_re = self.rel_re.weight  # [num_rel, dim]
        r_im = self.rel_im.weight
        s_re = s_re.unsqueeze(1)
        s_im = s_im.unsqueeze(1)
        o_re = o_re.unsqueeze(1)
        o_im = o_im.unsqueeze(1)
        r_re = r_re.unsqueeze(0)
        r_im = r_im.unsqueeze(0)
        scores = self._score_complex(s_re, s_im, r_re, r_im, o_re, o_im)
        return scores

    def relation_embedding_from_weights(self, weights):
        """根据 one-hot/概率分布 weights 生成对应的关系嵌入(real, imag)"""
        rel_weight = weights.to(self.rel_re.weight.dtype)
        r_re = rel_weight @ self.rel_re.weight
        r_im = rel_weight @ self.rel_im.weight
        return r_re, r_im

    def score_all_relations_with_context(self, subj, obj, context_re=None, context_im=None, subj_feat=None, obj_feat=None, text_feat=None):
        """
        在实体嵌入中注入上下文（如由 rel_AB 预测得到的嵌入），返回所有关系得分。
        context_* 形状为 [B, embedding_dim]。
        """
        s_re, s_im = self._fetch_entity_emb(subj, subj_feat, text_feat)
        o_re, o_im = self._fetch_entity_emb(obj, obj_feat, text_feat)
        if context_re is not None and context_im is not None:
            s_re = s_re + context_re
            s_im = s_im + context_im
            o_re = o_re + context_re
            o_im = o_im + context_im
        r_re = self.rel_re.weight.unsqueeze(0)
        r_im = self.rel_im.weight.unsqueeze(0)
        s_re = s_re.unsqueeze(1)
        s_im = s_im.unsqueeze(1)
        o_re = o_re.unsqueeze(1)
        o_im = o_im.unsqueeze(1)
        return self._score_complex(s_re, s_im, r_re, r_im, o_re, o_im)


class NComplExPredictor(nn.Module):
    """
    NComplEx分类头，支持N-ary后级联（输入三元组/路径的复数embedding向量）
    """
    def __init__(self, embedding_dim, num_relation_classes, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_relation_classes = num_relation_classes
        self.classifier = nn.Sequential(
            nn.Linear(num_relation_classes, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_relation_classes),
        )

    def forward(self, relation_scores):
        """relation_scores: [B, num_relation_classes]"""
        return self.classifier(relation_scores)
