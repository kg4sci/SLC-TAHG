import logging
import logging.config

import inspect
import numpy as np

# PyTorch related imports
import torch
from torch.nn import Parameter
from torch.nn.init import xavier_normal_

np.set_printoptions(precision=4)


class MessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),

    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://rusty1s.github.io/pytorch_geometric/build/html/notes/
    create_gnn.html>`__ for the accompanying tutorial.

    """

    def __init__(self, aggr='add'):
        super(MessagePassing, self).__init__()

        # In the defined message function: get the list of arguments as list of string|
        # For eg. in r-gcn this will be ['x_j', 'edge_type', 'edge_norm'] (args of message fn)
        self.message_args = inspect.getargspec(self.message)[0][1:]
        # Same for update function starting from 3rd argument | first=self, second=out
        self.update_args = inspect.getargspec(self.update)[0][2:]

    def propagate(self, aggr, edge_index, **kwargs):
        r"""The initial call to start propagating messages.
        Takes in an aggregation scheme (:obj:`"add"`, :obj:`"mean"` or
        :obj:`"max"`), the edge indices, and all additional data which is
        needed to construct messages and to update node embeddings."""

        assert aggr in ['add', 'mean', 'max']
        kwargs['edge_index'] = edge_index

        size = None
        message_args = []
        for arg in self.message_args:
            if arg[-2:] == '_i':  # If arguments ends with _i then include indic
                tmp = kwargs[
                    arg[:-2]]  # Take the front part of the variable | Mostly it will be 'x',
                size = tmp.size(0)
                message_args.append(tmp[edge_index[0]])  # Lookup for head entities in edges
            elif arg[-2:] == '_j':
                tmp = kwargs[arg[:-2]]  # tmp = kwargs['x']
                size = tmp.size(0)
                message_args.append(tmp[edge_index[1]])  # Lookup for tail entities in edges
            else:
                message_args.append(kwargs[arg])  # Take things from kwargs

        update_args = [kwargs[arg] for arg in self.update_args]  # Take update args from kwargs

        out = self.message(*message_args)
        out = scatter_(aggr, out, edge_index[0],
                       dim_size=size)  # Aggregated neighbors for each vertex
        out = self.update(out, *update_args)

        return out

    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
        for each edge in :math:`(i,j) \in \mathcal{E}`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, features can be lifted to the source node :math:`i` and
        target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
        variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""

        return x_j

    def update(self, aggr_out):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`."""

        return aggr_out

def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


def _infer_dim_size(index, dim_size):
    if dim_size is not None:
        return dim_size
    if index.numel() == 0:
        return 0
    return int(index.max().item()) + 1


def _expand_index(index, src, dim):
    if dim != 0:
        raise NotImplementedError("scatter operations currently support dim=0 only.")
    if index.dim() != 1:
        raise ValueError("index must be a 1D tensor.")
    if src.size(dim) != index.size(0):
        raise ValueError("index length must match src.size(dim).")

    view_shape = [index.size(0)] + [1] * (src.dim() - 1)
    return index.view(view_shape).expand_as(src)


def scatter_add(src, index, dim=0, dim_size=None):
    dim_size = _infer_dim_size(index, dim_size)
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.zeros(out_shape, device=src.device, dtype=src.dtype)

    if src.numel() == 0:
        return out

    expanded_index = _expand_index(index, src, dim)
    out.scatter_add_(dim, expanded_index, src)
    return out


def scatter_mean(src, index, dim=0, dim_size=None):
    out = scatter_add(src, index, dim=dim, dim_size=dim_size)
    if src.numel() == 0:
        return out

    counts = scatter_add(
        torch.ones(index.size(0), device=src.device, dtype=src.dtype),
        index,
        dim=0,
        dim_size=_infer_dim_size(index, dim_size)
    )
    while counts.dim() < out.dim():
        counts = counts.unsqueeze(-1)
    return out / counts.clamp(min=1)


def scatter_max(src, index, dim=0, dim_size=None, fill_value=None):
    dim_size = _infer_dim_size(index, dim_size)
    out_shape = list(src.shape)
    out_shape[dim] = dim_size

    if fill_value is None:
        if src.is_floating_point():
            fill_value = torch.finfo(src.dtype).min
        else:
            fill_value = torch.iinfo(src.dtype).min

    out = torch.full(out_shape, fill_value, device=src.device, dtype=src.dtype)

    if src.numel() == 0:
        argmax = torch.full(out_shape, -1, device=index.device, dtype=torch.long)
        return out, argmax

    expanded_index = _expand_index(index, src, dim)
    out.scatter_reduce_(dim, expanded_index, src, reduce='amax', include_self=True)
    argmax = torch.full(out_shape, -1, device=index.device, dtype=torch.long)
    return out, argmax


def softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (
        scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out

def get_param(shape):
    param = Parameter(torch.Tensor(*shape))
    xavier_normal_(param.data)
    return param


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def cconv(a, b):
    return torch.irfft(com_mult(torch.rfft(a, 1), torch.rfft(b, 1)), 1,
                       signal_sizes=(a.shape[-1],))


# def ccorr(a, b):
#     return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1,
#                        signal_sizes=(a.shape[-1],))
def ccorr(a, b):
    """
    使用新版 PyTorch (torch.fft) 实现的 Circular Correlation
    """
    # 1. 将输入转为频域 (Real-to-Complex FFT)
    # dim=-1 表示在最后一个维度进行变换
    a_fft = torch.fft.rfft(a, dim=-1)
    b_fft = torch.fft.rfft(b, dim=-1)
    
    # 2. 在频域进行共轭乘法
    # 对应原代码中的: com_mult(conj(fft(a)), fft(b))
    # 新版 PyTorch 支持复数直接相乘
    frequency_product = torch.conj(a_fft) * b_fft
    
    # 3. 逆变换回时域 (Complex-to-Real IFFT)
    # n=a.shape[-1] 确保输出维度与输入一致
    res = torch.fft.irfft(frequency_product, n=a.shape[-1], dim=-1)
    
    return res

def rotate(h, r):
    # re: first half, im: second half
    # assume embedding dim is the last dimension
    d = h.shape[-1]
    h_re, h_im = torch.split(h, d // 2, -1)
    r_re, r_im = torch.split(r, d // 2, -1)
    return torch.cat([h_re * r_re - h_im * r_im,
                        h_re * r_im + h_im * r_re], dim=-1)


def scatter_(name, src, index, dim_size=None):
    r"""Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).

    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    assert name in ['add', 'mean', 'max']

    fill_value = -1e38 if name == 'max' else 0

    if name == 'add':
        out = scatter_add(src, index, dim=0, dim_size=dim_size)
    elif name == 'mean':
        out = scatter_mean(src, index, dim=0, dim_size=dim_size)
    else:  # max
        out = scatter_max(src, index, dim=0, dim_size=dim_size, fill_value=fill_value)
    if isinstance(out, tuple):
        out = out[0]

    if name == 'max':
        out[out == fill_value] = 0

    return out
