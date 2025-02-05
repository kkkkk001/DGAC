# -*- coding: utf-8 -*-


import torch
import numpy as np


def numpy_to_torch(a, is_sparse=False):
    """
    numpy array to torch tensor
    :param a: the numpy array
    :param is_sparse: is sparse tensor or not
    :return a: torch tensor
    """
    if is_sparse:
        a = torch.sparse.Tensor(a)
    else:
        a = torch.from_numpy(a)
    return a


def torch_to_numpy(t):
    """
    torch tensor to numpy array
    :param t: the torch tensor
    :return t: numpy array
    """
    return t.numpy()


def normalize_adj(adj, self_loop=True, symmetry=True):
    """
    normalize the adj matrix
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return norm_adj: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        adj_tmp = adj + np.eye(adj.shape[0])
    else:
        adj_tmp = adj

    # calculate degree matrix and it's inverse matrix
    d = adj_tmp.sum(0)
    d[d==0] = 1
    d = np.diag(d)
    d_inv = np.linalg.inv(d)

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)

    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = np.matmul(d_inv, adj_tmp)

    return norm_adj


def normalize_adj_torch(adj, self_loop=True, symmetry=True):
    """
    normalize the adj matrix (torch version)
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return norm_adj: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        adj_tmp = adj + torch.eye(adj.shape[0], device=adj.device)
    else:
        adj_tmp = adj

    # calculate degree matrix and it's inverse matrix
    deg_vec = adj_tmp.sum(0)
    deg_vec[deg_vec == 0] = 1


    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        deg_vec.pow_(-0.5)
        norm_adj = deg_vec.unsqueeze(0) * adj_tmp * deg_vec.unsqueeze(1)

    # non-symmetry normalize: D^{-1} A
    else:
        deg_vec.pow_(-1)
        norm_adj = deg_vec.unsqueeze(0) * adj_tmp

    return norm_adj


import torch

def add_self_loops(adj):
    # 获取稀疏矩阵的索引和数据
    indices = adj._indices()
    values = adj._values()
    
    # 创建自环的索引和数据
    num_nodes = adj.size(0)
    self_loop_indices = torch.arange(0, num_nodes, device=adj.device, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    self_loop_values = torch.ones(num_nodes, device=adj.device)
    
    # 合并原始索引和自环索引
    new_indices = torch.cat([indices, self_loop_indices], dim=1)
    new_values = torch.cat([values, self_loop_values])
    
    # 创建新的稀疏矩阵
    new_adj = torch.sparse_coo_tensor(new_indices, new_values, adj.size(), device=adj.device)
    
    return new_adj

def normalize_adj_torch_sparse(adj, self_loop=True, symmetry=True):
    """
    normalize the adj matrix (torch sparse version)
    :param adj: input adj matrix (torch sparse tensor)
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return norm_adj: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        adj = add_self_loops(adj)
        # adj = adj + torch.eye(adj.shape[0], device=adj.device).to_sparse()


    # calculate degree matrix and its inverse matrix
    deg_vec = torch.sparse.sum(adj, dim=0).to_dense()
    deg_vec[deg_vec == 0] = 1

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        deg_vec = deg_vec.pow(-0.5)
        D_inv_sqrt = torch.diag(deg_vec)
        norm_adj = torch.sparse.mm(torch.sparse.mm(D_inv_sqrt.to_sparse(), adj), D_inv_sqrt.to_sparse())

    # non-symmetry normalize: D^{-1} A
    else:
        deg_vec = deg_vec.pow(-1)
        D_inv = torch.diag(deg_vec)
        norm_adj = torch.sparse.mm(D_inv.to_sparse(), adj)

    return norm_adj



def construct_graph(feat, k=5, metric="euclidean"):
    """
    construct the knn graph for a non-graph dataset
    :param feat: the input feature matrix
    :param k: hyper-parameter of knn
    :param metric: the metric of distance calculation
    - euclidean: euclidean distance
    - cosine: cosine distance
    - heat: heat kernel
    :return knn_graph: the constructed graph
    """

    # euclidean distance, sqrt((x-y)^2)
    if metric == "euclidean" or metric == "heat":
        xy = np.matmul(feat, feat.transpose())
        xx = (feat * feat).sum(1).reshape(-1, 1)
        xx_yy = xx + xx.transpose()
        euclidean_distance = xx_yy - 2 * xy
        euclidean_distance[euclidean_distance < 1e-5] = 0
        distance_matrix = np.sqrt(euclidean_distance)

        # heat kernel, exp^{- euclidean^2/t}
        if metric == "heat":
            distance_matrix = - (distance_matrix ** 2) / 2
            distance_matrix = np.exp(distance_matrix)

    # cosine distance, 1 - cosine similarity
    if metric == "cosine":
        norm_feat = feat / np.sqrt(np.sum(feat ** 2, axis=1)).reshape(-1, 1)
        cosine_distance = 1 - np.matmul(norm_feat, norm_feat.transpose())
        cosine_distance[cosine_distance < 1e-5] = 0
        distance_matrix = cosine_distance

    # top k
    distance_matrix = numpy_to_torch(distance_matrix)
    top_k, index = torch.topk(distance_matrix, k)
    top_k_min = torch.min(top_k, dim=-1).values.unsqueeze(-1).repeat(1, distance_matrix.shape[-1])
    ones = torch.ones_like(distance_matrix)
    zeros = torch.zeros_like(distance_matrix)
    knn_graph = torch.where(torch.ge(distance_matrix, top_k_min), ones, zeros)
    knn_graph = torch_to_numpy(knn_graph)

    return knn_graph


def construct_filter(adj, l=5, alpha=0.5):
    """
    torch version
    construct the low-pass filter: \sum_0^l alpha^l A^l
    :param adj: the input adj matrix
    :param l: the order of filter
    :param alpha: the hyper-parameter of filter
    :return low_pass_filter
    """
    I = torch.eye(adj.shape[0]).to(adj.device)
    low_pass = I
    for i in range(l):
        low_pass = alpha * torch.mm(low_pass, adj) + I

    return low_pass


